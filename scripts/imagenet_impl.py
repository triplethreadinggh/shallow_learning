#!/usr/bin/env python3
import sys
import os
import getopt
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote machine
import matplotlib.pyplot as plt
from datasets import load_dataset

from shallow_learning.deepl import ImageNetCNN, CNNTrainer


def get_best_gpu(strategy="utilization"):
    if strategy == "memory":
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i)
            free_mem.append(props[0])
        return free_mem.index(max(free_mem))
    elif strategy == "utilization":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def preprocess_train(examples):
    images = [train_transform(img.convert('RGB')) for img in examples['image']]
    return {'pixel_values': images, 'labels': examples['label']}


def preprocess_val(examples):
    images = [val_transform(img.convert('RGB')) for img in examples['image']]
    return {'pixel_values': images, 'labels': examples['label']}


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


def save_sample_image(dataset_split, class_names, filename, title_prefix):
    example  = dataset_split[0]
    image    = example['image']
    label_id = example['label']
    full_label   = class_names[label_id]
    primary_name = full_label.split(',')[0].strip()

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"{title_prefix} — ID {label_id}: {primary_name}\n({full_label})", fontsize=9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Sample image saved to {filename}")


def main(argv):
    # ── Defaults ──────────────────────────────────────────────────────────────
    epochs      = 10000
    train_ratio = 0.10
    val_ratio   = 0.05
    batch_size  = 128
    dropout     = 0.5

    # ── CLI args (mirrors multiclass_impl.py style) ───────────────────────────
    try:
        opts, args = getopt.getopt(argv, "he:t:v:",
                                   ["help", "epochs=", "train_ratio=", "val_ratio=",
                                    "batch_size=", "dropout="])
    except getopt.GetoptError:
        print('Usage: imagenet_impl.py -e <epochs> -t <train_ratio> -v <val_ratio> '
              '[--batch_size <n>] [--dropout <f>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Usage: imagenet_impl.py -e <epochs> -t <train_ratio> -v <val_ratio> '
                  '[--batch_size <n>] [--dropout <f>]')
            sys.exit()
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        elif opt in ('-t', '--train_ratio'):
            train_ratio = float(arg)
        elif opt in ('-v', '--val_ratio'):
            val_ratio = float(arg)
        elif opt in ('--batch_size',):
            batch_size = int(arg)
        elif opt in ('--dropout',):
            dropout = float(arg)

    # ── GPU ───────────────────────────────────────────────────────────────────
    device_id = get_best_gpu(strategy="utilization")
    device    = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: cuda:{device_id}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading ImageNet dataset...")
    from datasets import load_from_disk
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    class_names = dataset['train'].features['label'].names
    num_classes = len(class_names)
    print(f"Number of classes:              {num_classes}")
    print(f"Original train size:            {len(dataset['train'])}")
    print(f"Original validation size:       {len(dataset['validation'])}")

    # ── Subset ────────────────────────────────────────────────────────────────
    train_size = int(len(dataset['train'])      * train_ratio)
    val_size   = int(len(dataset['validation']) * val_ratio)
    print(f"Using {train_size} train samples  ({train_ratio*100:.1f}%)")
    print(f"Using {val_size}   val samples    ({val_ratio*100:.1f}%)")

    raw_train = dataset['train'].select(range(train_size))
    raw_val   = dataset['validation'].select(range(val_size))

    # ── Save sample images (raw PIL, before transforms) ───────────────────────
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_imagenet_cnn")
    os.makedirs(scripts_dir, exist_ok=True)
    save_sample_image(raw_train, class_names,
                      os.path.join(scripts_dir, "sample_train.png"), "Train")
    save_sample_image(raw_val, class_names,
                      os.path.join(scripts_dir, "sample_val.png"), "Validation")

    # ── Apply transforms ──────────────────────────────────────────────────────
    train_dataset = raw_train.with_transform(preprocess_train)
    val_dataset   = raw_val.with_transform(preprocess_val)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ImageNetCNN(num_classes=num_classes, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CNNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train(train_loader, val_loader, epochs=epochs)

    # ── Save ONNX ─────────────────────────────────────────────────────────────
    trainer.save_onnx(os.path.join(scripts_dir, "imagenet_cnn.onnx"))

    # ── Save plot ─────────────────────────────────────────────────────────────
    trainer.save_plot(os.path.join(scripts_dir, "training_plot.png"))


if __name__ == "__main__":
    main(sys.argv[1:])
