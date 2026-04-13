#!/usr/bin/env python3
import sys
import os
import getopt
import subprocess
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import zipfile
import io
from torch.utils.data import Dataset
from shallow_learning.deepl import VAE, GAN, DiffusionModel, GenModelTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CelebAZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path  = zip_path
        self.transform = transform
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.image_names = sorted([
                name for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# GPU
# ─────────────────────────────────────────────────────────────────────────────

def get_best_gpu(strategy="utilization"):
    import subprocess
    if strategy == "memory":
        free_mem = []
        for i in range(torch.cuda.device_count()):
            free_mem.append(torch.cuda.mem_get_info(i)[0])
        return free_mem.index(max(free_mem))
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv):
    # ── Defaults ──────────────────────────────────────────────────────────────
    model_type   = "VAE"
    epochs       = 10
    train_ratio  = 0.10
    batch_size   = 128
    lr           = 2e-4
    save_every   = 10
    zip_path     = "/data/CPE_487-587/img_align_celeba.zip"
    save_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "output_genmodel")

    try:
        opts, args = getopt.getopt(argv, "hm:e:t:b:s:",
                                   ["help", "model=", "epochs=", "train_ratio=",
                                    "batch_size=", "save_every=", "lr="])
    except getopt.GetoptError:
        print("Usage: genmodel_impl.py -m <VAE|GAN|Diffusion> -e <epochs> "
              "-t <train_ratio> -b <batch_size> -s <save_every> --lr <lr>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("Usage: genmodel_impl.py -m <VAE|GAN|Diffusion> -e <epochs> "
                  "-t <train_ratio> -b <batch_size> -s <save_every> --lr <lr>")
            sys.exit()
        elif opt in ('-m', '--model'):
            model_type = arg
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        elif opt in ('-t', '--train_ratio'):
            train_ratio = float(arg)
        elif opt in ('-b', '--batch_size'):
            batch_size = int(arg)
        elif opt in ('-s', '--save_every'):
            save_every = int(arg)
        elif opt in ('--lr',):
            lr = float(arg)

    # ── GPU ───────────────────────────────────────────────────────────────────
    device_id = get_best_gpu(strategy="utilization")
    device    = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: cuda:{device_id}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    print(f"Loading CelebA dataset from {zip_path}...")
    full_dataset = CelebAZipDataset(zip_path, transform=transform)
    subset_size  = int(len(full_dataset) * train_ratio)
    dataset      = Subset(full_dataset, range(subset_size))
    print(f"Total images: {len(full_dataset)}  Using: {subset_size} ({train_ratio*100:.1f}%)")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_type = model_type.strip()
    if model_type.upper() == "VAE":
        model = VAE(latent_dim=128)
    elif model_type.upper() == "GAN":
        model = GAN(latent_dim=128)
    elif model_type.upper() in ("DIFFUSION", "DIFF"):
        model = DiffusionModel(T=1000)
    else:
        print(f"Unknown model type: {model_type}. Choose VAE, GAN, or Diffusion.")
        sys.exit(2)

    print(f"\nModel: {model_type}")
    print(f"Epochs: {epochs}  Batch size: {batch_size}  "
          f"LR: {lr}  Save every: {save_every} epochs")

    # ── Trainer ───────────────────────────────────────────────────────────────
    model_save_dir = os.path.join(save_dir, model_type.lower())
    os.makedirs(model_save_dir, exist_ok=True)

    trainer = GenModelTrainer(
        model=model,
        device=device,
        lr=lr,
        save_dir=model_save_dir
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train(dataloader, epochs=epochs, save_every=save_every)

    print(f"\nDone! Outputs saved to {model_save_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
