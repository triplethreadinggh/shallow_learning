#!/usr/bin/env python3
import sys
import getopt
from pathlib import Path
from datetime import datetime
import os
import csv
import torch
from shallow_learning.deepl.acc_classifier import ACCNet, ACCTrainer, build_dataloaders, get_best_device
 
def main(argv):
    data_dir = "/data/CPE_487-587/ACCDataset"
    epochs = 50
    learning_rate = 0.001
    batch_size = 256
    keyword = "acc"
 
    try:
        opts, args = getopt.getopt(argv, "hd:e:", ["help", "data=", "epochs=", "lr=", "batch=", "keyword="])
    except getopt.GetoptError:
        print("Usage: acc_impl.py --data <dir> --epochs <n> --lr <lr> --batch <n> --keyword <kw>")
        sys.exit(2)
 
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: acc_impl.py --data <dir> --epochs <n> --lr <lr> --batch <n> --keyword <kw>")
            sys.exit()
        elif opt in ("-d", "--data"):
            data_dir = arg
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("--lr",):
            learning_rate = float(arg)
        elif opt in ("--batch",):
            batch_size = int(arg)
        elif opt in ("--keyword",):
            keyword = arg
 
    device = get_best_device()
 
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, scaler = build_dataloaders(data_dir, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
 
    model = ACCNet(window_size=11)
    trainer = ACCTrainer(model, device, lr=learning_rate)
    print(f"\nTraining ACCNet for {epochs} epochs...\n")
 
    best_val_acc = 0.0
    metrics_rows = [["Epoch", "Train Loss", "Accuracy", "Precision", "Recall", "F1"]]
 
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        m = trainer.evaluate(val_loader)
        metrics_rows.append([epoch, round(train_loss, 6), round(m["accuracy"], 4),
                              round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4)])
        print(f"Epoch {epoch:>3}  loss={train_loss:.4f}  acc={m['accuracy']:.4f}  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}")
 
        if m["accuracy"] > best_val_acc:
            best_val_acc = m["accuracy"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "scaler": scaler, "val_acc": best_val_acc},
                       "scripts/output_acc/accnet_best.pt")
 
    # save metrics csv
    os.makedirs("scripts/output_acc", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"scripts/output_acc/metrics_{keyword}_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(metrics_rows)
    print(f"\nMetrics saved to {csv_path}")
 
    # export best model to ONNX
    ckpt = torch.load("scripts/output_acc/accnet_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    onnx_path = f"scripts/output_acc/accnet_{keyword}.onnx"
    trainer.save_onnx(onnx_path, scaler)
    print(f"Best val accuracy: {best_val_acc:.4f}")
 
if __name__ == "__main__":
    main(sys.argv[1:])
