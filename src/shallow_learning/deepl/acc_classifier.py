"""
ACC state classifier — predict if adaptive cruise control is active (1) or not (0)
based on a sliding window of front-left wheel speed readings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# --- Zero-Order Hold: assign each speed row the most recent label <= its timestamp ---

def zero_order_hold(speed_times: np.ndarray, label_times: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    aligned = np.full(len(speed_times), -1, dtype=np.int64)
    j = 0
    for i, t in enumerate(speed_times):
        while j + 1 < len(label_times) and label_times[j + 1] <= t:
            j += 1
        if label_times[j] <= t:
            aligned[i] = label_values[j]
    return aligned


# --- Custom layer: appends finite differences (acceleration signal) to speed window ---

class TemporalDeltaLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return torch.cat([x, x - x_prev], dim=1)


# --- ACCNet: MLP classifier ---

class ACCNet(nn.Module):
    def __init__(self, window_size: int = 11, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.feature_layer = TemporalDeltaLayer()
        self.net = nn.Sequential(
            nn.Linear(window_size * 2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.feature_layer(x))


# --- Focal Tversky Loss: handles class imbalance by penalising false negatives more ---

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p, t = preds.view(-1), targets.view(-1).float()
        tp = (p * t).sum()
        tversky = (tp + 1e-6) / (tp + self.alpha * (p * (1 - t)).sum() + self.beta * ((1 - p) * t).sum() + 1e-6)
        return (1 - tversky) ** self.gamma


# --- Dataset: loads all CSV pairs, applies ZOH, builds sliding windows ---

class ACCDataset(Dataset):
    def __init__(self, data_dir: str | Path, k: int = 10, scaler_params: Optional[Dict] = None):
        data_dir = Path(data_dir)
        speed_files = sorted(data_dir.glob("*_CAN_Messages_decoded_wheel_speed_fl.csv"))
        if not speed_files:
            raise FileNotFoundError(f"No speed CSVs found in {data_dir}")

        all_speeds, all_labels = [], []
        for sf in speed_files:
            prefix = sf.name.replace("_wheel_speed_fl.csv", "")
            lf = data_dir / f"{prefix}_acc_status.csv"
            if not lf.exists():
                print(f"[WARN] No label file for {sf.name}, skipping.")
                continue
            speeds, labels = self._load_pair(sf, lf)
            if speeds is not None:
                all_speeds.append(speeds)
                all_labels.append(labels)

        raw_speed = np.concatenate(all_speeds)
        raw_label = np.concatenate(all_labels)

        # z-score normalise speed
        if scaler_params:
            self.scaler_params = scaler_params
        else:
            self.scaler_params = {"mean": float(raw_speed.mean()), "std": float(raw_speed.std()) + 1e-8}
        raw_speed = (raw_speed - self.scaler_params["mean"]) / self.scaler_params["std"]

        # build sliding windows: row i = [v_t, v_{t-1}, ..., v_{t-k}]
        N = len(raw_speed)
        idx = np.arange(k, N)[:, None] - np.arange(k + 1)[None, :]
        self.X = torch.from_numpy(raw_speed[idx].astype(np.float32))
        self.y = torch.from_numpy(raw_label[k:].astype(np.float32))

    def _load_pair(self, sf: Path, lf: Path):
        try:
            spd = pd.read_csv(sf, usecols=["Time", "Message"])
            lbl = pd.read_csv(lf, usecols=["Time", "Message", "Bus"])
        except Exception as e:
            print(f"[ERROR] {sf.name}: {e}")
            return None, None

        spd = spd.dropna().sort_values("Time")
        spd["Message"] = pd.to_numeric(spd["Message"], errors="coerce") / 3.6  # km/h -> m/s

        # drop duplicate timestamps by keeping Bus=0 only
        lbl = lbl[lbl["Bus"] == 0].dropna().sort_values("Time")
        lbl["Message"] = (pd.to_numeric(lbl["Message"], errors="coerce").astype(int) == 6).astype(int)

        aligned = zero_order_hold(spd["Time"].values, lbl["Time"].values, lbl["Message"].values)
        valid = aligned >= 0
        return spd["Message"].values[valid].astype(np.float32), aligned[valid].astype(np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# --- Trainer: training loop, evaluation, and ONNX export ---

class ACCTrainer:
    def __init__(self, model: ACCNet, device: torch.device, lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalTverskyLoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X).squeeze(1), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(X)
        self.scheduler.step()
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, threshold: float = 0.5) -> Dict:
        self.model.eval()
        all_p, all_y = [], []
        for X, y in loader:
            all_p.append(self.model(X.to(self.device)).squeeze(1).cpu())
            all_y.append(y)
        preds, labels = torch.cat(all_p), torch.cat(all_y)
        binary = (preds >= threshold).float()
        tp = ((binary == 1) & (labels == 1)).sum().item()
        fp = ((binary == 1) & (labels == 0)).sum().item()
        fn = ((binary == 0) & (labels == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        return {
            "accuracy":  (binary == labels).float().mean().item(),
            "precision": precision,
            "recall":    recall,
            "f1":        2 * precision * recall / (precision + recall + 1e-8),
        }

    def save_onnx(self, path: str | Path, scaler_params: Dict, window_size: int = 11):
        self.model.eval()
        path = Path(path)
        torch.onnx.export(
            self.model, torch.zeros(1, window_size, device=self.device), str(path),
            input_names=["speed_window"], output_names=["acc_probability"],
            dynamic_axes={"speed_window": {0: "batch"}, "acc_probability": {0: "batch"}},
            opset_version=17,
        )
        path.with_suffix(".scaler.json").write_text(json.dumps(scaler_params, indent=2))
        print(f"Saved ONNX → {path}")


# --- pick GPU with most free memory ---

def get_best_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best = max(range(torch.cuda.device_count()),
               key=lambda i: torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i))
    print(f"[Device] GPU {best}: {torch.cuda.get_device_properties(best).name}")
    return torch.device(f"cuda:{best}")


# --- build train/val dataloaders ---

def build_dataloaders(data_dir: str | Path, k: int = 10, batch_size: int = 256,
                      val_split: float = 0.2, num_workers: int = 2, seed: int = 42):
    ds = ACCDataset(data_dir, k=k)
    n_val = int(len(ds) * val_split)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed)
    )
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        ds.scaler_params,
    )
