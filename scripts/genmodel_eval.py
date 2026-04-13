#!/usr/bin/env python3
import sys
import os
import getopt
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from shallow_learning.deepl import VAE, GAN, DiffusionModel


# ─────────────────────────────────────────────────────────────────────────────
# Image Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(img_np):
    """Convert (H, W, 3) float image to grayscale (H, W)."""
    return 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]


def variance_of_laplacian(gray):
    """VoL — second-order sharpness measure."""
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    lap = convolve(gray, kernel)
    return float(np.var(lap))


def tenengrad(gray):
    """TEN — first-order edge strength measure."""
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    from scipy.ndimage import convolve
    gx = convolve(gray, sx)
    gy = convolve(gray, sy)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.var(mag))


def high_freq_energy(gray, alpha=0.1):
    """HFE — ratio of high-frequency energy in Fourier domain."""
    H, W  = gray.shape
    f     = np.fft.fftshift(np.fft.fft2(gray))
    mag   = np.abs(f)
    r     = int(alpha * min(W, H))
    cy, cx = H // 2, W // 2
    Y, X  = np.ogrid[:H, :W]
    mask  = (X - cx)**2 + (Y - cy)**2 <= r**2
    total = mag.sum()
    if total == 0:
        return 0.0
    return float(mag[~mask].sum() / total)


def mean_local_std(gray, w=7):
    """MLSD — mean local standard deviation, texture measure."""
    mu   = uniform_filter(gray, size=w)
    mu2  = uniform_filter(gray**2, size=w)
    var  = np.maximum(0, mu2 - mu**2)
    return float(np.mean(np.sqrt(var)))


def glcm_contrast(gray, Q=64, d=1, angle=0):
    """GLCM Contrast — co-occurrence based texture measure."""
    iq = (gray * (Q - 1)).astype(np.int32).clip(0, Q - 1)
    dx = int(round(d * np.cos(angle)))
    dy = int(round(d * np.sin(angle)))
    H, W = iq.shape
    C = np.zeros((Q, Q), dtype=np.float64)
    # Shifted image
    y1 = max(0, dy);  y2 = min(H, H + dy)
    x1 = max(0, dx);  x2 = min(W, W + dx)
    sy1 = max(0, -dy); sy2 = min(H, H - dy)
    sx1 = max(0, -dx); sx2 = min(W, W - dx)
    i_vals = iq[y1:y2, x1:x2].ravel()
    j_vals = iq[sy1:sy2, sx1:sx2].ravel()
    for i, j in zip(i_vals, j_vals):
        C[i, j] += 1
    # Symmetrize and normalize
    C = (C + C.T) / 2
    total = C.sum()
    if total == 0:
        return 0.0
    P = C / total
    i_idx, j_idx = np.meshgrid(np.arange(Q), np.arange(Q), indexing='ij')
    return float(np.sum((i_idx - j_idx)**2 * P))


def compute_metrics(img_tensor):
    """
    Compute all 5 metrics for a single image tensor (C, H, W) in [-1, 1].
    Returns dict of metric name -> value.
    """
    # Convert to [0, 1] numpy HWC
    img_np = ((img_tensor.cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    gray   = to_grayscale(img_np)

    return {
        'VoL':  variance_of_laplacian(gray),
        'TEN':  tenengrad(gray),
        'HFE':  high_freq_energy(gray),
        'MLSD': mean_local_std(gray),
        'GLCM': glcm_contrast(gray),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load model and generate samples
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_sample(model_type, onnx_dir, device, n=25):
    """Load trained PyTorch model and generate n samples."""
    if model_type == "VAE":
        model = VAE(latent_dim=128).to(device)
        ckpt  = os.path.join(onnx_dir, "vae", "vae_model.onnx")
    elif model_type == "GAN":
        model = GAN(latent_dim=128).to(device)
        ckpt  = os.path.join(onnx_dir, "gan", "gan_model.onnx")
    elif model_type == "Diffusion":
        model = DiffusionModel(T=1000).to(device)
        ckpt  = os.path.join(onnx_dir, "diffusion", "diffusion_model.onnx")

    # We sample directly from PyTorch model (ONNX is saved for submission)
    model.eval()
    with torch.no_grad():
        samples = model.sample(n, device)
    return samples


def load_pt_model(model_type, save_dir, device):
    """Load model weights from saved .pt checkpoint if available."""
    pt_path = os.path.join(save_dir, model_type.lower(), f"{model_type.lower()}_best.pt")
    if model_type == "VAE":
        model = VAE(latent_dim=128)
    elif model_type == "GAN":
        model = GAN(latent_dim=128)
    elif model_type == "Diffusion":
        model = DiffusionModel(T=1000)

    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location=device))
        print(f"Loaded weights from {pt_path}")
    else:
        print(f"No .pt checkpoint found at {pt_path}, using untrained model")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Save 25 sample images grid
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_grid(samples, model_type, save_dir):
    """Save 25 generated images as a 5x5 grid."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = ((samples[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(f"{model_type} — 25 Generated Samples", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_type.lower()}_samples.png")
    plt.savefig(path)
    plt.close()
    print(f"Sample grid saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_plot(all_metrics, save_dir):
    """
    Bar plot comparing all 5 metrics across VAE, GAN, Diffusion.
    all_metrics: dict of model_type -> dict of metric -> list of values
    """
    metric_names = ['VoL', 'TEN', 'HFE', 'MLSD', 'GLCM']
    model_names  = list(all_metrics.keys())
    x            = np.arange(len(metric_names))
    width        = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names):
        means = [np.mean(all_metrics[model_name][m]) for m in metric_names]
        stds  = [np.std(all_metrics[model_name][m])  for m in metric_names]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=model_name, capsize=4)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Metric Value")
    ax.set_title("Image Quality Metrics: VAE vs GAN vs Diffusion")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Comparison plot saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv):
    save_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "output_genmodel")
    n_samples = 25
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        opts, args = getopt.getopt(argv, "hs:n:",
                                   ["help", "save_dir=", "n_samples="])
    except getopt.GetoptError:
        print("Usage: genmodel_eval.py [-s <save_dir>] [-n <n_samples>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("Usage: genmodel_eval.py [-s <save_dir>] [-n <n_samples>]")
            sys.exit()
        elif opt in ('-s', '--save_dir'):
            save_dir = arg
        elif opt in ('-n', '--n_samples'):
            n_samples = int(arg)

    print(f"Evaluating models — generating {n_samples} samples each...")
    print(f"Output dir: {save_dir}")

    all_metrics  = {}
    model_types  = ["VAE", "GAN", "Diffusion"]

    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Model: {model_type}")
        print(f"{'='*50}")

        # Load model
        model = load_pt_model(model_type, save_dir, device)
        model.eval()

        # Generate 25 samples
        with torch.no_grad():
            samples = model.sample(n_samples, device)

        # Save sample grid
        save_sample_grid(samples, model_type, save_dir)

        # Compute metrics for each sample
        metrics = {m: [] for m in ['VoL', 'TEN', 'HFE', 'MLSD', 'GLCM']}
        for i in range(n_samples):
            m = compute_metrics(samples[i])
            for k, v in m.items():
                metrics[k].append(v)

        # Print summary
        print(f"\n  Metrics (mean ± std over {n_samples} samples):")
        for metric_name, values in metrics.items():
            print(f"  {metric_name:<6} {np.mean(values):.4f} ± {np.std(values):.4f}")

        all_metrics[model_type] = metrics

    # Comparison plot
    save_comparison_plot(all_metrics, save_dir)
    print(f"\nAll done! Check {save_dir} for outputs.")


if __name__ == "__main__":
    main(sys.argv[1:])
