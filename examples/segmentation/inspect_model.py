"""Visually inspect a trained U-Net's segmentation quality on validation slices.

Loads the checkpoint saved by train_unet.py and saves a PNG grid comparing
the FLAIR input, ground-truth mask, and predicted mask for a handful of
validation slices - a mix of tumor-containing and tumor-free, so both
"can it find a tumor" and "does it stay quiet where there isn't one" are
visible at a glance. Also prints each sample's Dice score.
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")  # no display needed - this only ever saves a PNG to disk
import matplotlib.pyplot as plt

from nabla.tensor import Tensor
from nabla.backend import gpu_available
from nabla.checkpoint import load_checkpoint
from nabla.grad_mode import no_grad
from nabla.nn.unet import UNet

from data import list_slice_files, split_by_patient, filter_tumor_containing, load_batch

DATA_DIR = "D:\\projects\\brain-tumor-segmentation\\data\\processed_slices"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".unet_checkpoint.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "inspection_output.png")

# must match train_unet.py's config - a checkpoint only stores parameter
# values, not the architecture/preprocessing that produced them
FEATURES = (64, 128, 256, 512)
DOWNSAMPLE = 3
NUM_TUMOR_SAMPLES = 6
NUM_EMPTY_SAMPLES = 2
THRESHOLD = 0.5


def dice_score(pred_binary: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    return float((2 * intersection + eps) / (union + eps))


def main() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH} - run train_unet.py first.")
        return

    files = list_slice_files(DATA_DIR)
    _, val = split_by_patient(files, 0.1)

    print("Scanning val slices for tumor content...")
    val_tumor = filter_tumor_containing(val)
    val_tumor_set = set(val_tumor)
    val_empty = [t for t in val if t not in val_tumor_set]
    print(f"{len(val_tumor)} tumor-containing / {len(val_empty)} tumor-free val slices")

    device = "cuda" if gpu_available() else "cpu"
    print(f"Running on {device}")

    model = UNet(in_channels=2, out_channels=1, features=FEATURES)
    model.to(device)
    metadata = load_checkpoint(CHECKPOINT_PATH, model)
    print(f"Loaded checkpoint from epoch {metadata['epoch']}, val_loss={metadata['val_loss']:.4f}")

    rng = np.random.default_rng(42)
    tumor_idx = rng.choice(len(val_tumor), size=min(NUM_TUMOR_SAMPLES, len(val_tumor)), replace=False)
    empty_idx = rng.choice(len(val_empty), size=min(NUM_EMPTY_SAMPLES, len(val_empty)), replace=False)

    samples = [(val_tumor, i) for i in tumor_idx] + [(val_empty, i) for i in empty_idx]
    n = len(samples)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    col_titles = ["FLAIR input", "Ground truth", "Predicted probability", f"Predicted mask (p>{THRESHOLD})"]

    model.eval()
    dice_scores = []
    with no_grad():
        for row, (pool, idx) in enumerate(samples):
            X, y = load_batch(pool, np.array([idx]), downsample=DOWNSAMPLE)
            logits = model(Tensor(X).to(device))
            probs = logits.sigmoid().data
            probs_cpu = probs if device == "cpu" else probs.get()
            pred_binary = (probs_cpu > THRESHOLD).astype(np.float32)

            score = dice_score(pred_binary[0, 0], y[0, 0])
            dice_scores.append(score)
            kind = "tumor" if pool is val_tumor else "empty"
            print(f"sample {row} ({kind}): dice = {score:.4f}")

            flair = X[0, 0]
            gt = y[0, 0]
            prob_map = probs_cpu[0, 0]
            pred_map = pred_binary[0, 0]

            axes[row, 0].imshow(flair, cmap="gray")
            axes[row, 1].imshow(flair, cmap="gray")
            axes[row, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap="autumn", alpha=0.6)
            axes[row, 2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
            axes[row, 3].imshow(flair, cmap="gray")
            axes[row, 3].imshow(np.ma.masked_where(pred_map == 0, pred_map), cmap="autumn", alpha=0.6)

            axes[row, 0].set_ylabel(f"{kind}\ndice={score:.3f}", fontsize=9)
            for col in range(4):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=10)

    fig.suptitle(f"UNet checkpoint @ epoch {metadata['epoch']} - mean dice = {np.mean(dice_scores):.4f}")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=120)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Mean dice over {n} samples: {np.mean(dice_scores):.4f}")


if __name__ == "__main__":
    main()
