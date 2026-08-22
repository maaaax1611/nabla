"""Loads preprocessed BraTS 2D slice pairs for the U-Net training example.

Not part of the nabla package itself - dataset-specific glue, kept
separate like `shakespeare_data.py` keeps Tiny Shakespeare's tokenizer
away from its training script. Expects `{patient_id}_{slice:03d}_img.npy`
/ `..._mask.npy` pairs as produced by `preprocess_brats.py`.
"""

from __future__ import annotations

import os
import re

import numpy as np
from numpy.typing import NDArray

_FILENAME_RE = re.compile(r"^(?P<patient_id>.+)_(?P<slice_idx>\d{3})_img\.npy$")


def list_slice_files(processed_dir: str) -> list[tuple[str, str, str]]:
    """Find every (patient_id, img_path, mask_path) triple in `processed_dir`.
    """
    triples = []
    for name in sorted(os.listdir(processed_dir)):
        match = _FILENAME_RE.match(name)
        if match is None:
            continue
        img_path = os.path.join(processed_dir, name)
        mask_path = os.path.join(processed_dir, name.replace("_img.npy", "_mask.npy"))
        if os.path.exists(mask_path):
            triples.append((match["patient_id"], img_path, mask_path))
    if not triples:
        raise FileNotFoundError(f"No {{patient}}_{{slice}}_img.npy/_mask.npy pairs found under {processed_dir}.")
    return triples


def split_by_patient(
    triples: list[tuple[str, str, str]],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Split slices into train/val by patient, not by slice.

    Slices from the same patient are highly correlated (neighboring
    slices look nearly identical), so a random per-slice split would leak
    train-set information into validation. Splitting whole patients
    instead keeps validation a genuinely unseen-patient measurement.
    """
    patient_ids = sorted({patient_id for patient_id, _, _ in triples})
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_fraction))
    val_patients = set(patient_ids[:n_val])

    train = [t for t in triples if t[0] not in val_patients]
    val = [t for t in triples if t[0] in val_patients]
    return train, val


def load_batch(triples: list[tuple[str, str, str]], indices: NDArray) -> tuple[NDArray, NDArray]:
    """Load and stack the img/mask pair for each of `indices` into a batch.

    Returns:
        X: (len(indices), 2, H, W) float32 - stacked FLAIR + T1ce slices.
        y: (len(indices), 1, H, W) float32 - binary tumor masks.
    """
    imgs = [np.load(triples[i][1]) for i in indices]
    masks = [np.load(triples[i][2]) for i in indices]
    return np.stack(imgs).astype(np.float32), np.stack(masks).astype(np.float32)


def get_batch(triples: list[tuple[str, str, str]], batch_size: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """Sample a random batch of slices (with replacement across calls)."""
    indices = rng.integers(0, len(triples), size=batch_size)
    return load_batch(triples, indices)
