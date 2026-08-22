"""Converts raw BraTS 2019 NIfTI volumes into 2D slice pairs for training.

Not part of the nabla package itself - dataset-specific glue for the
segmentation example, kept separate like `shakespeare_data.py` keeps
Tiny Shakespeare's tokenizer away from the training script. Requires the
`segmentation` extra (`uv sync --extra segmentation`) for `nibabel`.

Expects the BraTS 2019 training set laid out as downloaded from Kaggle/
MICCAI, with per-patient folders under `HGG/` and `LGG/` subdirectories,
each containing `*_flair.nii`, `*_t1ce.nii`, and `*_seg.nii` volumes.
"""

from __future__ import annotations

import argparse
import glob
import os

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def normalize(slice_arr: NDArray) -> NDArray:
    """Z-score normalize a 2D slice using only its non-background pixels.

    A raw MRI slice is mostly black background (value 0) around the
    brain; including it in the mean/std would wash out the actual
    tissue intensities that matter, so statistics are computed over the
    brain pixels only and then applied to the whole slice.
    """
    mask = slice_arr > 0
    if mask.sum() == 0:
        return slice_arr
    mean = slice_arr[mask].mean()
    std = slice_arr[mask].std()
    return (slice_arr - mean) / (std + 1e-8)


def preprocess_dataset(root_dir: str, output_dir: str) -> None:
    """Slice every patient volume under `root_dir` and save img/mask pairs.

    For each patient, stacks the FLAIR and T1ce modalities into a
    (2, H, W) image per slice and a binarized (1, H, W) tumor mask
    (BraTS's 4-class segmentation labels collapsed to "tumor present"),
    saved as `{patient_id}_{slice_index:03d}_img.npy` /
    `..._mask.npy` under `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    patient_paths = []
    for subfolder in ("HGG", "LGG"):
        path = os.path.join(root_dir, subfolder)
        if os.path.exists(path):
            patient_paths.extend(sorted(glob.glob(os.path.join(path, "*"))))

    if not patient_paths:
        raise FileNotFoundError(
            f"No HGG/ or LGG/ patient folders found under {root_dir}. "
            "Point --data_path at an extracted BraTS 2019 training set."
        )
    print(f"Found {len(patient_paths)} patients under {root_dir}.")

    total_slices = 0
    for i, patient_path in enumerate(patient_paths):
        patient_id = os.path.basename(patient_path)
        if i % 20 == 0:
            print(f"  [{i}/{len(patient_paths)}] {patient_id}")

        try:
            flair_path = glob.glob(os.path.join(patient_path, "*flair.nii*"))[0]
            t1ce_path = glob.glob(os.path.join(patient_path, "*t1ce.nii*"))[0]
            seg_path = glob.glob(os.path.join(patient_path, "*seg.nii*"))[0]
        except IndexError:
            print(f"  skipping {patient_id}: missing flair/t1ce/seg volume")
            continue

        flair_vol = nib.load(flair_path).get_fdata()
        t1ce_vol = nib.load(t1ce_path).get_fdata()
        seg_vol = nib.load(seg_path).get_fdata()

        # BraTS volumes are (H, W, num_slices) - iterate the z-axis
        for z in range(flair_vol.shape[2]):
            img_flair = normalize(flair_vol[:, :, z])
            img_t1ce = normalize(t1ce_vol[:, :, z])
            image_stacked = np.stack([img_flair, img_t1ce], axis=0).astype(np.float32)

            mask = seg_vol[:, :, z].copy()
            mask[mask > 0] = 1.0  # collapse BraTS's 4-class labels to binary tumor/not
            mask = mask[np.newaxis, :, :].astype(np.float32)

            save_name = f"{patient_id}_{z:03d}"
            np.save(os.path.join(output_dir, f"{save_name}_img.npy"), image_stacked)
            np.save(os.path.join(output_dir, f"{save_name}_mask.npy"), mask)
            total_slices += 1

    print(f"Done - saved {total_slices} slices to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), ".brats_raw", "MICCAI_BraTS_2019_Data_Training"),
        help="Path to the extracted BraTS 2019 training set (contains HGG/ and LGG/).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), ".brats_slices"),
        help="Where to write the preprocessed {patient}_{slice}_{img,mask}.npy pairs.",
    )
    args = parser.parse_args()

    preprocess_dataset(args.data_path, args.output_path)
