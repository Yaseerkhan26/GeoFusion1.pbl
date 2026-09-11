"""
===============================================================================
File: src/data/create_patches.py
Purpose: Multimodal Patch Generation and Quality Filtering for FusionLand AI

Description:
    1. Ensures patch directories exist (data/patches/sentinel1, sentinel2, labels, data/splits).
    2. Opens processed Sentinel-1, Sentinel-2, ESA WorldCover labels, and valid pixel mask.
    3. Verifies spatial alignment (width, height, CRS, transform) across all inputs.
    4. Extracts non-overlapping 256x256 image patches.
    5. Applies quality filtering:
       - Valid pixel ratio >= 0.80 (MIN_VALID_RATIO).
       - Label validity (must contain valid land-cover labels > 0).
       - Dominant class threshold (discards patches where any class > 95%).
    6. Saves accepted patches as .npy files (S1: float32 (2,256,256), S2: float32 (6,256,256), Label: uint8 (256,256)).
    7. Generates data/patches/patch_metadata.csv with spatial and quality attributes.
    8. Prints patch extraction statistics.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    PROCESSED_DIR,
    PATCHES_DIR,
    SPLITS_DIR,
    PATCH_SIZE,
    MIN_VALID_RATIO,
)


def ensure_directories():
    """
    Creates necessary patch and split directories automatically.
    """
    s1_patch_dir = PATCHES_DIR / "sentinel1"
    s2_patch_dir = PATCHES_DIR / "sentinel2"
    label_patch_dir = PATCHES_DIR / "labels"

    for d in [PATCHES_DIR, s1_patch_dir, s2_patch_dir, label_patch_dir, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    return s1_patch_dir, s2_patch_dir, label_patch_dir


def create_patches():
    """
    Main execution pipeline for extracting multimodal patch triplets and metadata.
    """
    s1_patch_dir, s2_patch_dir, label_patch_dir = ensure_directories()

    s1_path = PROCESSED_DIR / "sentinel1" / "sentinel1_processed.tif"
    s2_path = PROCESSED_DIR / "sentinel2" / "sentinel2_processed.tif"
    label_path = PROCESSED_DIR / "labels" / "worldcover_labels.tif"
    mask_path = PROCESSED_DIR / "masks" / "valid_pixel_mask.tif"
    metadata_csv_path = PATCHES_DIR / "patch_metadata.csv"

    # Check input files existence
    for path, name in [
        (s1_path, "Sentinel-1"),
        (s2_path, "Sentinel-2"),
        (label_path, "WorldCover Labels"),
        (mask_path, "Valid Pixel Mask"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found at: {path}")

    print("=== Starting Stage 4B Spatial Patch Extraction ===")
    print(f"[INFO] Sentinel-1 Input: {s1_path}")
    print(f"[INFO] Sentinel-2 Input: {s2_path}")
    print(f"[INFO] Labels Input:     {label_path}")
    print(f"[INFO] Mask Input:       {mask_path}")
    print(f"[INFO] Patch Size:       {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"[INFO] Min Valid Ratio:  {MIN_VALID_RATIO * 100:.0f}%")

    with rasterio.open(s1_path) as s1_ds, \
         rasterio.open(s2_path) as s2_ds, \
         rasterio.open(label_path) as label_ds, \
         rasterio.open(mask_path) as mask_ds:

        # Verification step: same width, height, CRS, transform
        meta_s1 = (s1_ds.width, s1_ds.height, s1_ds.crs, s1_ds.transform)
        meta_s2 = (s2_ds.width, s2_ds.height, s2_ds.crs, s2_ds.transform)
        meta_lbl = (label_ds.width, label_ds.height, label_ds.crs, label_ds.transform)
        meta_msk = (mask_ds.width, mask_ds.height, mask_ds.crs, mask_ds.transform)

        if not (meta_s1 == meta_s2 == meta_lbl == meta_msk):
            print("[ERROR] Input rasters spatial metadata mismatch!")
            print(f"  Sentinel-1: {meta_s1}")
            print(f"  Sentinel-2: {meta_s2}")
            print(f"  Labels:     {meta_lbl}")
            print(f"  Mask:       {meta_msk}")
            raise ValueError("All input rasters must have identical dimensions, CRS, and transform.")

        print(f"[SUCCESS] Spatial Alignment Verified: {s1_ds.width}x{s1_ds.height}, CRS: {s1_ds.crs}")

        width = s1_ds.width
        height = s1_ds.height
        transform = s1_ds.transform

        num_patches_y = height // PATCH_SIZE
        num_patches_x = width // PATCH_SIZE
        total_possible_patches = num_patches_y * num_patches_x

        print(f"[INFO] Grid Dimensions: {num_patches_y} rows x {num_patches_x} cols = {total_possible_patches} possible patches.")

        accepted_count = 0
        rejected_invalid_pixels = 0
        rejected_invalid_labels = 0
        rejected_dominant_class = 0

        metadata_records = []

        for r_idx in range(num_patches_y):
            for c_idx in range(num_patches_x):
                y_off = r_idx * PATCH_SIZE
                x_off = c_idx * PATCH_SIZE

                window = Window(x_off, y_off, PATCH_SIZE, PATCH_SIZE)

                # 1. Read mask patch
                mask_patch = mask_ds.read(1, window=window)
                valid_pixel_count = int(np.sum(mask_patch == 1))
                valid_pixel_ratio = float(valid_pixel_count / (PATCH_SIZE * PATCH_SIZE))

                if valid_pixel_ratio < MIN_VALID_RATIO:
                    rejected_invalid_pixels += 1
                    continue

                # 2. Read label patch
                label_patch = label_ds.read(1, window=window)
                valid_labels = label_patch[label_patch > 0]

                if valid_labels.size == 0:
                    rejected_invalid_labels += 1
                    continue

                # 3. Dominant class check
                classes, counts = np.unique(label_patch, return_counts=True)
                max_count_idx = int(np.argmax(counts))
                dom_class = int(classes[max_count_idx])
                dom_class_ratio = float(counts[max_count_idx] / (PATCH_SIZE * PATCH_SIZE))

                if dom_class_ratio > 0.95:
                    rejected_dominant_class += 1
                    continue

                # Read S1 & S2 data
                s1_patch = s1_ds.read(window=window).astype(np.float32)  # Shape (2, 256, 256)
                s2_patch = s2_ds.read(window=window).astype(np.float32)  # Shape (6, 256, 256)
                label_patch_uint8 = label_patch.astype(np.uint8)         # Shape (256, 256)

                accepted_count += 1
                patch_id = f"patch_{accepted_count:05d}"

                # Geographic bounds for metadata
                left, bottom, right, top = rasterio.windows.bounds(window, transform)

                # Save .npy files
                np.save(s1_patch_dir / f"{patch_id}.npy", s1_patch)
                np.save(s2_patch_dir / f"{patch_id}.npy", s2_patch)
                np.save(label_patch_dir / f"{patch_id}.npy", label_patch_uint8)

                metadata_records.append({
                    "patch_id": patch_id,
                    "row": r_idx,
                    "column": c_idx,
                    "x_offset": x_off,
                    "y_offset": y_off,
                    "valid_pixel_ratio": round(valid_pixel_ratio, 4),
                    "dominant_class": dom_class,
                    "dominant_class_ratio": round(dom_class_ratio, 4),
                    "left": round(left, 6),
                    "bottom": round(bottom, 6),
                    "right": round(right, 6),
                    "top": round(top, 6),
                })

        # Save metadata CSV
        df_meta = pd.DataFrame(metadata_records)
        df_meta.to_csv(metadata_csv_path, index=False)

        total_rejected = rejected_invalid_pixels + rejected_invalid_labels + rejected_dominant_class

        print("\n=== PATCH EXTRACTION STATISTICS ===")
        print(f"  Total Possible Patches:            {total_possible_patches}")
        print(f"  Accepted Patches:                  {accepted_count}")
        print(f"  Total Rejected Patches:            {total_rejected}")
        print(f"    - Rejected (Invalid Pixels <80%): {rejected_invalid_pixels}")
        print(f"    - Rejected (Invalid Labels):      {rejected_invalid_labels}")
        print(f"    - Rejected (Dominant Class >95%): {rejected_dominant_class}")
        print(f"  Metadata Saved To:                 {metadata_csv_path}")

        return {
            "width": width,
            "height": height,
            "patch_size": PATCH_SIZE,
            "total_possible": total_possible_patches,
            "accepted": accepted_count,
            "rejected_invalid_pixels": rejected_invalid_pixels,
            "rejected_invalid_labels": rejected_invalid_labels,
            "rejected_dominant_class": rejected_dominant_class,
            "total_rejected": total_rejected,
            "metadata_csv": metadata_csv_path,
        }


if __name__ == "__main__":
    create_patches()
