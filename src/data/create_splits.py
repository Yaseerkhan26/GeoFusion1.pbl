"""
===============================================================================
File: src/data/create_splits.py
Purpose: Reproducible Spatially-Aware Train/Val/Test Dataset Splitting

Description:
    1. Loads accepted patch metadata from data/patches/patch_metadata.csv.
    2. Implements a spatially-aware block partition to prevent spatial data leakage
       between contiguous satellite patches.
    3. Partitions patch spatial grid locations into contiguous geographic blocks (e.g., 2x2 blocks).
    4. Randomly assigns spatial blocks to Train (70%), Validation (15%), and Test (15%)
       using RANDOM_SEED = 42 for strict reproducibility.
    5. Saves split CSV files:
       - data/splits/train.csv
       - data/splits/val.csv
       - data/splits/test.csv
    6. Ensures patch IDs across Sentinel-1, Sentinel-2, and labels are fully aligned.
===============================================================================
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    PATCHES_DIR,
    SPLITS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def create_spatial_splits(block_size=2):
    """
    Splits accepted patches into Train (70%), Val (15%), and Test (15%) using
    spatially isolated grid blocks to prevent spatial data leakage.

    Args:
        block_size (int): Grid block dimension (in number of patches) for spatial clustering.

    Returns:
        dict: Counts of patches in train, val, and test splits.
    """
    metadata_csv_path = PATCHES_DIR / "patch_metadata.csv"
    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Patch metadata file not found at: {metadata_csv_path}")

    df = pd.read_csv(metadata_csv_path)
    total_patches = len(df)

    if total_patches == 0:
        raise ValueError("No accepted patches found in metadata CSV.")

    print("=== Starting Stage 4B Dataset Partitioning & Splitting ===")
    print(f"[INFO] Total Accepted Patches: {total_patches}")
    print(f"[INFO] Target Ratio: Train={TRAIN_RATIO*100:.0f}%, Val={VAL_RATIO*100:.0f}%, Test={TEST_RATIO*100:.0f}%")
    print(f"[INFO] Random Seed: {RANDOM_SEED}")
    print(f"[INFO] Spatial Block Size: {block_size}x{block_size} patches")

    # Create spatial block IDs based on patch row and column
    df["block_row"] = df["row"] // block_size
    df["block_col"] = df["column"] // block_size
    df["block_id"] = df["block_row"].astype(str) + "_" + df["block_col"].astype(str)

    unique_blocks = df["block_id"].unique()
    num_blocks = len(unique_blocks)

    np.random.seed(RANDOM_SEED)
    shuffled_blocks = np.random.permutation(unique_blocks)

    # Compute target block split indices
    n_train_blocks = int(np.round(num_blocks * TRAIN_RATIO))
    n_val_blocks = int(np.round(num_blocks * VAL_RATIO))

    # Ensure at least 1 block in val and test if possible
    if n_val_blocks == 0 and num_blocks >= 3:
        n_val_blocks = 1
    if (num_blocks - n_train_blocks - n_val_blocks) <= 0 and num_blocks >= 3:
        n_train_blocks = max(1, n_train_blocks - 1)

    train_blocks = set(shuffled_blocks[:n_train_blocks])
    val_blocks = set(shuffled_blocks[n_train_blocks:n_train_blocks + n_val_blocks])
    test_blocks = set(shuffled_blocks[n_train_blocks + n_val_blocks:])

    def assign_split(b_id):
        if b_id in train_blocks:
            return "train"
        elif b_id in val_blocks:
            return "val"
        else:
            return "test"

    df["split"] = df["block_id"].apply(assign_split)

    # Construct file paths for each patch modality
    df["sentinel1_path"] = df["patch_id"].apply(lambda pid: str(Path("data/patches/sentinel1") / f"{pid}.npy"))
    df["sentinel2_path"] = df["patch_id"].apply(lambda pid: str(Path("data/patches/sentinel2") / f"{pid}.npy"))
    df["label_path"] = df["patch_id"].apply(lambda pid: str(Path("data/patches/labels") / f"{pid}.npy"))

    columns_to_save = ["patch_id", "sentinel1_path", "sentinel2_path", "label_path"]

    df_train = df[df["split"] == "train"][columns_to_save]
    df_val = df[df["split"] == "val"][columns_to_save]
    df_test = df[df["split"] == "test"][columns_to_save]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)

    pct_train = (n_train / total_patches) * 100
    pct_val = (n_val / total_patches) * 100
    pct_test = (n_test / total_patches) * 100

    print("\n=== DATASET SPLIT SUMMARY ===")
    print(f"  Training Set:   {n_train:<5} patches ({pct_train:.1f}%) -> {train_csv}")
    print(f"  Validation Set: {n_val:<5} patches ({pct_val:.1f}%) -> {val_csv}")
    print(f"  Testing Set:    {n_test:<5} patches ({pct_test:.1f}%) -> {test_csv}")
    print(f"  Total Patches:  {total_patches:<5} patches (100.0%)")

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "pct_train": pct_train,
        "pct_val": pct_val,
        "pct_test": pct_test,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }


if __name__ == "__main__":
    create_spatial_splits()
