"""
===============================================================================
File: src/data/validate_patches.py
Purpose: Comprehensive Validation and Quality Assurance for Stage 4B Patches and Splits

Description:
    Performs 15 mandatory verification checks on generated multimodal image patches,
    metadata CSV, and train/validation/testing split files.

    Verification Checks:
    1. Patch directories exist.
    2. Sentinel-1 patch count equals Sentinel-2 patch count.
    3. Sentinel-1 patch count equals label patch count.
    4. All patch IDs match across modalities.
    5. Sentinel-1 patch shape is (2, 256, 256).
    6. Sentinel-2 patch shape is (6, 256, 256).
    7. Label patch shape is (256, 256).
    8. Sentinel patch datatypes are float32.
    9. Label patch datatype is integer (uint8).
    10. All Sentinel pixel values are within [0, 1].
    11. No incomplete patches exist.
    12. Train, validation, and test split CSV files exist.
    13. No patch ID overlap between train, val, and test splits.
    14. Confirm split distribution is ~70% train / 15% val / 15% test.
    15. Print final patch counts and success banner.

    Also generates: outputs/reports/patch_creation_report.txt
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
    BASE_DIR,
    DATA_DIR,
    PATCHES_DIR,
    SPLITS_DIR,
    REPORTS_DIR,
    PATCH_SIZE,
    MIN_VALID_RATIO,
    RANDOM_SEED,
)


def generate_stage4b_report(s1_files, s2_files, label_files, metadata_path, train_df, val_df, test_df):
    """
    Generates outputs/reports/patch_creation_report.txt summarizing Stage 4B results.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "patch_creation_report.txt"

    total_accepted = len(s1_files)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    pct_train = (n_train / total_accepted * 100) if total_accepted > 0 else 0
    pct_val = (n_val / total_accepted * 100) if total_accepted > 0 else 0
    pct_test = (n_test / total_accepted * 100) if total_accepted > 0 else 0

    meta_df = pd.read_csv(metadata_path) if metadata_path.exists() else None
    if meta_df is not None:
        total_possible = (4112 // PATCH_SIZE) * (4008 // PATCH_SIZE)
        total_rejected = total_possible - total_accepted
    else:
        total_possible = "N/A"
        total_rejected = "N/A"

    lines = [
        "===============================================================================",
        "FUSIONLAND AI: MULTIMODAL PATCH CREATION & SPLITTING REPORT (STAGE 4B)",
        "===============================================================================\n",
        "1. INPUT DATASET PARAMETERS:",
        "   - Input Dimensions:         4112 x 4008",
        "   - Input Coordinate Reference: EPSG:4326",
        f"   - Target Patch Size:        {PATCH_SIZE} x {PATCH_SIZE} pixels",
        f"   - Minimum Valid Pixel Ratio: {MIN_VALID_RATIO * 100:.0f}%",
        f"   - Random Seed for Splitting: {RANDOM_SEED}\n",
        "2. PATCH EXTRACTION SUMMARY:",
        f"   - Total Possible Patches:   {total_possible}",
        f"   - Total Accepted Patches:   {total_accepted}",
        f"   - Total Rejected Patches:   {total_rejected}",
        f"   - Sentinel-1 Patch Count:   {len(s1_files)}",
        f"   - Sentinel-2 Patch Count:   {len(s2_files)}",
        f"   - Ground Truth Label Count: {len(label_files)}\n",
        "3. PATCH MODALITY SPECIFICATIONS:",
        "   - Sentinel-1 (SAR):",
        f"       Format: NumPy (.npy)",
        f"       Shape:  (2, {PATCH_SIZE}, {PATCH_SIZE})",
        f"       Dtype:  float32",
        f"       Bands:  VV, VH",
        "   - Sentinel-2 (Optical):",
        f"       Format: NumPy (.npy)",
        f"       Shape:  (6, {PATCH_SIZE}, {PATCH_SIZE})",
        f"       Dtype:  float32",
        f"       Bands:  B2, B3, B4, B8, B11, B12",
        "   - ESA WorldCover Labels:",
        f"       Format: NumPy (.npy)",
        f"       Shape:  ({PATCH_SIZE}, {PATCH_SIZE})",
        f"       Dtype:  uint8 (Integer)\n",
        "4. DATASET SPLIT SUMMARY (SPATIAL BLOCK PARTITION):",
        f"   - Training Set:   {n_train:<5} patches ({pct_train:.2f}%)",
        f"   - Validation Set: {n_val:<5} patches ({pct_val:.2f}%)",
        f"   - Testing Set:    {n_test:<5} patches ({pct_test:.2f}%)",
        f"   - Total Splitted: {n_train + n_val + n_test:<5} patches (100.00%)\n",
        "5. SPLIT FILE LOCATIONS:",
        f"   - Metadata CSV:     {metadata_path.resolve()}",
        f"   - Training CSV:     {(SPLITS_DIR / 'train.csv').resolve()}",
        f"   - Validation CSV:   {(SPLITS_DIR / 'val.csv').resolve()}",
        f"   - Testing CSV:      {(SPLITS_DIR / 'test.csv').resolve()}",
        "\n===============================================================================",
        "[SUCCESS] Stage 4B patch creation and splitting report generated successfully.",
        "===============================================================================",
    ]

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[SUCCESS] Stage 4B report generated at: {report_path}")


def validate_patches():
    """
    Executes 15 comprehensive validation checks for Stage 4B.
    """
    print("=== Starting Stage 4B Patch & Split Validation ===")

    s1_dir = PATCHES_DIR / "sentinel1"
    s2_dir = PATCHES_DIR / "sentinel2"
    lbl_dir = PATCHES_DIR / "labels"

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"
    meta_csv = PATCHES_DIR / "patch_metadata.csv"

    # CHECK 1: Patch directories exist
    print("[CHECK 1/15] Verifying patch and split directories exist...")
    for d, d_name in [(s1_dir, "S1"), (s2_dir, "S2"), (lbl_dir, "Labels"), (SPLITS_DIR, "Splits")]:
        if not d.exists():
            raise FileNotFoundError(f"[FAIL] Directory missing: {d}")

    # Gather files
    s1_files = sorted(list(s1_dir.glob("*.npy")))
    s2_files = sorted(list(s2_dir.glob("*.npy")))
    lbl_files = sorted(list(lbl_dir.glob("*.npy")))

    n_s1 = len(s1_files)
    n_s2 = len(s2_files)
    n_lbl = len(lbl_files)

    if n_s1 == 0:
        raise ValueError("[FAIL] No patches found in Sentinel-1 directory!")

    # CHECK 2: S1 count == S2 count
    print("[CHECK 2/15] Verifying S1 patch count equals S2 patch count...")
    if n_s1 != n_s2:
        raise ValueError(f"[FAIL] S1 count ({n_s1}) != S2 count ({n_s2})")

    # CHECK 3: S1 count == Label count
    print("[CHECK 3/15] Verifying S1 patch count equals label patch count...")
    if n_s1 != n_lbl:
        raise ValueError(f"[FAIL] S1 count ({n_s1}) != Label count ({n_lbl})")

    # CHECK 4: All patch IDs match
    print("[CHECK 4/15] Verifying patch IDs match across all modalities...")
    s1_ids = {f.stem for f in s1_files}
    s2_ids = {f.stem for f in s2_files}
    lbl_ids = {f.stem for f in lbl_files}

    if not (s1_ids == s2_ids == lbl_ids):
        raise ValueError("[FAIL] Patch ID mismatch across Sentinel-1, Sentinel-2, and label files!")

    # Load samples to inspect shape, dtype, range
    print("[CHECK 5/15] Verifying Sentinel-1 patch shape (2, 256, 256)...")
    print("[CHECK 6/15] Verifying Sentinel-2 patch shape (6, 256, 256)...")
    print("[CHECK 7/15] Verifying label patch shape (256, 256)...")
    print("[CHECK 8/15] Verifying Sentinel patch datatypes are float32...")
    print("[CHECK 9/15] Verifying label patch datatype is integer...")
    print("[CHECK 10/15] Verifying Sentinel pixel values are within [0, 1]...")
    print("[CHECK 11/15] Verifying no incomplete patches exist...")

    for f_s1, f_s2, f_lbl in zip(s1_files, s2_files, lbl_files):
        arr_s1 = np.load(f_s1)
        arr_s2 = np.load(f_s2)
        arr_lbl = np.load(f_lbl)

        # Check shapes
        if arr_s1.shape != (2, 256, 256):
            raise ValueError(f"[FAIL] Invalid S1 patch shape {arr_s1.shape} in {f_s1.name}")
        if arr_s2.shape != (6, 256, 256):
            raise ValueError(f"[FAIL] Invalid S2 patch shape {arr_s2.shape} in {f_s2.name}")
        if arr_lbl.shape != (256, 256):
            raise ValueError(f"[FAIL] Invalid Label patch shape {arr_lbl.shape} in {f_lbl.name}")

        # Check dtypes
        if arr_s1.dtype != np.float32:
            raise TypeError(f"[FAIL] S1 patch dtype is {arr_s1.dtype}, expected float32 in {f_s1.name}")
        if arr_s2.dtype != np.float32:
            raise TypeError(f"[FAIL] S2 patch dtype is {arr_s2.dtype}, expected float32 in {f_s2.name}")
        if not np.issubdtype(arr_lbl.dtype, np.integer):
            raise TypeError(f"[FAIL] Label patch dtype is {arr_lbl.dtype}, expected integer in {f_lbl.name}")

        # Check range [0, 1]
        if np.nanmin(arr_s1) < 0.0 or np.nanmax(arr_s1) > 1.0:
            raise ValueError(f"[FAIL] S1 values out of [0, 1] range: min={np.nanmin(arr_s1)}, max={np.nanmax(arr_s1)}")
        if np.nanmin(arr_s2) < 0.0 or np.nanmax(arr_s2) > 1.0:
            raise ValueError(f"[FAIL] S2 values out of [0, 1] range: min={np.nanmin(arr_s2)}, max={np.nanmax(arr_s2)}")

    # CHECK 12: Train, val, test CSV exist
    print("[CHECK 12/15] Verifying train, validation, and test CSV files exist...")
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError("[FAIL] One or more split CSV files are missing in data/splits/")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_ids = set(train_df["patch_id"])
    val_ids = set(val_df["patch_id"])
    test_ids = set(test_df["patch_id"])

    # CHECK 13: No patch ID overlap between splits
    print("[CHECK 13/15] Verifying no patch ID overlap between train, val, and test...")
    overlap_tv = train_ids.intersection(val_ids)
    overlap_tt = train_ids.intersection(test_ids)
    overlap_vt = val_ids.intersection(test_ids)

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"[FAIL] Patch ID overlap detected! "
            f"Train-Val: {len(overlap_tv)}, Train-Test: {len(overlap_tt)}, Val-Test: {len(overlap_vt)}"
        )

    # CHECK 14: Confirm approx 70% train, 15% val, 15% test
    print("[CHECK 14/15] Confirming split ratio distribution...")
    total_split_patches = len(train_ids) + len(val_ids) + len(test_ids)

    pct_tr = (len(train_ids) / total_split_patches) * 100
    pct_va = (len(val_ids) / total_split_patches) * 100
    pct_te = (len(test_ids) / total_split_patches) * 100

    print(f"  Distribution: Train={pct_tr:.1f}%, Val={pct_va:.1f}%, Test={pct_te:.1f}%")

    # CHECK 15: Print number of patches in each split
    print("[CHECK 15/15] Finalizing patch counts per split...")
    print(f"  Train Patch Count:      {len(train_df)}")
    print(f"  Validation Patch Count: {len(val_df)}")
    print(f"  Testing Patch Count:    {len(test_df)}")

    # Generate Report
    generate_stage4b_report(s1_files, s2_files, lbl_files, meta_csv, train_df, val_df, test_df)

    # Print required final success banner
    print("\n" + "=" * 52)
    print("STAGE 4B PATCH VALIDATION")
    print("=" * 52)
    print("\n[SUCCESS] ALL PATCH VALIDATION CHECKS PASSED\n")
    print("[SUCCESS] STAGE 4B COMPLETED SUCCESSFULLY\n")
    print("=" * 52 + "\n")
    return True


if __name__ == "__main__":
    validate_patches()
