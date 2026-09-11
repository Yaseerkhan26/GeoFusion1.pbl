"""
===============================================================================
File: src/data/validate_labels.py
Purpose: Validation Script for Processed Ground Truth Labels (ESA WorldCover)

Description:
    Performs 10 integrity & spatial alignment checks on:
    data/processed/labels/worldcover_labels.tif
    against Sentinel-1 and Sentinel-2 reference rasters.
===============================================================================
"""

import sys
import numpy as np
import rasterio
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import PROCESSED_DIR


def validate_labels():
    """
    Executes 10 validation checks on processed ESA WorldCover label GeoTIFF.
    """
    print("====================================================")
    print("STARTING STAGE 4A GROUND TRUTH LABEL VALIDATION")
    print("====================================================\n")

    processed_label_path = PROCESSED_DIR / "labels" / "worldcover_labels.tif"
    ref_s1_path = PROCESSED_DIR / "sentinel1" / "sentinel1_processed.tif"
    ref_s2_path = PROCESSED_DIR / "sentinel2" / "sentinel2_processed.tif"

    all_passed = True

    # CHECK 1: Processed label file exists
    print("[CHECK 1/10] Checking existence of processed label file...")
    if processed_label_path.exists():
        print(f"  [PASS] Processed label file exists: {processed_label_path.resolve()}")
    else:
        print(f"  [FAIL] Processed label file missing at: {processed_label_path.resolve()}")
        all_passed = False
        # If label file does not exist, remaining raster checks cannot run
        print("\n====================================================")
        print("STAGE 4A LABEL VALIDATION")
        print("====================================================")
        print("[FAIL] STAGE 4A NOT COMPLETE")
        return False

    # Ensure reference Sentinel-1 exists
    if not ref_s1_path.exists():
        print(f"  [FAIL] Reference Sentinel-1 file missing at: {ref_s1_path.resolve()}")
        all_passed = False

    # Ensure reference Sentinel-2 exists
    if not ref_s2_path.exists():
        print(f"  [FAIL] Reference Sentinel-2 file missing at: {ref_s2_path.resolve()}")
        all_passed = False

    if not (ref_s1_path.exists() and ref_s2_path.exists()):
        print("\n====================================================")
        print("STAGE 4A LABEL VALIDATION")
        print("====================================================")
        print("[FAIL] STAGE 4A NOT COMPLETE")
        return False

    with rasterio.open(processed_label_path) as lbl_ds, \
         rasterio.open(ref_s1_path) as s1_ds, \
         rasterio.open(ref_s2_path) as s2_ds:

        # CHECK 2: Label file has one band
        print("\n[CHECK 2/10] Checking single-band raster count...")
        if lbl_ds.count == 1:
            print("  [PASS] Label raster has exactly 1 band.")
        else:
            print(f"  [FAIL] Label raster has {lbl_ds.count} bands, expected 1.")
            all_passed = False

        # CHECK 3: Dimensions match Sentinel-1
        print("\n[CHECK 3/10] Checking dimensions match Sentinel-1...")
        lbl_dims = (lbl_ds.width, lbl_ds.height)
        s1_dims = (s1_ds.width, s1_ds.height)
        if lbl_dims == s1_dims:
            print(f"  [PASS] Dimensions match Sentinel-1: {lbl_dims[0]} x {lbl_dims[1]}")
        else:
            print(f"  [FAIL] Dimension mismatch. Labels: {lbl_dims}, Sentinel-1: {s1_dims}")
            all_passed = False

        # CHECK 4: CRS matches Sentinel-1
        print("\n[CHECK 4/10] Checking Coordinate Reference System (CRS) matches Sentinel-1...")
        if lbl_ds.crs == s1_ds.crs:
            print(f"  [PASS] CRS matches Sentinel-1: {lbl_ds.crs}")
        else:
            print(f"  [FAIL] CRS mismatch. Labels: {lbl_ds.crs}, Sentinel-1: {s1_ds.crs}")
            all_passed = False

        # CHECK 5: Transform matches Sentinel-1
        print("\n[CHECK 5/10] Checking Affine Geotransform matches Sentinel-1...")
        if lbl_ds.transform == s1_ds.transform:
            print("  [PASS] Geotransform exactly matches Sentinel-1.")
        else:
            print(f"  [FAIL] Transform mismatch.\n    Labels:     {lbl_ds.transform}\n    Sentinel-1: {s1_ds.transform}")
            all_passed = False

        # CHECK 6: Bounds match Sentinel-1
        print("\n[CHECK 6/10] Checking Geographic Bounds match Sentinel-1...")
        if lbl_ds.bounds == s1_ds.bounds:
            print("  [PASS] Geographic bounds match Sentinel-1 perfectly.")
        else:
            print(f"  [FAIL] Bounds mismatch.\n    Labels:     {lbl_ds.bounds}\n    Sentinel-1: {s1_ds.bounds}")
            all_passed = False

        # CHECK 7: Integer data type
        print("\n[CHECK 7/10] Checking integer data type...")
        lbl_dtype = np.dtype(lbl_ds.dtypes[0])
        if np.issubdtype(lbl_dtype, np.integer):
            print(f"  [PASS] Data type is integer ({lbl_dtype}).")
        else:
            print(f"  [FAIL] Data type is {lbl_dtype}, expected integer (e.g. uint8, uint16).")
            all_passed = False

        # CHECK 8: Valid categorical class values
        print("\n[CHECK 8/10] Checking valid categorical class values...")
        lbl_data = lbl_ds.read(1)
        unique_vals = np.unique(lbl_data)
        if len(unique_vals) > 0 and np.issubdtype(unique_vals.dtype, np.integer):
            print(f"  [PASS] Valid integer class values present ({len(unique_vals)} classes): {list(unique_vals)}")
        else:
            print(f"  [FAIL] Invalid or empty class values in label array.")
            all_passed = False

        # CHECK 9: Spatial alignment with Sentinel-1
        print("\n[CHECK 9/10] Checking spatial alignment with Sentinel-1...")
        s1_align = (lbl_ds.crs == s1_ds.crs) and (lbl_ds.bounds == s1_ds.bounds) and (lbl_ds.transform == s1_ds.transform)
        if s1_align:
            print("  [PASS] Labels are fully spatially aligned with Sentinel-1.")
        else:
            print("  [FAIL] Spatial alignment check with Sentinel-1 failed.")
            all_passed = False

        # CHECK 10: Spatial alignment with Sentinel-2
        print("\n[CHECK 10/10] Checking spatial alignment with Sentinel-2...")
        s2_align = (lbl_ds.crs == s2_ds.crs) and (lbl_ds.bounds == s2_ds.bounds) and (lbl_ds.transform == s2_ds.transform)
        if s2_align:
            print("  [PASS] Labels are fully spatially aligned with Sentinel-2.")
        else:
            print("  [FAIL] Spatial alignment check with Sentinel-2 failed.")
            all_passed = False

    print("\n====================================================")
    print("STAGE 4A LABEL VALIDATION")
    print("====================================================")

    if all_passed:
        print("[SUCCESS] ALL LABEL VALIDATION CHECKS PASSED")
        print("[SUCCESS] STAGE 4A COMPLETED SUCCESSFULLY")
        return True
    else:
        print("[FAIL] STAGE 4A NOT COMPLETE")
        return False


if __name__ == "__main__":
    validate_labels()

