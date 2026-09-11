"""
===============================================================================
File: src/data/validate_preprocessing.py
Purpose: Validation Script for Preprocessed Multimodal GeoTIFF Rasters

Description:
    Performs 11 comprehensive integrity and spatial alignment checks on:
    - Sentinel-1 processed raster (data/processed/sentinel1/sentinel1_processed.tif)
    - Sentinel-2 processed raster (data/processed/sentinel2/sentinel2_processed.tif)
    - Valid pixel mask (data/processed/masks/valid_pixel_mask.tif)
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

from src.utils.config import (
    PROCESSED_DIR,
)


def validate_preprocessing():
    """
    Executes the 11 validation checks on preprocessed multimodal satellite rasters.

    Raises:
        AssertionError / ValueError / FileNotFoundError if any validation check fails.
    """
    print("=== Starting Preprocessing Validation Checks ===")

    s1_processed_path = PROCESSED_DIR / "sentinel1" / "sentinel1_processed.tif"
    s2_processed_path = PROCESSED_DIR / "sentinel2" / "sentinel2_processed.tif"
    mask_path = PROCESSED_DIR / "masks" / "valid_pixel_mask.tif"

    # CHECK 1: Processed files exist
    print("\n[CHECK 1/11] Checking existence of processed files...")
    for path, name in [
        (s1_processed_path, "Sentinel-1 Processed"),
        (s2_processed_path, "Sentinel-2 Processed"),
        (mask_path, "Valid Pixel Mask"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Validation FAILED: {name} file does not exist at {path}")
        print(f"  [PASS] Found {name}: {path}")

    # Open dataset handles
    with rasterio.open(s1_processed_path) as s1_ds, \
         rasterio.open(s2_processed_path) as s2_ds, \
         rasterio.open(mask_path) as mask_ds:

        # CHECK 2: Sentinel-1 has 2 bands
        print("\n[CHECK 2/11] Validating Sentinel-1 band count...")
        if s1_ds.count != 2:
            raise ValueError(f"Validation FAILED: Sentinel-1 band count is {s1_ds.count}, expected 2.")
        print(f"  [PASS] Sentinel-1 has {s1_ds.count} bands.")

        # CHECK 3: Sentinel-2 has 6 bands
        print("\n[CHECK 3/11] Validating Sentinel-2 band count...")
        if s2_ds.count != 6:
            raise ValueError(f"Validation FAILED: Sentinel-2 band count is {s2_ds.count}, expected 6.")
        print(f"  [PASS] Sentinel-2 has {s2_ds.count} bands.")

        # CHECK 4: Dimensions match
        print("\n[CHECK 4/11] Validating raster dimensions (width x height)...")
        s1_dims = (s1_ds.width, s1_ds.height)
        s2_dims = (s2_ds.width, s2_ds.height)
        mask_dims = (mask_ds.width, mask_ds.height)
        if not (s1_dims == s2_dims == mask_dims):
            raise ValueError(
                f"Validation FAILED: Dimension mismatch. S1: {s1_dims}, S2: {s2_dims}, Mask: {mask_dims}"
            )
        print(f"  [PASS] Dimensions match across all rasters: {s1_dims[0]} x {s1_dims[1]}")

        # CHECK 5: CRS matches
        print("\n[CHECK 5/11] Validating Coordinate Reference Systems (CRS)...")
        if not (s1_ds.crs == s2_ds.crs == mask_ds.crs):
            raise ValueError(
                f"Validation FAILED: CRS mismatch. S1: {s1_ds.crs}, S2: {s2_ds.crs}, Mask: {mask_ds.crs}"
            )
        print(f"  [PASS] CRS matches across all rasters: {s1_ds.crs}")

        # CHECK 6: Transform matches
        print("\n[CHECK 6/11] Validating Affine Geotransforms...")
        if not (s1_ds.transform == s2_ds.transform == mask_ds.transform):
            raise ValueError(
                f"Validation FAILED: Transform mismatch.\nS1: {s1_ds.transform}\nS2: {s2_ds.transform}\nMask: {mask_ds.transform}"
            )
        print("  [PASS] Geotransforms match perfectly across all rasters.")

        # CHECK 7: Geographic Bounds match
        print("\n[CHECK 7/11] Validating Geographic Bounding Boxes...")
        if not (s1_ds.bounds == s2_ds.bounds == mask_ds.bounds):
            raise ValueError(
                f"Validation FAILED: Bounds mismatch.\nS1: {s1_ds.bounds}\nS2: {s2_ds.bounds}\nMask: {mask_ds.bounds}"
            )
        print("  [PASS] Bounding boxes match across all rasters.")

        # CHECK 8: Band descriptions are correct
        print("\n[CHECK 8/11] Validating band descriptions...")
        s1_descs = tuple(s1_ds.descriptions)
        s2_descs = tuple(s2_ds.descriptions)
        expected_s1 = ("VV", "VH")
        expected_s2 = ("B2", "B3", "B4", "B8", "B11", "B12")

        if s1_descs != expected_s1:
            raise ValueError(f"Validation FAILED: S1 band descriptions {s1_descs} do not match expected {expected_s1}")
        if s2_descs != expected_s2:
            raise ValueError(f"Validation FAILED: S2 band descriptions {s2_descs} do not match expected {expected_s2}")
        print(f"  [PASS] Sentinel-1 Band Descriptions: {s1_descs}")
        print(f"  [PASS] Sentinel-2 Band Descriptions: {s2_descs}")

        # Load raster pixel arrays for numeric inspection
        s1_data = s1_ds.read()
        s2_data = s2_ds.read()
        mask_data = mask_ds.read(1)

        # CHECK 10: Valid pixel mask exists and contains binary 0 and 1 values
        print("\n[CHECK 10/11] Validating Valid Pixel Mask values...")
        unique_mask_vals = set(np.unique(mask_data))
        if not unique_mask_vals.issubset({0, 1}):
            raise ValueError(f"Validation FAILED: Mask contains invalid unique values {unique_mask_vals}")
        print(f"  [PASS] Valid pixel mask contains binary values: {sorted(list(unique_mask_vals))}")

        # CHECK 9: Valid normalized values are between 0 and 1
        print("\n[CHECK 9/11] Validating normalized value ranges for valid pixels...")
        valid_indices = mask_data == 1

        for i, band_name in enumerate(s1_descs):
            valid_vals = s1_data[i][valid_indices]
            min_val, max_val = np.min(valid_vals), np.max(valid_vals)
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(
                    f"Validation FAILED: S1 band {band_name} valid values out of range [0, 1]: [{min_val}, {max_val}]"
                )
            print(f"  [PASS] S1 band {band_name:<4} valid pixel values range: [{min_val:.6f}, {max_val:.6f}]")

        for i, band_name in enumerate(s2_descs):
            valid_vals = s2_data[i][valid_indices]
            min_val, max_val = np.min(valid_vals), np.max(valid_vals)
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(
                    f"Validation FAILED: S2 band {band_name} valid values out of range [0, 1]: [{min_val}, {max_val}]"
                )
            print(f"  [PASS] S2 band {band_name:<4} valid pixel values range: [{min_val:.6f}, {max_val:.6f}]")

        # CHECK 11: Sentinel-1 and Sentinel-2 spatial alignment
        print("\n[CHECK 11/11] Confirming spatial alignment between Sentinel-1 and Sentinel-2...")
        is_aligned = (
            (s1_ds.crs == s2_ds.crs) and
            (s1_ds.transform == s2_ds.transform) and
            (s1_ds.width == s2_ds.width) and
            (s1_ds.height == s2_ds.height) and
            (s1_ds.bounds == s2_ds.bounds)
        )
        if not is_aligned:
            raise ValueError("Validation FAILED: Spatial alignment check between S1 and S2 failed.")
        print("  [PASS] Sentinel-1 and Sentinel-2 are perfectly spatially aligned!")

    print("\n===============================================================================")
    print("[SUCCESS] All 11 preprocessing validation checks passed successfully!")
    print("===============================================================================")
    return True


if __name__ == "__main__":
    validate_preprocessing()
