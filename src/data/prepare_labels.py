"""
===============================================================================
File: src/data/prepare_labels.py
Purpose: Ground Truth Label Preparation (ESA WorldCover) for FusionLand AI

Description:
    1. Ensures data/raw/labels/, data/processed/labels/, and outputs/reports/ exist.
    2. Searches data/raw/labels/ for ESA WorldCover GeoTIFF raster (*.tif / *.tiff).
    3. Inspects raw label metadata (file name, bands, dimensions, CRS, resolution,
       bounds, data type, NoData value, and unique class values).
    4. Compares ESA WorldCover raster against Sentinel-1 reference grid (EPSG:4326, 4112x4008).
    5. Aligns and reprojects labels using NEAREST-NEIGHBOR resampling to preserve
       categorical integer class IDs without float conversion or normalization.
    6. Saves aligned labels to data/processed/labels/worldcover_labels.tif.
    7. Generates detailed report at outputs/reports/label_preparation_report.txt.
===============================================================================
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    RAW_LABELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)

# Standard ESA WorldCover 10m Class Mapping Legend
WORLDCOVER_LEGEND = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def find_raw_label_files(raw_labels_dir):
    """
    Searches for GeoTIFF label files (*.tif, *.tiff) in data/raw/labels/.

    Args:
        raw_labels_dir (Path): Directory path to search.

    Returns:
        list of Path: List of raw label GeoTIFF paths found.
    """
    raw_labels_dir = Path(raw_labels_dir)
    if not raw_labels_dir.exists():
        raw_labels_dir.mkdir(parents=True, exist_ok=True)

    tif_files = [
        f for f in raw_labels_dir.glob("*")
        if f.is_file() and f.suffix.lower() in [".tif", ".tiff"] and not f.name.startswith(".")
    ]
    return sorted(tif_files)


def prepare_labels():
    """
    Inspects, aligns, and reprojects ESA WorldCover GeoTIFF to match Sentinel reference grid.
    """
    # TASK 1: Ensure required folders exist
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    processed_labels_dir = PROCESSED_DIR / "labels"
    processed_labels_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_label_path = processed_labels_dir / "worldcover_labels.tif"
    report_path = REPORTS_DIR / "label_preparation_report.txt"
    ref_s1_path = PROCESSED_DIR / "sentinel1" / "sentinel1_processed.tif"

    # TASK 2: Check for ESA WorldCover File (*.tif / *.tiff)
    raw_tif_files = find_raw_label_files(RAW_LABELS_DIR)

    if not raw_tif_files:
        print("\n" + "=" * 79)
        print("[WARNING] NO ESA WORLDCOVER GEOTIFF FOUND IN data/raw/labels/")
        print("=" * 79)
        print("Please download ESA WorldCover GeoTIFF and place it inside:")
        print(f"  -> {RAW_LABELS_DIR.resolve()}\n")
        print("Download Instructions:")
        print("  1. Visit the ESA WorldCover Viewer: https://viewer.esa-worldcover.org/worldcover/")
        print("  2. Or download directly via Zenodo / AWS Open Data:")
        print("     - WorldCover 2021 (v200): https://esa-worldcover.org/en/data-access")
        print("     - WorldCover 2020 (v100): https://esa-worldcover.s3.amazonaws.com/")
        print("  3. Save the downloaded .tif / .tiff file to:")
        print(f"     -> {RAW_LABELS_DIR.resolve()}\\<your_worldcover_file>.tif")
        print("  4. Re-run this script: python src/data/prepare_labels.py")
        print("=" * 79 + "\n")
        return False

    if len(raw_tif_files) > 1:
        print("\n" + "=" * 79)
        print("[ERROR] MULTIPLE GEOTIFF FILES FOUND IN data/raw/labels/")
        print("=" * 79)
        print("Found the following TIFF files:")
        for f in raw_tif_files:
            print(f"  - {f.name}")
        print("\nPlease keep only ONE ESA WorldCover GeoTIFF file in:")
        print(f"  -> {RAW_LABELS_DIR.resolve()}\n")
        print("=" * 79 + "\n")
        return False

    raw_label_path = raw_tif_files[0]

    # Task 7: Reference Sentinel checks
    if not ref_s1_path.exists():
        print(f"[ERROR] Reference Sentinel-1 file not found at: {ref_s1_path}")
        print("Please run 'python src/data/preprocess.py' first.")
        return False

    print(f"[INFO] Found raw label file: {raw_label_path.name}")
    print(f"[INFO] Full path: {raw_label_path.resolve()}")
    print(f"[INFO] Reference Sentinel-1 file: {ref_s1_path.resolve()}")

    # Read reference raster metadata
    with rasterio.open(ref_s1_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_width = ref_ds.width
        ref_height = ref_ds.height
        ref_bounds = ref_ds.bounds
        ref_res = (ref_transform.a, abs(ref_transform.e))

    # Task 3 & 7: Inspect raw label raster and handle corrupted / missing metadata
    try:
        raw_ds = rasterio.open(raw_label_path)
    except Exception as e:
        print(f"[ERROR] Failed to open raw label GeoTIFF '{raw_label_path.name}': {e}")
        print("The file may be corrupted or in an invalid format.")
        return False

    with raw_ds:
        if raw_ds.crs is None:
            print(f"[ERROR] Raw label GeoTIFF '{raw_label_path.name}' is missing CRS metadata.")
            return False

        raw_crs = raw_ds.crs
        raw_transform = raw_ds.transform
        raw_width = raw_ds.width
        raw_height = raw_ds.height
        raw_bounds = raw_ds.bounds
        raw_dtype = raw_ds.dtypes[0]
        raw_nodata = raw_ds.nodata
        raw_count = raw_ds.count
        raw_res = (raw_transform.a, abs(raw_transform.e))

        print("\n--- Raw ESA WorldCover Raster Inspection ---")
        print(f"  File Name: {raw_label_path.name}")
        print(f"  Number of Bands: {raw_count}")
        print(f"  Dimensions: {raw_width} x {raw_height}")
        print(f"  CRS: {raw_crs}")
        print(f"  Pixel Resolution: ({raw_res[0]:.6f}, {raw_res[1]:.6f})")
        print(f"  Bounds: {raw_bounds}")
        print(f"  Data Type: {raw_dtype}")
        print(f"  NoData Value: {raw_nodata}")

        try:
            raw_data = raw_ds.read(1)
        except Exception as e:
            print(f"[ERROR] Could not read band 1 data from raw label file: {e}")
            return False

        raw_unique_classes = np.unique(raw_data)
        print(f"  Unique Raw Class IDs: {list(raw_unique_classes)}")

        # TASK 3 & 4: Compare alignment and preserve categorical labels
        needs_reproject = (
            (raw_crs != ref_crs) or
            (raw_width != ref_width) or
            (raw_height != ref_height) or
            (raw_transform != ref_transform) or
            (raw_bounds != ref_bounds)
        )

        aligned_data = np.zeros((ref_height, ref_width), dtype=np.uint8)

        if needs_reproject:
            print("\n[INFO] Reprojecting and aligning labels to Sentinel reference grid...")
            print("  Resampling Method: NEAREST NEIGHBOR (Nearest-neighbor sampling enforced for categorical labels)")
            resampling_method = "NEAREST NEIGHBOR (Resampling.nearest)"
            resampling_required = True

            reproject(
                source=raw_data,
                destination=aligned_data,
                src_transform=raw_transform,
                src_crs=raw_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                src_nodata=raw_nodata,
                dst_nodata=0,
            )
        else:
            print("\n[INFO] Raw labels already match Sentinel reference grid perfectly.")
            aligned_data = raw_data.astype(np.uint8)
            resampling_method = "None (Raster already matches reference grid)"
            resampling_required = False

    # Calculate statistics on aligned labels
    proc_unique_classes, counts = np.unique(aligned_data, return_counts=True)
    total_pixels = aligned_data.size
    class_distribution = dict(zip(proc_unique_classes, counts))

    print("\n--- Aligned Label Dataset Summary ---")
    print(f"  Target Dimensions: {ref_width} x {ref_height}")
    print(f"  Target CRS: {ref_crs}")
    print(f"  Unique Aligned Class IDs: {list(proc_unique_classes)}")

    # Write processed label GeoTIFF (TASK 4: Categorical Integer Preservation)
    out_profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "nodata": 0,
        "width": ref_width,
        "height": ref_height,
        "count": 1,
        "crs": ref_crs,
        "transform": ref_transform,
    }

    with rasterio.open(output_label_path, "w", **out_profile) as dst:
        dst.write(aligned_data, 1)
        dst.set_band_description(1, "worldcover_landcover")

    print(f"\n[SUCCESS] Aligned labels saved to: {output_label_path}")

    # TASK 5: Create Report
    report_lines = [
        "===============================================================================",
        "FUSIONLAND AI: GROUND TRUTH LABEL PREPARATION REPORT (STAGE 4A)",
        "===============================================================================\n",
        "1. INPUT & REFERENCE PATHS:",
        f"   - Input Raw ESA WorldCover Label File: {raw_label_path.resolve()}",
        f"   - Reference Sentinel Raster File:      {ref_s1_path.resolve()}",
        f"   - Output Processed Label File Path:    {output_label_path.resolve()}\n",
        "2. ORIGINAL RAW LABEL RASTER METADATA:",
        f"   - Input File Name:      {raw_label_path.name}",
        f"   - Original Band Count:  {raw_count}",
        f"   - Original CRS:         {raw_crs}",
        f"   - Original Dimensions:  {raw_width} x {raw_height}",
        f"   - Original Resolution:  ({raw_res[0]:.6f}, {raw_res[1]:.6f})",
        f"   - Original Bounds:      {raw_bounds}",
        f"   - Original Data Type:   {raw_dtype}",
        f"   - Original NoData:      {raw_nodata}",
        f"   - Unique Class Values:  {list(raw_unique_classes)}\n",
        "3. ALIGNMENT & RESAMPLING DETAILS:",
        f"   - Reprojection Required: {needs_reproject}",
        f"   - Resampling Required:   {resampling_required}",
        f"   - Resampling Method:     {resampling_method}",
        "   - Class Integrity:       Categorical Integer IDs Preserved (No float/interpolation)\n",
        "4. FINAL PROCESSED LABEL RASTER METADATA:",
        f"   - Final CRS:            {ref_crs}",
        f"   - Final Dimensions:     {ref_width} x {ref_height}",
        f"   - Final Resolution:     ({ref_res[0]:.6f}, {ref_res[1]:.6f})",
        f"   - Final Bounds:         {ref_bounds}",
        f"   - Output Data Type:     uint8 (Integer)",
        f"   - Output File Path:     {output_label_path.resolve()}\n",
        "5. PROCESSED LABEL CLASS DISTRIBUTION:",
        "   Class ID | Class Name                | Pixel Count  | Percentage",
        "   ----------------------------------------------------------------",
    ]

    for cid, cnt in class_distribution.items():
        name = WORLDCOVER_LEGEND.get(cid, "Unclassified / Background")
        pct = (cnt / total_pixels) * 100
        report_lines.append(f"   {cid:<8} | {name:<26} | {cnt:<12,} | {pct:.2f}%")

    report_lines.extend([
        "\n===============================================================================",
        "[SUCCESS] Stage 4A Label preparation report complete.",
        "===============================================================================",
    ])

    report_content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[SUCCESS] Label preparation report generated at: {report_path}")
    return True


if __name__ == "__main__":
    prepare_labels()

