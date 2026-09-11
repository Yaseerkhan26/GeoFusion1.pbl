"""
===============================================================================
File: src/data/download_worldcover.py
Purpose: Automated ESA WorldCover Label Download via Google Earth Engine (GEE)

Description:
    1. Reads geographic bounds & metadata from Sentinel-1 reference raster.
    2. Connects and initializes Google Earth Engine Python API (ee).
    3. Retrieves ESA WorldCover 10m land cover dataset for the study area.
    4. Downloads and extracts raw land-cover label raster to data/raw/labels/.
    5. Reprojects and aligns labels to match Sentinel-1 reference grid (EPSG:4326, 4112x4008)
       using NEAREST-NEIGHBOR resampling to preserve integer class IDs.
    6. Saves output to data/processed/labels/worldcover_labels.tif.
    7. Generates report at outputs/reports/label_preparation_report.txt.
    8. Automatically runs validation checks via src/data/validate_labels.py.
===============================================================================
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import ee

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


PROJECT_ID = "sentinel-land-cover"


def init_earth_engine():
    """
    Initializes Google Earth Engine Python API. Handles unauthenticated exceptions
    and provides clear step-by-step authentication instructions.

    Returns:
        bool: True if initialization succeeds, False otherwise.
    """
    print("\n[INFO] Connecting to Google Earth Engine...")
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"[SUCCESS] Connected and initialized Google Earth Engine successfully (Project: {PROJECT_ID}).")
        return True
    except Exception as e:
        print("\n" + "=" * 79)
        print("[WARNING] GOOGLE EARTH ENGINE AUTHENTICATION REQUIRED")
        print("=" * 79)
        print("Earth Engine initialization encountered the following exception:")
        print(f"  {e}\n")
        print("INSTRUCTIONS TO AUTHENTICATE GOOGLE EARTH ENGINE:")
        print("  1. Open your command line / terminal prompt.")
        print("  2. Run the authentication command:")
        print("         earthengine authenticate")
        print("  3. Log in with your Google / Earth Engine account in the web browser.")
        print("  4. Grant authorization permissions and copy the verification code if prompted.")
        print("  5. Paste the verification code back into your terminal prompt.")
        print("  6. Re-run this script:")
        print("         python src/data/download_worldcover.py")
        print("=" * 79 + "\n")
        return False


def download_and_prepare_worldcover():
    """
    Automates ESA WorldCover label download via GEE, aligns to Sentinel grid,
    saves aligned GeoTIFF, and triggers label validation.
    """
    # Ensure runtime directories exist
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    processed_labels_dir = PROCESSED_DIR / "labels"
    processed_labels_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_s1_path = PROCESSED_DIR / "sentinel1" / "sentinel1_processed.tif"
    output_label_path = processed_labels_dir / "worldcover_labels.tif"
    report_path = REPORTS_DIR / "label_preparation_report.txt"

    # Step 1: Read reference Sentinel raster metadata
    if not ref_s1_path.exists():
        print(f"[ERROR] Reference Sentinel-1 raster not found at: {ref_s1_path.resolve()}")
        print("Please run 'python src/data/preprocess.py' first.")
        return False

    print(f"[INFO] Reading reference raster bounds from: {ref_s1_path.resolve()}")
    with rasterio.open(ref_s1_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_width = ref_ds.width
        ref_height = ref_ds.height
        ref_bounds = ref_ds.bounds
        ref_res = (ref_transform.a, abs(ref_transform.e))

    print("--- Sentinel-1 Reference Grid Metadata ---")
    print(f"  Dimensions: {ref_width} x {ref_height}")
    print(f"  CRS: {ref_crs}")
    print(f"  Pixel Resolution: ({ref_res[0]:.6f}, {ref_res[1]:.6f})")
    print(f"  Bounds: {ref_bounds}")

    # Step 2: Initialize Earth Engine
    if not init_earth_engine():
        return False

    # Step 3: Define study area BBox in GEE
    bbox = [ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top]
    geometry = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])

    print("\n[INFO] Querying ESA WorldCover dataset in Earth Engine...")
    try:
        wc_collection = ee.ImageCollection("ESA/WorldCover/v200")
        if wc_collection.size().getInfo() == 0:
            wc_collection = ee.ImageCollection("ESA/WorldCover/v100")
    except Exception:
        wc_collection = ee.ImageCollection("ESA/WorldCover/v100")

    worldcover_img = wc_collection.first().select("Map").clip(geometry)

    # Step 4: Download dataset covering exact bounding box
    print("[INFO] Requesting raster download URL from Earth Engine...")
    try:
        download_url = worldcover_img.getDownloadURL({
            "name": "worldcover_gee",
            "bands": ["Map"],
            "region": geometry,
            "scale": 10,
            "crs": str(ref_crs),
            "format": "GEO_TIFF",
        })
    except Exception as e:
        print(f"[ERROR] Failed to obtain download URL from Earth Engine: {e}")
        return False

    print("[INFO] Downloading ESA WorldCover GeoTIFF dataset...")
    target_raw_tif = RAW_LABELS_DIR / "worldcover_gee_raw.tif"
    temp_download_file = RAW_LABELS_DIR / "worldcover_download.tmp"

    import shutil
    if temp_download_file.exists():
        temp_download_file.unlink()

    try:
        urllib.request.urlretrieve(download_url, temp_download_file)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        if temp_download_file.exists():
            temp_download_file.unlink()
        return False

    if not temp_download_file.exists() or temp_download_file.stat().st_size == 0:
        print("[ERROR] Downloaded file is missing or 0 bytes.")
        if temp_download_file.exists():
            temp_download_file.unlink()
        return False

    downloaded_tif_path = None

    # Check if the downloaded payload is a ZIP archive
    if zipfile.is_zipfile(temp_download_file):
        print("[INFO] Downloaded payload is a ZIP archive. Extracting contents...")
        temp_extract_dir = RAW_LABELS_DIR / "temp_extract_gee"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(temp_download_file, "r") as z:
                tif_members = [
                    m for m in z.namelist()
                    if m.lower().endswith((".tif", ".tiff")) and not m.startswith("__MACOSX")
                ]
                if not tif_members:
                    print("[ERROR] ZIP archive contains no GeoTIFF (.tif / .tiff) files.")
                    return False

                z.extractall(temp_extract_dir)

                extracted_tifs = sorted(
                    list(temp_extract_dir.glob("**/*.tif")) + list(temp_extract_dir.glob("**/*.tiff"))
                )
                if not extracted_tifs:
                    print("[ERROR] No GeoTIFF files found in extracted archive.")
                    return False

                # Sort by size descending to select primary raster
                extracted_tifs.sort(key=lambda f: f.stat().st_size, reverse=True)
                selected_tif = extracted_tifs[0]
                print(f"[INFO] Selected GeoTIFF from archive: {selected_tif.name}")

                if target_raw_tif.exists():
                    target_raw_tif.unlink()
                shutil.copy2(selected_tif, target_raw_tif)
                downloaded_tif_path = target_raw_tif
        except Exception as e:
            print(f"[ERROR] Failed to extract ZIP archive: {e}")
            return False
        finally:
            if temp_download_file.exists():
                temp_download_file.unlink()
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
    else:
        print("[INFO] Downloaded payload is a direct GeoTIFF raster.")
        try:
            with rasterio.open(temp_download_file) as test_ds:
                _ = test_ds.count

            if target_raw_tif.exists():
                target_raw_tif.unlink()
            shutil.move(str(temp_download_file), str(target_raw_tif))
            downloaded_tif_path = target_raw_tif
        except Exception as e:
            print(f"[ERROR] Downloaded file is corrupted or not a valid GeoTIFF raster: {e}")
            if temp_download_file.exists():
                temp_download_file.unlink()
            return False

    if downloaded_tif_path is None or not downloaded_tif_path.exists():
        print("[ERROR] Failed to locate extracted GeoTIFF file from download.")
        return False

    raw_tif_path = downloaded_tif_path
    print("\n[SUCCESS] ESA WorldCover GeoTIFF downloaded successfully\n")

    # Step 5: Inspect raw raster & reproject/align using Nearest Neighbor
    with rasterio.open(raw_tif_path) as raw_ds:
        raw_crs = raw_ds.crs
        raw_transform = raw_ds.transform
        raw_width = raw_ds.width
        raw_height = raw_ds.height
        raw_bounds = raw_ds.bounds
        raw_dtype = raw_ds.dtypes[0]
        raw_nodata = raw_ds.nodata
        raw_count = raw_ds.count
        raw_res = (raw_transform.a, abs(raw_transform.e))

        raw_data = raw_ds.read(1)
        raw_unique_classes = np.unique(raw_data)

        print("\n--- Downloaded ESA WorldCover Raster Inspection ---")
        print(f"  File Name: {raw_tif_path.name}")
        print(f"  Dimensions: {raw_width} x {raw_height}")
        print(f"  CRS: {raw_crs}")
        print(f"  Pixel Resolution: ({raw_res[0]:.6f}, {raw_res[1]:.6f})")
        print(f"  Bounds: {raw_bounds}")
        print(f"  Data Type: {raw_dtype}")
        print(f"  Unique Raw Classes: {list(raw_unique_classes)}")

        needs_reproject = (
            (raw_crs != ref_crs) or
            (raw_width != ref_width) or
            (raw_height != ref_height) or
            (raw_transform != ref_transform) or
            (raw_bounds != ref_bounds)
        )

        aligned_data = np.zeros((ref_height, ref_width), dtype=np.uint8)

        if needs_reproject:
            print("\n[INFO] Reprojecting and aligning GEE labels to Sentinel reference grid...")
            print("  Resampling Method: NEAREST NEIGHBOR (Preserves categorical integer class IDs)")
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
            print("\n[INFO] Raw GEE labels already match Sentinel reference grid perfectly.")
            aligned_data = raw_data.astype(np.uint8)
            resampling_method = "None (Raster already matches reference grid)"
            resampling_required = False

    # Step 6: Save aligned label GeoTIFF (Preserve categorical uint8 integer)
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

    print(f"\n[SUCCESS] Aligned labels saved to: {output_label_path.resolve()}")

    # Step 7: Generate Report
    proc_unique_classes, counts = np.unique(aligned_data, return_counts=True)
    total_pixels = aligned_data.size
    class_distribution = dict(zip(proc_unique_classes, counts))

    report_lines = [
        "===============================================================================",
        "FUSIONLAND AI: AUTOMATED ESA WORLDCOVER PREPARATION REPORT (STAGE 4A - GEE)",
        "===============================================================================\n",
        "1. INPUT & REFERENCE PATHS:",
        f"   - Downloaded GEE Label File:       {raw_tif_path.resolve()}",
        f"   - Reference Sentinel Raster File:  {ref_s1_path.resolve()}",
        f"   - Output Processed Label File:     {output_label_path.resolve()}\n",
        "2. RAW GEE RASTER METADATA:",
        f"   - Input File Name:      {raw_tif_path.name}",
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
        "   - Class ID Integrity:    Categorical Integer Preserved (uint8)\n",
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
        "[SUCCESS] Stage 4A Label download and preparation complete.",
        "===============================================================================",
    ])

    report_content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[SUCCESS] Label preparation report generated at: {report_path.resolve()}\n")

    # Step 8: Automatically run validation checks
    print("--- Automatically Triggering Stage 4A Validation Checks ---")
    from src.data.validate_labels import validate_labels
    val_passed = validate_labels()

    if val_passed:
        print("\n[SUCCESS] STAGE 4A COMPLETED SUCCESSFULLY")
        return True
    else:
        print("\n[FAIL] STAGE 4A NOT COMPLETE")
        return False


if __name__ == "__main__":
    download_and_prepare_worldcover()
