"""
===============================================================================
File: src/data/inspect_data.py
Project: FusionLand AI: Multimodal Satellite Data Fusion for Enhanced Land Cover Classification
Purpose: Dataset Inspection Script for Sentinel-1 (SAR) and Sentinel-2 (Optical) GeoTIFFs

Description:
    Automatically searches data/raw/sentinel1/ and data/raw/sentinel2/ for GeoTIFF files.
    Inspects key metadata (Bands, Dimensions, CRS, Resolution, Bounds, Affine Transform, Data Types, NoData values).
    Calculates band-by-band statistics (Min, Max, Mean, NoData count) without loading full raster at once.
    Compares Sentinel-1 and Sentinel-2 datasets for fusion compatibility (CRS, Resolution, Dimensions, Bounds Overlap).
    Generates a structured report on stdout and saves it to outputs/reports/dataset_inspection.txt.
===============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import rasterio

# Base directory setup and path resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_S1_DIR = BASE_DIR / "data" / "raw" / "sentinel1"
RAW_S2_DIR = BASE_DIR / "data" / "raw" / "sentinel2"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


def find_geotiff_files(directory: Path) -> List[Path]:
    """
    Search directory for GeoTIFF files (.tif, .tiff, .TIF, .TIFF).

    Args:
        directory (Path): Path to raw dataset directory.

    Returns:
        List[Path]: Sorted list of unique GeoTIFF file paths found.
    """
    if not directory.exists():
        return []
    
    extensions = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    found_files = []
    for ext in extensions:
        found_files.extend(list(directory.glob(ext)))
    
    # Return sorted unique list
    return sorted(list(set(found_files)))


def calculate_band_statistics(src: rasterio.DatasetReader, band_idx: int) -> Dict[str, Any]:
    """
    Calculate statistics for a single band without loading all bands into memory at once.

    Args:
        src (rasterio.DatasetReader): Opened rasterio dataset object.
        band_idx (int): 1-indexed band number.

    Returns:
        Dict[str, Any]: Dictionary containing min, max, mean, and nodata count.
    """
    # Read only the target band
    band_data = src.read(band_idx)
    
    # Identify NoData value
    nodata_val = None
    if src.nodatavals and band_idx <= len(src.nodatavals):
        nodata_val = src.nodatavals[band_idx - 1]
    if nodata_val is None:
        nodata_val = src.nodata

    # Create NoData mask
    if nodata_val is not None:
        if np.isnan(nodata_val):
            nodata_mask = np.isnan(band_data)
        else:
            nodata_mask = (band_data == nodata_val)
    else:
        nodata_mask = np.zeros(band_data.shape, dtype=bool)

    # Check for additional NaNs / Infs in floating point rasters
    if np.issubdtype(band_data.dtype, np.floating):
        nodata_mask = nodata_mask | np.isnan(band_data) | np.isinf(band_data)

    nodata_count = int(np.sum(nodata_mask))
    valid_pixels = band_data[~nodata_mask]

    if valid_pixels.size > 0:
        min_val = float(np.min(valid_pixels))
        max_val = float(np.max(valid_pixels))
        mean_val = float(np.mean(valid_pixels))
    else:
        min_val = float("nan")
        max_val = float("nan")
        mean_val = float("nan")

    return {
        "band_number": band_idx,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "nodata_count": nodata_count,
        "valid_pixels": int(valid_pixels.size),
        "total_pixels": int(band_data.size)
    }


def inspect_geotiff_file(file_path: Path) -> Dict[str, Any]:
    """
    Inspect metadata and band statistics of a GeoTIFF file.

    Args:
        file_path (Path): Path to GeoTIFF file.

    Returns:
        Dict[str, Any]: Comprehensive metadata and per-band statistics dictionary.
    """
    try:
        with rasterio.open(file_path) as src:
            descriptions = [
                desc if desc is not None else f"Band {i+1}"
                for i, desc in enumerate(src.descriptions)
            ] if src.descriptions else [f"Band {i+1}" for i in range(src.count)]

            nodatavals = list(src.nodatavals) if src.nodatavals else [src.nodata] * src.count

            # Calculate per-band statistics
            band_stats = []
            for b in range(1, src.count + 1):
                stats = calculate_band_statistics(src, b)
                band_stats.append(stats)

            file_info = {
                "file_name": file_path.name,
                "full_path": str(file_path.resolve()),
                "num_bands": src.count,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "resolution": src.res,  # (res_x, res_y)
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "bounds_tuple": (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                "transform": str(src.transform),
                "data_type": list(src.dtypes),
                "nodata_value": nodatavals,
                "band_descriptions": descriptions,
                "band_stats": band_stats
            }
            return file_info
    except Exception as e:
        return {
            "file_name": file_path.name,
            "full_path": str(file_path.resolve()),
            "error": str(e)
        }


def check_bounds_overlap(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> bool:
    """
    Check if two bounding boxes (left, bottom, right, top) overlap.

    Args:
        b1 (Tuple): Bounds (l, b, r, t) of raster 1
        b2 (Tuple): Bounds (l, b, r, t) of raster 2

    Returns:
        bool: True if bounds overlap, False otherwise.
    """
    l1, bot1, r1, top1 = b1
    l2, bot2, r2, top2 = b2

    overlap_x = (l1 < r2) and (r1 > l2)
    overlap_y = (bot1 < top2) and (top1 > bot2)

    return overlap_x and overlap_y


def format_file_report(info: Dict[str, Any], title: str) -> str:
    """
    Format metadata and per-band statistics report string for a dataset file.

    Args:
        info (Dict[str, Any]): Inspected metadata dictionary.
        title (str): Section title header.

    Returns:
        str: Formatted report text.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"{title}")
    lines.append("=" * 60)

    if "error" in info:
        lines.append(f"File: {info['file_name']}")
        lines.append(f"Full Path: {info['full_path']}")
        lines.append(f"ERROR READING FILE: {info['error']}\n")
        return "\n".join(lines)

    lines.append(f"File: {info['file_name']}")
    lines.append(f"Full Path: {info['full_path']}")
    lines.append(f"Number of Bands: {info['num_bands']}")
    lines.append(f"Dimensions: {info['width']} x {info['height']} (Width x Height)")
    lines.append(f"CRS: {info['crs']}")
    lines.append(f"Resolution: {info['resolution']} (Pixel width, Pixel height)")
    b = info['bounds']
    lines.append(f"Bounds: Left={b['left']}, Bottom={b['bottom']}, Right={b['right']}, Top={b['top']}")
    lines.append(f"Affine Transform:\n{info['transform']}")
    lines.append(f"Data Type(s): {info['data_type']}")
    lines.append(f"NoData Value(s): {info['nodata_value']}")
    lines.append(f"Band Descriptions: {info['band_descriptions']}")

    lines.append("\n--- BAND STATISTICS ---")
    for stat in info["band_stats"]:
        b_idx = stat["band_number"]
        desc = info["band_descriptions"][b_idx - 1] if b_idx <= len(info["band_descriptions"]) else ""
        lines.append(
            f"  Band {b_idx} ({desc}): "
            f"Min={stat['min']:.4f}, "
            f"Max={stat['max']:.4f}, "
            f"Mean={stat['mean']:.4f}, "
            f"NoData Pixels={stat['nodata_count']} / {stat['total_pixels']}"
        )
    lines.append("")
    return "\n".join(lines)


def run_dataset_inspection() -> str:
    """
    Execute full inspection workflow across Sentinel-1 and Sentinel-2 folders,
    compare datasets, print clear output, and save to report file.

    Returns:
        str: Complete dataset inspection report content.
    """
    report_lines = []
    
    # 1. Locate files
    s1_files = find_geotiff_files(RAW_S1_DIR)
    s2_files = find_geotiff_files(RAW_S2_DIR)

    report_lines.append("FUSIONLAND AI - MULTIMODAL SATELLITE DATASET INSPECTION")
    report_lines.append("=" * 60)
    report_lines.append(f"Sentinel-1 Directory: {RAW_S1_DIR}")
    report_lines.append(f"Sentinel-1 TIFF Files Found: {len(s1_files)}")
    report_lines.append(f"Sentinel-2 Directory: {RAW_S2_DIR}")
    report_lines.append(f"Sentinel-2 TIFF Files Found: {len(s2_files)}")
    report_lines.append("")

    s1_infos = []
    s2_infos = []

    # 2. Inspect Sentinel-1 files
    if s1_files:
        for idx, file_path in enumerate(s1_files, 1):
            info = inspect_geotiff_file(file_path)
            s1_infos.append(info)
            header = f"SENTINEL-1 DATASET (File {idx}/{len(s1_files)})"
            report_lines.append(format_file_report(info, header))
    else:
        report_lines.append("=" * 60)
        report_lines.append("SENTINEL-1 DATASET")
        report_lines.append("=" * 60)
        report_lines.append("No GeoTIFF (.tif / .tiff) files found in data/raw/sentinel1/\n")

    # 3. Inspect Sentinel-2 files
    if s2_files:
        for idx, file_path in enumerate(s2_files, 1):
            info = inspect_geotiff_file(file_path)
            s2_infos.append(info)
            header = f"SENTINEL-2 DATASET (File {idx}/{len(s2_files)})"
            report_lines.append(format_file_report(info, header))
    else:
        report_lines.append("=" * 60)
        report_lines.append("SENTINEL-2 DATASET")
        report_lines.append("=" * 60)
        report_lines.append("No GeoTIFF (.tif / .tiff) files found in data/raw/sentinel2/\n")

    # 4. Compare Sentinel-1 and Sentinel-2 datasets
    report_lines.append("=" * 60)
    report_lines.append("DATASET COMPARISON")
    report_lines.append("=" * 60)

    if s1_infos and s2_infos and "error" not in s1_infos[0] and "error" not in s2_infos[0]:
        s1 = s1_infos[0]
        s2 = s2_infos[0]

        # CRS Check
        crs_match = (s1["crs"] == s2["crs"])
        crs_str = "YES" if crs_match else f"NO (S1: {s1['crs']} vs S2: {s2['crs']})"

        # Resolution Check
        res_match = np.isclose(s1["resolution"], s2["resolution"]).all()
        res_str = "YES" if res_match else f"NO (S1: {s1['resolution']} vs S2: {s2['resolution']})"

        # Dimensions Check
        dim_match = (s1["width"] == s2["width"]) and (s1["height"] == s2["height"])
        dim_str = "YES" if dim_match else f"NO (S1: {s1['width']}x{s1['height']} vs S2: {s2['width']}x{s2['height']})"

        # Bounds Overlap Check
        bounds_overlap = check_bounds_overlap(s1["bounds_tuple"], s2["bounds_tuple"])
        bounds_str = "YES" if bounds_overlap else "NO"

        # Ready for Fusion Check
        ready_for_fusion = crs_match and res_match and dim_match and bounds_overlap
        ready_str = "YES" if ready_for_fusion else "NO"

        report_lines.append(f"CRS Match: {crs_str}")
        report_lines.append(f"Resolution Match: {res_str}")
        report_lines.append(f"Dimensions Match: {dim_str}")
        report_lines.append(f"Bounds Overlap: {bounds_str}")
        report_lines.append(f"Ready for Fusion: {ready_str}")
        
        if not ready_for_fusion:
            report_lines.append("\nNote / Action Required:")
            if not crs_match:
                report_lines.append(" - Reprojection needed to align CRS.")
            if not res_match:
                report_lines.append(" - Resampling needed to match spatial resolution (e.g., 10m grid).")
            if not dim_match:
                report_lines.append(" - Spatial alignment/cropping needed to match raster dimensions.")
            if not bounds_overlap:
                report_lines.append(" - Warning: Spatial bounding boxes do not overlap!")

    elif not s1_files or not s2_files:
        report_lines.append("CRS Match: UNKNOWN (Missing dataset files)")
        report_lines.append("Resolution Match: UNKNOWN (Missing dataset files)")
        report_lines.append("Dimensions Match: UNKNOWN (Missing dataset files)")
        report_lines.append("Bounds Overlap: UNKNOWN (Missing dataset files)")
        report_lines.append("Ready for Fusion: NO (Please place GeoTIFF files in data/raw/sentinel1/ and data/raw/sentinel2/)")
    else:
        report_lines.append("CRS Match: ERROR")
        report_lines.append("Resolution Match: ERROR")
        report_lines.append("Dimensions Match: ERROR")
        report_lines.append("Bounds Overlap: ERROR")
        report_lines.append("Ready for Fusion: NO (Failed to read raster file metadata)")

    report_lines.append("\n" + "=" * 60)
    report_content = "\n".join(report_lines)

    # 5. Save report to file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "dataset_inspection.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


if __name__ == "__main__":
    full_report = run_dataset_inspection()
    print(full_report)
    print(f"\nReport successfully saved to: {REPORTS_DIR / 'dataset_inspection.txt'}")
