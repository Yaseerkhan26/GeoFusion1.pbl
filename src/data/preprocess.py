"""
===============================================================================
File: src/data/preprocess.py
Purpose: Preprocessing and Normalization for Sentinel-1 SAR and Sentinel-2 Optical Imagery

Description:
    - Loads raw Sentinel-1 and Sentinel-2 GeoTIFF rasters.
    - Detects invalid pixels (NaN / Inf).
    - Computes statistics (min, max, mean, std, 1st & 99th percentiles) on valid pixels.
    - Performs 1st-99th percentile clipping and Min-Max normalization to [0, 1].
    - Excludes invalid pixels.
    - Generates a combined binary valid pixel mask (1=valid, 0=invalid).
    - Preserves spatial metadata (CRS, Transform, Dimensions, Band Descriptions).
    - Saves processed GeoTIFF rasters and a comprehensive preprocessing report.
===============================================================================
"""

import os
import sys
import numpy as np
import rasterio
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    RAW_S1_DIR,
    RAW_S2_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)


def preprocess_sentinel1(s1_filepath, output_filepath):
    """
    Preprocesses Sentinel-1 SAR imagery (VV, VH bands):
    1. Detects NaN and infinite values.
    2. Calculates statistics on valid pixels only.
    3. Applies percentile clipping [1st, 99th].
    4. Applies Min-Max normalization to [0, 1].
    5. Preserves spatial metadata and writes output GeoTIFF.

    Args:
        s1_filepath (Path/str): Path to raw Sentinel-1 GeoTIFF.
        output_filepath (Path/str): Target path for processed GeoTIFF.

    Returns:
        tuple: (processed_array, valid_mask, stats_dict, meta_dict)
    """
    s1_filepath = Path(s1_filepath)
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not s1_filepath.exists():
        raise FileNotFoundError(f"Sentinel-1 raw file not found at {s1_filepath}")

    print(f"[INFO] Loading Sentinel-1 SAR raster: {s1_filepath}")
    with rasterio.open(s1_filepath) as src:
        data = src.read()  # Shape: (count, height, width)
        profile = src.profile.copy()
        band_descriptions = src.descriptions if src.descriptions and any(src.descriptions) else ("VV", "VH")

    num_bands, height, width = data.shape
    total_pixels_per_band = height * width

    # Detect valid pixels across all S1 bands
    valid_mask_per_band = np.isfinite(data)
    combined_s1_valid = np.all(valid_mask_per_band, axis=0)  # Shape: (height, width)

    processed_data = np.zeros_like(data, dtype=np.float32)
    stats_dict = {}

    for i in range(num_bands):
        band_name = band_descriptions[i] if band_descriptions[i] else f"Band_{i+1}"
        band_raw = data[i]
        band_valid_mask = valid_mask_per_band[i]

        valid_pixels = band_raw[band_valid_mask]
        n_valid = int(valid_pixels.size)
        n_invalid = int(total_pixels_per_band - n_valid)

        if n_valid == 0:
            raise ValueError(f"Sentinel-1 band {band_name} contains no valid pixels.")

        min_val = float(np.min(valid_pixels))
        max_val = float(np.max(valid_pixels))
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))

        p1 = float(np.percentile(valid_pixels, 1))
        p99 = float(np.percentile(valid_pixels, 99))

        # Clipping and Min-Max Normalization to [0, 1]
        clipped = np.clip(band_raw, p1, p99)
        if p99 > p1:
            norm_band = (clipped - p1) / (p99 - p1)
        else:
            norm_band = np.zeros_like(clipped)

        norm_band = np.clip(norm_band, 0.0, 1.0)
        norm_band[~band_valid_mask] = 0.0  # Keep invalid pixels excluded (0.0)

        processed_data[i] = norm_band.astype(np.float32)

        stats_dict[band_name] = {
            "total_pixels": total_pixels_per_band,
            "valid_pixels": n_valid,
            "invalid_pixels": n_invalid,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "p1": p1,
            "p99": p99,
            "norm_min": 0.0,
            "norm_max": 1.0,
        }

        print(
            f"  [S1 - {band_name}] Valid: {n_valid}/{total_pixels_per_band} | "
            f"Orig Range: [{min_val:.4f}, {max_val:.4f}] | "
            f"Percentiles [1%, 99%]: [{p1:.4f}, {p99:.4f}]"
        )

    # Save processed GeoTIFF preserving CRS, transform, dimensions, descriptions
    profile.update(dtype=rasterio.float32, count=num_bands, nodata=0.0)
    with rasterio.open(output_filepath, "w", **profile) as dst:
        for i in range(num_bands):
            dst.write(processed_data[i], i + 1)
            dst.set_band_description(i + 1, band_descriptions[i] or f"Band_{i+1}")

    print(f"[SUCCESS] Processed Sentinel-1 saved to: {output_filepath}")

    meta_dict = {
        "crs": str(profile.get("crs")),
        "transform": profile.get("transform"),
        "width": width,
        "height": height,
        "count": num_bands,
        "band_descriptions": band_descriptions,
    }

    return processed_data, combined_s1_valid, stats_dict, meta_dict


def preprocess_sentinel2(s2_filepath, output_filepath):
    """
    Preprocesses Sentinel-2 Optical imagery (B2, B3, B4, B8, B11, B12 bands):
    1. Detects NaN and infinite values.
    2. Calculates statistics on valid pixels only.
    3. Reports original value ranges without assuming reflectance scale.
    4. Applies percentile clipping [1st, 99th].
    5. Applies Min-Max normalization to [0, 1].
    6. Preserves spatial metadata and writes output GeoTIFF.

    Args:
        s2_filepath (Path/str): Path to raw Sentinel-2 GeoTIFF.
        output_filepath (Path/str): Target path for processed GeoTIFF.

    Returns:
        tuple: (processed_array, valid_mask, stats_dict, meta_dict)
    """
    s2_filepath = Path(s2_filepath)
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not s2_filepath.exists():
        raise FileNotFoundError(f"Sentinel-2 raw file not found at {s2_filepath}")

    print(f"[INFO] Loading Sentinel-2 Optical raster: {s2_filepath}")
    with rasterio.open(s2_filepath) as src:
        data = src.read()  # Shape: (count, height, width)
        profile = src.profile.copy()
        default_descs = ("B2", "B3", "B4", "B8", "B11", "B12")
        band_descriptions = src.descriptions if src.descriptions and any(src.descriptions) else default_descs

    num_bands, height, width = data.shape
    total_pixels_per_band = height * width

    # Detect valid pixels across all S2 bands
    valid_mask_per_band = np.isfinite(data)
    combined_s2_valid = np.all(valid_mask_per_band, axis=0)  # Shape: (height, width)

    processed_data = np.zeros_like(data, dtype=np.float32)
    stats_dict = {}

    for i in range(num_bands):
        band_name = band_descriptions[i] if band_descriptions[i] else f"Band_{i+1}"
        band_raw = data[i]
        band_valid_mask = valid_mask_per_band[i]

        valid_pixels = band_raw[band_valid_mask]
        n_valid = int(valid_pixels.size)
        n_invalid = int(total_pixels_per_band - n_valid)

        if n_valid == 0:
            raise ValueError(f"Sentinel-2 band {band_name} contains no valid pixels.")

        min_val = float(np.min(valid_pixels))
        max_val = float(np.max(valid_pixels))
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))

        p1 = float(np.percentile(valid_pixels, 1))
        p99 = float(np.percentile(valid_pixels, 99))

        # Clipping and Min-Max Normalization to [0, 1]
        clipped = np.clip(band_raw, p1, p99)
        if p99 > p1:
            norm_band = (clipped - p1) / (p99 - p1)
        else:
            norm_band = np.zeros_like(clipped)

        norm_band = np.clip(norm_band, 0.0, 1.0)
        norm_band[~band_valid_mask] = 0.0  # Keep invalid pixels excluded (0.0)

        processed_data[i] = norm_band.astype(np.float32)

        stats_dict[band_name] = {
            "total_pixels": total_pixels_per_band,
            "valid_pixels": n_valid,
            "invalid_pixels": n_invalid,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "p1": p1,
            "p99": p99,
            "norm_min": 0.0,
            "norm_max": 1.0,
        }

        print(
            f"  [S2 - {band_name}] Valid: {n_valid}/{total_pixels_per_band} | "
            f"Orig Range: [{min_val:.4f}, {max_val:.4f}] | "
            f"Percentiles [1%, 99%]: [{p1:.4f}, {p99:.4f}]"
        )

    # Save processed GeoTIFF preserving CRS, transform, dimensions, descriptions
    profile.update(dtype=rasterio.float32, count=num_bands, nodata=0.0)
    with rasterio.open(output_filepath, "w", **profile) as dst:
        for i in range(num_bands):
            dst.write(processed_data[i], i + 1)
            dst.set_band_description(i + 1, band_descriptions[i] or f"Band_{i+1}")

    print(f"[SUCCESS] Processed Sentinel-2 saved to: {output_filepath}")

    meta_dict = {
        "crs": str(profile.get("crs")),
        "transform": profile.get("transform"),
        "width": width,
        "height": height,
        "count": num_bands,
        "band_descriptions": band_descriptions,
    }

    return processed_data, combined_s2_valid, stats_dict, meta_dict


def save_combined_valid_mask(s1_valid_mask, s2_valid_mask, reference_filepath, mask_output_filepath):
    """
    Creates and saves a combined binary valid pixel mask GeoTIFF (1=valid, 0=invalid).
    A pixel is valid ONLY if valid in all S1 bands AND all S2 bands.

    Args:
        s1_valid_mask (np.ndarray): Boolean mask for S1 (height, width).
        s2_valid_mask (np.ndarray): Boolean mask for S2 (height, width).
        reference_filepath (Path/str): Raw GeoTIFF to copy spatial metadata from.
        mask_output_filepath (Path/str): Target path for valid mask GeoTIFF.

    Returns:
        np.ndarray: Combined binary valid mask (uint8).
    """
    mask_output_filepath = Path(mask_output_filepath)
    mask_output_filepath.parent.mkdir(parents=True, exist_ok=True)

    combined_mask = (s1_valid_mask & s2_valid_mask).astype(np.uint8)

    with rasterio.open(reference_filepath) as src:
        profile = src.profile.copy()

    profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(mask_output_filepath, "w", **profile) as dst:
        dst.write(combined_mask, 1)
        dst.set_band_description(1, "valid_mask")

    valid_count = int(np.sum(combined_mask == 1))
    invalid_count = int(np.sum(combined_mask == 0))
    total_count = int(combined_mask.size)

    print(f"[SUCCESS] Combined valid pixel mask saved to: {mask_output_filepath}")
    print(f"  Valid pixels (1): {valid_count} ({valid_count/total_count*100:.2f}%)")
    print(f"  Invalid pixels (0): {invalid_count} ({invalid_count/total_count*100:.2f}%)")

    return combined_mask, valid_count, invalid_count, total_count


def generate_preprocessing_report(
    s1_raw_path,
    s2_raw_path,
    s1_proc_path,
    s2_proc_path,
    mask_proc_path,
    s1_stats,
    s2_stats,
    valid_count,
    invalid_count,
    total_count,
    report_output_filepath,
):
    """
    Writes detailed text report summarizing preprocessing statistics and metadata.
    """
    report_output_filepath = Path(report_output_filepath)
    report_output_filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("===============================================================================")
    lines.append("FUSIONLAND AI: DATA PREPROCESSING AND VALIDATION REPORT (STAGE 3)")
    lines.append("===============================================================================\n")

    lines.append("1. INPUT FILES & PATHS:")
    lines.append(f"   - Sentinel-1 Raw: {s1_raw_path}")
    lines.append(f"   - Sentinel-2 Raw: {s2_raw_path}\n")

    lines.append("2. PROCESSED OUTPUT LOCATIONS:")
    lines.append(f"   - Sentinel-1 Processed: {s1_proc_path}")
    lines.append(f"   - Sentinel-2 Processed: {s2_proc_path}")
    lines.append(f"   - Combined Valid Pixel Mask: {mask_proc_path}\n")

    lines.append("3. COMBINED MASK PIXEL COUNT & COVERAGE:")
    lines.append(f"   - Total Pixels:   {total_count:,}")
    lines.append(f"   - Valid Pixels:   {valid_count:,} ({valid_count / total_count * 100:.2f}%)")
    lines.append(f"   - Invalid Pixels: {invalid_count:,} ({invalid_count / total_count * 100:.2f}%)\n")

    lines.append("4. SENTINEL-1 BAND STATISTICS & NORMALIZATION PARAMETERS:")
    lines.append("   Band | Valid Pixels | Invalid Pixels | Min Value | Max Value | Mean | Std | 1st Pct (Lower) | 99th Pct (Upper) | Norm Range")
    lines.append("   -----------------------------------------------------------------------------------------------------------------------")
    for band_name, s in s1_stats.items():
        lines.append(
            f"   {band_name:<4} | {s['valid_pixels']:<12,} | {s['invalid_pixels']:<14,} | "
            f"{s['min']:<9.4f} | {s['max']:<9.4f} | {s['mean']:<6.4f} | {s['std']:<6.4f} | "
            f"{s['p1']:<15.4f} | {s['p99']:<16.4f} | [{s['norm_min']:.1f}, {s['norm_max']:.1f}]"
        )
    lines.append("")

    lines.append("5. SENTINEL-2 BAND STATISTICS & NORMALIZATION PARAMETERS:")
    lines.append("   Band | Valid Pixels | Invalid Pixels | Min Value | Max Value | Mean | Std | 1st Pct (Lower) | 99th Pct (Upper) | Norm Range")
    lines.append("   -----------------------------------------------------------------------------------------------------------------------")
    for band_name, s in s2_stats.items():
        lines.append(
            f"   {band_name:<4} | {s['valid_pixels']:<12,} | {s['invalid_pixels']:<14,} | "
            f"{s['min']:<9.4f} | {s['max']:<9.4f} | {s['mean']:<6.4f} | {s['std']:<6.4f} | "
            f"{s['p1']:<15.4f} | {s['p99']:<16.4f} | [{s['norm_min']:.1f}, {s['norm_max']:.1f}]"
        )
    lines.append("")

    lines.append("6. SUMMARY & METHODOLOGY:")
    lines.append("   - Detect NaNs and infinite values across all bands.")
    lines.append("   - Exclude invalid background/nodata pixels from percentiles and statistics calculation.")
    lines.append("   - Apply 1st and 99th percentile clipping to eliminate outliers and extreme values.")
    lines.append("   - Min-Max normalization applied: normalized = (clipped - p1) / (p99 - p1).")
    lines.append("   - Excluded invalid pixels retained as 0.0 in processed arrays and 0 in binary mask.")
    lines.append("   - Spatial metadata (CRS EPSG:4326, Transform, Dimensions 4112x4008, Band Descriptions) fully preserved.")
    lines.append("\n===============================================================================")

    report_content = "\n".join(lines)
    with open(report_output_filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[SUCCESS] Preprocessing report generated at: {report_output_filepath}")


def run_preprocessing_pipeline():
    """
    Main orchestration function for Stage 3 Preprocessing.
    """
    s1_raw = RAW_S1_DIR / "Sentinel1_VV_VH.tif"
    s2_raw = RAW_S2_DIR / "Sentinel2_Bands.tif"

    s1_proc = PROCESSED_DIR / "sentinel1" / "sentinel1_processed.tif"
    s2_proc = PROCESSED_DIR / "sentinel2" / "sentinel2_processed.tif"
    mask_proc = PROCESSED_DIR / "masks" / "valid_pixel_mask.tif"
    report_proc = REPORTS_DIR / "preprocessing_report.txt"

    print("=== Starting Stage 3 Data Preprocessing Pipeline ===")

    # 1. Preprocess Sentinel-1
    s1_arr, s1_mask, s1_stats, s1_meta = preprocess_sentinel1(s1_raw, s1_proc)

    # 2. Preprocess Sentinel-2
    s2_arr, s2_mask, s2_stats, s2_meta = preprocess_sentinel2(s2_raw, s2_proc)

    # 3. Create combined valid pixel mask
    combined_mask, valid_cnt, invalid_cnt, total_cnt = save_combined_valid_mask(
        s1_mask, s2_mask, s1_raw, mask_proc
    )

    # 4. Generate Preprocessing Report
    generate_preprocessing_report(
        s1_raw_path=s1_raw,
        s2_raw_path=s2_raw,
        s1_proc_path=s1_proc,
        s2_proc_path=s2_proc,
        mask_proc_path=mask_proc,
        s1_stats=s1_stats,
        s2_stats=s2_stats,
        valid_count=valid_cnt,
        invalid_count=invalid_cnt,
        total_count=total_cnt,
        report_output_filepath=report_proc,
    )

    print("=== Stage 3 Data Preprocessing Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_preprocessing_pipeline()
