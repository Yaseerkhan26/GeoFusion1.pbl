"""
===============================================================================
Script: check_data.py
Project: Multimodal Satellite Data Fusion for Enhanced Land Cover Classification
         Using Sentinel-1 and Sentinel-2

Description:
    Reads, inspects, aligns, and visualizes Sentinel-1 (SAR) and Sentinel-2 (Optical)
    GeoTIFF datasets using rasterio and matplotlib.

Requirements Handled:
    1. Reads GeoTIFF files (data/Sentinel1_VV_VH.tif & data/Sentinel2_Bands.tif)
    2. Prints number of bands
    3. Prints width and height
    4. Prints CRS and transform matrix
    5. Checks if spatial dimensions match
    6. Checks geographic alignment (CRS, transform, bounds)
    7. Displays Sentinel-1 VV and VH bands
    8. Displays Sentinel-2 RGB composite image using B4 (Red), B3 (Green), B2 (Blue)
    9. Saves textual report and figure visualizations
===============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def normalize_band(band_data: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
    """
    Percentile stretch normalization for visual display.
    """
    valid_pixels = band_data[~np.isnan(band_data)]
    if valid_pixels.size == 0:
        return np.zeros_like(band_data, dtype=np.float32)
    
    vmin, vmax = np.percentile(valid_pixels, (p_min, p_max))
    if vmin == vmax:
        return np.zeros_like(band_data, dtype=np.float32)
    
    normalized = (band_data - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def main():
    print("===============================================================================")
    print(" MULTIMODAL SATELLITE DATASET INSPECTION & ALIGNMENT CHECK")
    print("===============================================================================\n")

    # 1. Define and check file paths
    base_dir = Path(__file__).resolve().parent
    s1_path = base_dir / "data" / "Sentinel1_VV_VH.tif"
    s2_path = base_dir / "data" / "Sentinel2_Bands.tif"

    if not s1_path.exists():
        print(f"[ERROR] Sentinel-1 file not found at: {s1_path}")
        sys.exit(1)
    if not s2_path.exists():
        print(f"[ERROR] Sentinel-2 file not found at: {s2_path}")
        sys.exit(1)

    print(f"Reading Sentinel-1 dataset: {s1_path}")
    print(f"Reading Sentinel-2 dataset: {s2_path}\n")

    # Read both GeoTIFF files
    s1 = rasterio.open(s1_path)
    s2 = rasterio.open(s2_path)

    # Output report string list for saving to file
    report_lines = []
    report_lines.append("MULTIMODAL SATELLITE DATASET INSPECTION REPORT")
    report_lines.append("===============================================================================\n")

    # 2. Number of bands
    s1_count = s1.count
    s2_count = s2.count
    report_lines.append(f"1. Number of Bands:")
    report_lines.append(f"   - Sentinel-1 (SAR): {s1_count} bands {s1.descriptions if s1.descriptions else ''}")
    report_lines.append(f"   - Sentinel-2 (Optical): {s2_count} bands {s2.descriptions if s2.descriptions else ''}\n")

    # 3. Width and Height
    s1_dims = (s1.width, s1.height)
    s2_dims = (s2.width, s2.height)
    report_lines.append(f"2. Spatial Dimensions (Width x Height):")
    report_lines.append(f"   - Sentinel-1: {s1.width} x {s1.height} pixels")
    report_lines.append(f"   - Sentinel-2: {s2.width} x {s2.height} pixels\n")

    # 4. CRS and Transform
    report_lines.append(f"3. Coordinate Reference System (CRS) & Affine Transform:")
    report_lines.append(f"   - Sentinel-1 CRS: {s1.crs}")
    report_lines.append(f"   - Sentinel-1 Transform:\n     {s1.transform}")
    report_lines.append(f"   - Sentinel-2 CRS: {s2.crs}")
    report_lines.append(f"   - Sentinel-2 Transform:\n     {s2.transform}\n")

    # 5. Check if both images have the same spatial dimensions
    same_dims = (s1_dims == s2_dims)
    report_lines.append(f"4. Spatial Dimensions Match Check:")
    report_lines.append(f"   - Same Dimensions? {'PASSED (YES)' if same_dims else 'FAILED (NO)'}")
    report_lines.append(f"   - Details: S1={s1_dims} | S2={s2_dims}\n")

    # 6. Check if both images are geographically aligned
    same_crs = (s1.crs == s2.crs)
    same_transform = (s1.transform == s2.transform)
    same_bounds = (s1.bounds == s2.bounds)

    is_aligned = same_dims and same_crs and same_transform and same_bounds

    report_lines.append(f"5. Geographic Alignment Check:")
    report_lines.append(f"   - Same CRS? {'YES' if same_crs else 'NO'}")
    report_lines.append(f"   - Same Transform Matrix? {'YES' if same_transform else 'NO'}")
    report_lines.append(f"   - Same Bounding Box? {'YES' if same_bounds else 'NO'}")
    report_lines.append(f"   - Geographically Aligned for Fusion? {'PASSED (YES)' if is_aligned else 'FAILED (NO)'}\n")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Load raster bands into memory for display
    print("Loading bands for visual output...")
    # Sentinel-1 bands: Band 1 = VV, Band 2 = VH
    vv_band = s1.read(1)
    vh_band = s2_count and s1.read(2)

    # Sentinel-2 bands: Band 3 = B4 (Red), Band 2 = B3 (Green), Band 1 = B2 (Blue)
    # Band descriptions in file: ('B2', 'B3', 'B4', 'B8', 'B11', 'B12')
    # Band 1: B2 (Blue), Band 2: B3 (Green), Band 3: B4 (Red)
    b4_red = s2.read(3)
    b3_green = s2.read(2)
    b2_blue = s2.read(1)

    # Normalize individual bands for plotting
    vv_norm = normalize_band(vv_band)
    vh_norm = normalize_band(vh_band)

    red_norm = normalize_band(b4_red)
    green_norm = normalize_band(b3_green)
    blue_norm = normalize_band(b2_blue)

    # Stack RGB image (Height, Width, 3)
    rgb_composite = np.dstack((red_norm, green_norm, blue_norm))

    # Output directory setup
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_reports = output_dir / "reports"
    outputs_reports.mkdir(parents=True, exist_ok=True)

    # 7 & 8. Display & Save Figures
    print("\nGenerating visualization plots...")

    # Figure 1: Sentinel-1 VV & VH Bands
    fig_s1, axes_s1 = plt.subplots(1, 2, figsize=(14, 7))
    
    im_vv = axes_s1[0].imshow(vv_norm, cmap='gray')
    axes_s1[0].set_title("Sentinel-1 SAR: VV Polarization", fontsize=14, fontweight='bold')
    axes_s1[0].set_xlabel("Pixel Column")
    axes_s1[0].set_ylabel("Pixel Row")
    plt.colorbar(im_vv, ax=axes_s1[0], fraction=0.046, pad=0.04, label="Normalized Intensity")

    im_vh = axes_s1[1].imshow(vh_norm, cmap='gray')
    axes_s1[1].set_title("Sentinel-1 SAR: VH Polarization", fontsize=14, fontweight='bold')
    axes_s1[1].set_xlabel("Pixel Column")
    axes_s1[1].set_ylabel("Pixel Row")
    plt.colorbar(im_vh, ax=axes_s1[1], fraction=0.046, pad=0.04, label="Normalized Intensity")

    plt.tight_layout()
    fig_s1_path = output_dir / "sentinel1_vv_vh.png"
    plt.savefig(fig_s1_path, dpi=300, bbox_inches='tight')
    print(f"Saved Sentinel-1 visualization: {fig_s1_path}")

    # Figure 2: Sentinel-2 RGB Composite (B4, B3, B2)
    fig_s2, ax_s2 = plt.subplots(figsize=(8, 8))
    ax_s2.imshow(rgb_composite)
    ax_s2.set_title("Sentinel-2 Optical RGB Composite (Bands 4, 3, 2)", fontsize=14, fontweight='bold')
    ax_s2.set_xlabel("Pixel Column")
    ax_s2.set_ylabel("Pixel Row")
    
    plt.tight_layout()
    fig_s2_path = output_dir / "sentinel2_rgb.png"
    plt.savefig(fig_s2_path, dpi=300, bbox_inches='tight')
    print(f"Saved Sentinel-2 RGB visualization: {fig_s2_path}")

    # Figure 3: Combined Multimodal Overview (Sentinel-1 VV, VH + Sentinel-2 RGB)
    fig_combined, axes_c = plt.subplots(1, 3, figsize=(18, 6))
    
    axes_c[0].imshow(vv_norm, cmap='magma')
    axes_c[0].set_title("Sentinel-1 (VV Band)", fontsize=12, fontweight='bold')
    axes_c[0].axis('off')

    axes_c[1].imshow(vh_norm, cmap='magma')
    axes_c[1].set_title("Sentinel-1 (VH Band)", fontsize=12, fontweight='bold')
    axes_c[1].axis('off')

    axes_c[2].imshow(rgb_composite)
    axes_c[2].set_title("Sentinel-2 (RGB: B4, B3, B2)", fontsize=12, fontweight='bold')
    axes_c[2].axis('off')

    plt.suptitle("Multimodal Satellite Data Check: Sentinel-1 & Sentinel-2", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    fig_comb_path = output_dir / "check_data_visualization.png"
    plt.savefig(fig_comb_path, dpi=300, bbox_inches='tight')
    print(f"Saved Combined visual summary: {fig_comb_path}")

    # Close GeoTIFF readers
    s1.close()
    s2.close()

    # 9. Save Results
    report_file_path = output_dir / "check_data_report.txt"
    report_file_path2 = outputs_reports / "check_data_report.txt"
    
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(report_file_path2, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved textual check report to: {report_file_path}")
    print("\n===============================================================================")
    print(" ALL CHECKS COMPLETED SUCCESSFULLY!")
    print("===============================================================================\n")

    # Display plot windows
    plt.show()


if __name__ == "__main__":
    main()
