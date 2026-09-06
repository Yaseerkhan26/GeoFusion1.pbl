"""
===============================================================================
File: src/data/preprocess.py
Purpose: Radiometric, Atmospheric, and Geometric Preprocessing for Multimodal Rasters

Description:
    Processes raw Sentinel-1 SAR and Sentinel-2 Optical rasters:
    1. Resampling and spatial alignment to a common grid & CRS (e.g., EPSG:4326 / UTM).
    2. Speckle filtering and dB amplitude scaling for Sentinel-1 VV/VH channels.
    3. Surface reflectance scaling (0-1 range) and cloud masking for Sentinel-2 bands.
    4. Exporting stacked multimodal rasters into data/processed/.
===============================================================================
"""

import sys
import numpy as np
import rasterio
from pathlib import Path

# Append project root to system path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import PROCESSED_DIR, SPATIAL_RESOLUTION, TARGET_EPSG


def preprocess_sentinel1(s1_filepath):
    """
    Applies speckle noise reduction (e.g., Lee Filter) and converts intensity
    values to decibels (dB) for Sentinel-1 SAR imagery.

    Args:
        s1_filepath (str/Path): Path to raw Sentinel-1 GeoTIFF file.

    Returns:
        np.ndarray: Preprocessed SAR array (Channels x Height x Width).
    """
    print(f"[STARTER] Preprocessing Sentinel-1 SAR raster: {s1_filepath}")
    # TODO: Open with rasterio, apply Lee filter, convert to dB (10 * log10(DN)), normalize
    return None


def preprocess_sentinel2(s2_filepath):
    """
    Normalizes optical surface reflectance values, applies cloud QA masking,
    and resampless 20m bands to 10m spatial resolution.

    Args:
        s2_filepath (str/Path): Path to raw Sentinel-2 GeoTIFF file.

    Returns:
        np.ndarray: Preprocessed Optical array (Bands x Height x Width).
    """
    print(f"[STARTER] Preprocessing Sentinel-2 Optical raster: {s2_filepath}")
    # TODO: Open with rasterio, scale reflectance (DN / 10000.0), resample 20m bands (B05-B07, B8A, B11-B12)
    return None


def align_and_stack_multimodal(s1_processed, s2_processed, output_filename="multimodal_stack.tif"):
    """
    Aligns bounding boxes and pixel grids of preprocessed Sentinel-1 and Sentinel-2
    rasters, creating a unified multimodal stack (e.g., 12-channel raster).

    Args:
        s1_processed (np.ndarray): SAR feature tensor.
        s2_processed (np.ndarray): Optical feature tensor.
        output_filename (str): Target GeoTIFF filename.

    Returns:
        Path: Path to saved stacked raster in data/processed/.
    """
    output_path = PROCESSED_DIR / output_filename
    print(f"[STARTER] Aligning and stacking multimodal rasters -> {output_path}")
    # TODO: Perform spatial reprojection match using rasterio.warp and write combined GeoTIFF
    return output_path


if __name__ == "__main__":
    print("=== Multimodal Data Preprocessing Module ===")
    print(f"[INFO] Processed output directory ready: {PROCESSED_DIR}")
