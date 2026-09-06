"""
===============================================================================
File: src/data/create_patches.py
Purpose: Spatial Window Tiling and Dataset Patch Generation

Description:
    Tiles large co-registered multimodal rasters (Sentinel-1, Sentinel-2) and ground
    truth label maps into uniform spatial patches (e.g., 128x128 or 256x256).
    Saves patch pairs to data/patches/ for deep learning model training.
===============================================================================
"""

import sys
import numpy as np
import rasterio
from pathlib import Path

# Append project root to system path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import PATCHES_DIR, PATCH_SIZE, PATCH_STRIDE


def extract_patches_from_raster(raster_path, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """
    Sliding window patch extraction algorithm for large spatial rasters.

    Args:
        raster_path (str/Path): Path to input processed raster GeoTIFF.
        patch_size (int): Height and width of square patch (default: 128).
        stride (int): Overlap stride between consecutive patches (default: 64).

    Returns:
        list of np.ndarray: List of extracted patches of shape (Channels, patch_size, patch_size).
    """
    print(f"[STARTER] Tiling raster into {patch_size}x{patch_size} patches with stride {stride}: {raster_path}")
    patches = []
    # TODO: Open raster with rasterio, iterate windows across height and width, extract valid patches
    return patches


def generate_multimodal_dataset(s1_raster_path, s2_raster_path, label_raster_path):
    """
    Synchronously extracts aligned spatial patch triplets (Sentinel-1, Sentinel-2, Ground Truth)
    and saves them to data/patches/ as structured numpy files (.npz or .npy).

    Args:
        s1_raster_path (Path): Preprocessed Sentinel-1 SAR raster.
        s2_raster_path (Path): Preprocessed Sentinel-2 Optical raster.
        label_raster_path (Path): Land cover ground truth label raster.

    Returns:
        int: Total number of patch samples generated.
    """
    print(f"[STARTER] Generating synchronized patch dataset in: {PATCHES_DIR}")
    # TODO: Extract aligned patch triplets, check nodata threshold, save to disk
    sample_count = 0
    return sample_count


if __name__ == "__main__":
    print("=== Spatial Patch Creation Module ===")
    print(f"[INFO] Patch Output Directory: {PATCHES_DIR}")
    print(f"[INFO] Configured Patch Size: {PATCH_SIZE}x{PATCH_SIZE}, Stride: {PATCH_STRIDE}")
