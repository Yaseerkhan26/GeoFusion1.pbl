"""
===============================================================================
File: src/data/download_data.py
Purpose: Google Earth Engine (GEE) Data Acquisition Script for Sentinel-1 & 2

Description:
    Authenticates with Google Earth Engine API, defines a Region of Interest (ROI)
    and time range, and downloads co-registered Sentinel-1 SAR (VV/VH) and
    Sentinel-2 MSI (Level-2A) image collections to local raw storage directories.
===============================================================================
"""

import sys
from pathlib import Path

# Append project root to system path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import RAW_S1_DIR, RAW_S2_DIR, SENTINEL1_BANDS, SENTINEL2_BANDS


def initialize_earth_engine():
    """
    Initializes and authenticates the Google Earth Engine Python API connection.
    """
    try:
        import ee
        ee.Initialize()
        print("[INFO] Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"[WARNING] Earth Engine initialization failed: {e}")
        print("[INFO] Please run `earthengine authenticate` in your terminal if using GEE downloading.")


def fetch_sentinel1_sar(roi_geojson, start_date, end_date):
    """
    Queries Sentinel-1 Ground Range Detected (GRD) SAR imagery from GEE.

    Args:
        roi_geojson (dict/str): GeoJSON dictionary or bounding box coordinates for ROI.
        start_date (str): Start date formatted as 'YYYY-MM-DD'.
        end_date (str): End date formatted as 'YYYY-MM-DD'.

    Returns:
        None (Saves exported GeoTIFF files to data/raw/sentinel1/).
    """
    print(f"[STARTER] Querying Sentinel-1 SAR (VV/VH) imagery from {start_date} to {end_date}...")
    print(f"[STARTER] Destination Directory: {RAW_S1_DIR}")
    # TODO: Implement GEE Sentinel-1 collection filter, speckle filtering, and export task


def fetch_sentinel2_optical(roi_geojson, start_date, end_date, cloud_threshold=10):
    """
    Queries Sentinel-2 Surface Reflectance (Level-2A) optical imagery from GEE.

    Args:
        roi_geojson (dict/str): GeoJSON dictionary or bounding box coordinates for ROI.
        start_date (str): Start date formatted as 'YYYY-MM-DD'.
        end_date (str): End date formatted as 'YYYY-MM-DD'.
        cloud_threshold (float): Maximum allowed percentage of cloud cover.

    Returns:
        None (Saves exported GeoTIFF files to data/raw/sentinel2/).
    """
    print(f"[STARTER] Querying Sentinel-2 Optical imagery (<{cloud_threshold}% clouds) from {start_date} to {end_date}...")
    print(f"[STARTER] Destination Directory: {RAW_S2_DIR}")
    # TODO: Implement GEE Sentinel-2 collection filter, cloud masking, median composite, and export task


if __name__ == "__main__":
    print("=== Multimodal Data Acquisition Module ===")
    initialize_earth_engine()
    
    # Example placeholder parameters
    sample_roi = [77.5946, 12.9716, 77.7500, 13.1000]  # [Min Lon, Min Lat, Max Lon, Max Lat]
    sample_start = "2024-01-01"
    sample_end = "2024-03-31"

    fetch_sentinel1_sar(sample_roi, sample_start, sample_end)
    fetch_sentinel2_optical(sample_roi, sample_start, sample_end)
