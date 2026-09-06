"""
===============================================================================
File: src/utils/config.py
Purpose: Centralized Configuration Management for Multimodal Satellite Data Fusion

Description:
    Contains project-wide path definitions, satellite band configurations,
    data patch parameters, model hyperparameters, land cover class labels,
    and visualization color mappings for Sentinel-1 & Sentinel-2 processing.
===============================================================================
"""

import os
from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_S1_DIR = DATA_DIR / "raw" / "sentinel1"
RAW_S2_DIR = DATA_DIR / "raw" / "sentinel2"
RAW_LABELS_DIR = DATA_DIR / "raw" / "labels"
PROCESSED_DIR = DATA_DIR / "processed"
PATCHES_DIR = DATA_DIR / "patches"

# Output Directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure runtime directories exist
for path in [
    RAW_S1_DIR, RAW_S2_DIR, RAW_LABELS_DIR, PROCESSED_DIR, PATCHES_DIR,
    MODELS_DIR, MAPS_DIR, GRAPHS_DIR, REPORTS_DIR
]:
    os.makedirs(path, exist_ok=True)

# Satellite Data Band Configurations
SENTINEL1_BANDS = ["VV", "VH"]
SENTINEL2_BANDS = [
    "B02",  # Blue (10m)
    "B03",  # Green (10m)
    "B04",  # Red (10m)
    "B08",  # NIR (10m)
    "B05",  # Vegetation Red Edge 1 (20m)
    "B06",  # Vegetation Red Edge 2 (20m)
    "B07",  # Vegetation Red Edge 3 (20m)
    "B8A",  # Narrow NIR (20m)
    "B11",  # SWIR 1 (20m)
    "B12",  # SWIR 2 (20m)
]

# Tiling & Data Preprocessing Hyperparameters
PATCH_SIZE = 128            # Spatial patch dimension (128x128 pixels)
PATCH_STRIDE = 64           # Overlap stride for patch extraction
TARGET_EPSG = "EPSG:4326"   # Standard WGS84 CRS (or target local UTM projection)
SPATIAL_RESOLUTION = 10.0   # Resampled target resolution (meters)

# Land Cover Classification Taxonomy
CLASSES = {
    0: "Background / Unclassified",
    1: "Water Bodies",
    2: "Built-up / Urban",
    3: "Dense Forest / Vegetation",
    4: "Cropland / Agricultural Land",
    5: "Barren / Bare Soil",
    6: "Wetlands",
}

NUM_CLASSES = len(CLASSES)

# Color Palette for Visualization (Hex codes for Folium / Matplotlib)
CLASS_COLORMAP = {
    0: "#000000",  # Black
    1: "#1f77b4",  # Blue
    2: "#d62728",  # Red
    3: "#2ca02c",  # Green
    4: "#bcbd22",  # Yellow/Olive
    5: "#8c564b",  # Brown
    6: "#9467bd",  # Purple
}

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Device Configuration
DEVICE = "cuda"  # Will fallback to "cpu" dynamically in PyTorch scripts if CUDA is unavailable
