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
SPLITS_DIR = DATA_DIR / "splits"

# Output Directories
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FINAL_RESULTS_DIR = OUTPUTS_DIR / "final_results"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"
MAPS_DIR = OUTPUTS_DIR / "maps"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure runtime directories exist
for path in [
    RAW_S1_DIR, RAW_S2_DIR, RAW_LABELS_DIR, PROCESSED_DIR, PATCHES_DIR,
    SPLITS_DIR, MODELS_DIR, FINAL_RESULTS_DIR, TRAINING_OUTPUTS_DIR, MAPS_DIR, GRAPHS_DIR, REPORTS_DIR
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
PATCH_SIZE = 256            # Spatial patch dimension (256x256 pixels)
PATCH_STRIDE = 256          # Non-overlapping patch extraction stride
MIN_VALID_RATIO = 0.80      # Minimum fraction of valid pixels required per patch
TARGET_EPSG = "EPSG:4326"   # Standard WGS84 CRS (or target local UTM projection)
SPATIAL_RESOLUTION = 10.0   # Resampled target resolution (meters)
IGNORE_INDEX = 255          # Unclassified / NoData pixel label index

# Land Cover Classification Taxonomy (Validated ESA WorldCover 8-Class Mapping)
CLASSES = {
    0: "Tree cover",
    1: "Shrubland",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare / sparse vegetation",
    6: "Permanent water bodies",
    7: "Herbaceous wetland",
}

NUM_CLASSES = len(CLASSES)

# Original ESA WorldCover numeric IDs
ESA_WORLDCOVER_IDS = {
    0: 10,
    1: 20,
    2: 30,
    3: 40,
    4: 50,
    5: 60,
    6: 80,
    7: 90,
}

# Color Palette for Visualization (Hex codes matching standard ESA WorldCover legend)
CLASS_COLORMAP = {
    0: "#1E5631",  # Tree cover (Green)
    1: "#4C9A2A",  # Shrubland (Light/Dark Green)
    2: "#ACD870",  # Grassland (Yellow-Green)
    3: "#E5B636",  # Cropland (Agricultural Yellow/Orange)
    4: "#808080",  # Built-up (Gray)
    5: "#A0826C",  # Bare / sparse vegetation (Brown)
    6: "#0066CC",  # Permanent water bodies (Blue)
    7: "#00A896",  # Herbaceous wetland (Cyan)
}

# Default Checkpoint & Evaluation Paths
DEFAULT_MODEL_PATH = MODELS_DIR / "GeoFusion_AI_Final.pth"
DEFAULT_EVAL_PATH = FINAL_RESULTS_DIR / "final_metrics.json"

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Device Configuration
DEVICE = "cuda"  # Will fallback to "cpu" dynamically in PyTorch scripts if CUDA is unavailable

