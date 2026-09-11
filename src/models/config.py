"""
===============================================================================
File: src/models/config.py
Purpose: Model Hyperparameters and Architecture Configuration for FusionLand AI

Description:
    Contains model-specific hyperparameter constants including input channels,
    patch size, feature dimensions, random seed, and dynamic NUM_CLASSES loading
    from data/class_mapping.json.
===============================================================================
"""

import json
import sys
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import DATA_DIR, BASE_DIR

# Model Architecture Configuration Parameters
SENTINEL1_CHANNELS = 2
SENTINEL2_CHANNELS = 6
PATCH_SIZE = 256
FEATURE_CHANNELS = 32
RANDOM_SEED = 42

CLASS_MAPPING_PATH = DATA_DIR / "class_mapping.json"


def load_num_classes():
    """
    Dynamically loads NUM_CLASSES from data/class_mapping.json.
    """
    if not CLASS_MAPPING_PATH.exists():
        # Fallback if class_mapping.json has not been generated yet
        return 8

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("num_classes", 8)


NUM_CLASSES = load_num_classes()
