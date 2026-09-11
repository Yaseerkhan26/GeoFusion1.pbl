"""
===============================================================================
File: src/models/create_class_mapping.py
Purpose: Dynamic Class Identification and Class Mapping Generator for FusionLand AI

Description:
    1. Inspects all ground truth label patches in data/patches/labels/*.npy.
    2. Identifies all unique ESA WorldCover class IDs present in the dataset.
    3. Generates a contiguous 0-indexed mapping:
       original_class_id -> model_class_id (0, 1, 2, ..., NUM_CLASSES - 1).
    4. Saves the resulting mapping and legend to data/class_mapping.json.
===============================================================================
"""

import json
import sys
import numpy as np
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import DATA_DIR, PATCHES_DIR

# WorldCover original class descriptions reference
WORLDCOVER_DESCRIPTIONS = {
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


def create_class_mapping():
    """
    Scans label patches to detect present class IDs and creates data/class_mapping.json.

    Returns:
        dict: The loaded/created class mapping dictionary.
    """
    labels_dir = PATCHES_DIR / "labels"
    output_json_path = DATA_DIR / "class_mapping.json"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Label patches directory not found at: {labels_dir}")

    label_files = list(labels_dir.glob("*.npy"))
    if not label_files:
        raise FileNotFoundError(f"No .npy label files found in: {labels_dir}")

    print("=== Scanning Label Patches to Detect Unique Class IDs ===")
    unique_classes_set = set()

    for f in label_files:
        lbl = np.load(f)
        classes_in_patch = np.unique(lbl)
        # Exclude background/unclassified 0 if nodata, or keep valid class IDs > 0
        valid_classes = [int(c) for c in classes_in_patch if c > 0]
        unique_classes_set.update(valid_classes)

    sorted_classes = sorted(list(unique_classes_set))
    num_classes = len(sorted_classes)

    print(f"[INFO] Detected {num_classes} unique ESA WorldCover class IDs: {sorted_classes}")

    original_to_model = {}
    model_to_original = {}
    class_legend = {}

    for idx, orig_id in enumerate(sorted_classes):
        original_to_model[str(orig_id)] = idx
        model_to_original[str(idx)] = orig_id
        class_legend[str(idx)] = WORLDCOVER_DESCRIPTIONS.get(orig_id, f"Class {orig_id}")

    mapping_data = {
        "original_to_model": original_to_model,
        "model_to_original": model_to_original,
        "class_legend": class_legend,
        "num_classes": num_classes,
        "unique_original_classes": sorted_classes,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=4)

    print(f"[SUCCESS] Class mapping saved to: {output_json_path}")
    print("\nClass Mapping Table:")
    print("  Model ID | Original ESA ID | Description")
    print("  ------------------------------------------------")
    for model_id_str, orig_id in model_to_original.items():
        desc = class_legend[model_id_str]
        print(f"  {model_id_str:<8} | {orig_id:<15} | {desc}")

    return mapping_data


if __name__ == "__main__":
    create_class_mapping()
