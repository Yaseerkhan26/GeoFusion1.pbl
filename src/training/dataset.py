"""
===============================================================================
File: src/training/dataset.py
Purpose: Training Dataset Module / Compatibility Wrapper for GeoFusion AI Stage 6

Description:
    Re-exports MultimodalSatelliteDataset from src.models.dataset for modular
    access within the training package.
===============================================================================
"""

import sys
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models.dataset import MultimodalSatelliteDataset

# Compatibility alias
Dataset = MultimodalSatelliteDataset

if __name__ == "__main__":
    from src.utils.config import SPLITS_DIR
    train_csv = SPLITS_DIR / "train.csv"
    if train_csv.exists():
        ds = MultimodalSatelliteDataset(train_csv)
        print(f"[DATASET CHECK] Successfully loaded dataset from {train_csv} with {len(ds)} samples.")
