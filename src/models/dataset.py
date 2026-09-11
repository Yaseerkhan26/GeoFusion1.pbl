"""
===============================================================================
File: src/models/dataset.py
Purpose: PyTorch Dataset Loader for Multimodal Satellite Data Fusion

Description:
    1. Loads patch metadata from split CSV files (train.csv, val.csv, test.csv).
    2. Synchronously reads Sentinel-1, Sentinel-2, and ESA WorldCover label .npy patches.
    3. Maps original ESA WorldCover class IDs to contiguous integer model indices (0..NUM_CLASSES-1)
       using class_mapping.json.
    4. Converts inputs to PyTorch Tensors:
       - Sentinel-1: torch.float32, shape [2, 256, 256]
       - Sentinel-2: torch.float32, shape [6, 256, 256]
       - Label:      torch.long,    shape [256, 256]
    5. Performs strict shape and file existence validation.
===============================================================================
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import BASE_DIR, DATA_DIR
from src.models.config import PATCH_SIZE, CLASS_MAPPING_PATH


class MultimodalSatelliteDataset(Dataset):
    """
    PyTorch Dataset for synchronized multimodal satellite patch triplets.
    """

    def __init__(self, split_csv_path, class_mapping_path=CLASS_MAPPING_PATH, root_dir=BASE_DIR):
        """
        Args:
            split_csv_path (str or Path): Path to split CSV file (train.csv, val.csv, or test.csv).
            class_mapping_path (str or Path): Path to class_mapping.json.
            root_dir (str or Path): Root project directory for resolving relative file paths.
        """
        self.split_csv_path = Path(split_csv_path)
        self.root_dir = Path(root_dir)
        self.class_mapping_path = Path(class_mapping_path)

        if not self.split_csv_path.exists():
            raise FileNotFoundError(f"Split CSV file not found at: {self.split_csv_path}")

        if not self.class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping file not found at: {self.class_mapping_path}")

        # Load CSV metadata
        self.df = pd.read_csv(self.split_csv_path)
        required_cols = {"patch_id", "sentinel1_path", "sentinel2_path", "label_path"}
        if not required_cols.issubset(set(self.df.columns)):
            raise ValueError(f"Split CSV missing required columns. Found: {list(self.df.columns)}")

        # Load class mapping
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        orig_to_model = mapping_data["original_to_model"]
        self.num_classes = mapping_data["num_classes"]

        # Create fast lookup array for mapping ESA WorldCover class IDs (0..255) to model IDs (0..num_classes-1)
        self.lut = np.zeros(256, dtype=np.int64)
        for orig_str, model_idx in orig_to_model.items():
            orig_id = int(orig_str)
            if 0 <= orig_id < 256:
                self.lut[orig_id] = int(model_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id = row["patch_id"]

        # Resolve paths
        s1_path = self.root_dir / row["sentinel1_path"]
        s2_path = self.root_dir / row["sentinel2_path"]
        lbl_path = self.root_dir / row["label_path"]

        # Verify file existence
        if not s1_path.exists():
            raise FileNotFoundError(f"Sentinel-1 patch file missing for {patch_id}: {s1_path}")
        if not s2_path.exists():
            raise FileNotFoundError(f"Sentinel-2 patch file missing for {patch_id}: {s2_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Label patch file missing for {patch_id}: {lbl_path}")

        # Load numpy arrays
        s1_arr = np.load(s1_path)
        s2_arr = np.load(s2_path)
        lbl_arr = np.load(lbl_path)

        # Validate patch shapes before conversion
        if s1_arr.shape != (2, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Invalid Sentinel-1 patch shape {s1_arr.shape} for {patch_id}, expected (2, {PATCH_SIZE}, {PATCH_SIZE})")
        if s2_arr.shape != (6, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Invalid Sentinel-2 patch shape {s2_arr.shape} for {patch_id}, expected (6, {PATCH_SIZE}, {PATCH_SIZE})")
        if lbl_arr.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Invalid Label patch shape {lbl_arr.shape} for {patch_id}, expected ({PATCH_SIZE}, {PATCH_SIZE})")

        # Vectorized mapping of class IDs to contiguous 0..num_classes-1
        mapped_label_arr = self.lut[lbl_arr]

        # Convert to PyTorch Tensors
        s1_tensor = torch.from_numpy(s1_arr).to(torch.float32)
        s2_tensor = torch.from_numpy(s2_arr).to(torch.float32)
        label_tensor = torch.from_numpy(mapped_label_arr).to(torch.long)

        return s1_tensor, s2_tensor, label_tensor


if __name__ == "__main__":
    from src.utils.config import SPLITS_DIR
    train_csv = SPLITS_DIR / "train.csv"
    if train_csv.exists():
        dataset = MultimodalSatelliteDataset(train_csv)
        print(f"[TEST] Dataset loaded successfully with {len(dataset)} samples.")
        s1, s2, lbl = dataset[0]
        print(f"  Sample 0 -> S1 shape: {s1.shape}, dtype: {s1.dtype}")
        print(f"  Sample 0 -> S2 shape: {s2.shape}, dtype: {s2.dtype}")
        print(f"  Sample 0 -> Label shape: {lbl.shape}, dtype: {lbl.dtype}, unique labels: {torch.unique(lbl).tolist()}")
