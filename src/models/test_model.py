"""
===============================================================================
File: src/models/test_model.py
Purpose: Test Runner for MultimodalFusionNet Architecture and Real Data Verification

Description:
    1. Loads data/class_mapping.json and retrieves NUM_CLASSES.
    2. Instantiates MultimodalFusionNet model.
    3. Runs forward pass on synthetic dummy tensors (batch size 2).
    4. Checks output tensor shape [2, NUM_CLASSES, 256, 256] and verifies no NaN/Inf values.
    5. Calculates total trainable parameter count and prints architecture summary.
    6. Loads real patch sample from data/splits/train.csv via MultimodalSatelliteDataset.
    7. Runs forward pass on real patch tensors and verifies output shape alignment with label map.
===============================================================================
"""

import json
import sys
import torch
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import SPLITS_DIR, DATA_DIR
from src.models.config import NUM_CLASSES, CLASS_MAPPING_PATH
from src.models.fusion_model import MultimodalFusionNet
from src.models.dataset import MultimodalSatelliteDataset


def test_model_architecture():
    """
    Executes dummy input test and real data sample test on MultimodalFusionNet.
    """
    print("=== Starting MultimodalFusionNet Architecture Test ===")

    # 1. Check class_mapping.json and NUM_CLASSES
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"class_mapping.json not found at: {CLASS_MAPPING_PATH}")

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    num_classes = mapping_data.get("num_classes", NUM_CLASSES)
    print(f"[INFO] Detected Number of Classes: {num_classes}")

    # 2. Instantiate Model
    model = MultimodalFusionNet(num_classes=num_classes)
    model.eval()

    # 3. Create Dummy Inputs
    batch_size = 2
    dummy_s1 = torch.randn(batch_size, 2, 256, 256, dtype=torch.float32)
    dummy_s2 = torch.randn(batch_size, 6, 256, 256, dtype=torch.float32)

    # 4. Dummy Forward Pass
    with torch.no_grad():
        dummy_out = model(dummy_s1, dummy_s2)

    expected_dummy_shape = (batch_size, num_classes, 256, 256)
    print(f"[INFO] Dummy Input S1 Shape: {dummy_s1.shape}")
    print(f"[INFO] Dummy Input S2 Shape: {dummy_s2.shape}")
    print(f"[INFO] Dummy Output Shape:   {dummy_out.shape}")

    if dummy_out.shape != expected_dummy_shape:
        raise ValueError(f"Dummy output shape mismatch! Expected {expected_dummy_shape}, got {dummy_out.shape}")

    # 5. Check NaNs and Infs
    if torch.isnan(dummy_out).any():
        raise ValueError("NaN values detected in dummy output tensor!")
    if torch.isinf(dummy_out).any():
        raise ValueError("Inf values detected in dummy output tensor!")

    print("[SUCCESS] Dummy Forward Pass Passed (No NaNs, No Infs).")

    # 6. Parameter Count & Summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== MODEL PARAMETER SUMMARY ===")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Architecture:\n{model}\n")

    # 7. Real Data Test
    train_csv = SPLITS_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Training split CSV not found at: {train_csv}")

    dataset = MultimodalSatelliteDataset(split_csv_path=train_csv)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty!")

    real_s1, real_s2, real_lbl = dataset[0]

    # Add batch dimension [1, C, H, W]
    real_s1_b = real_s1.unsqueeze(0)
    real_s2_b = real_s2.unsqueeze(0)
    real_lbl_b = real_lbl.unsqueeze(0)

    with torch.no_grad():
        real_out = model(real_s1_b, real_s2_b)

    expected_real_shape = (1, num_classes, 256, 256)
    print("=== REAL DATA FORWARD PASS ===")
    print(f"  Real S1 Patch Shape:    {real_s1_b.shape}")
    print(f"  Real S2 Patch Shape:    {real_s2_b.shape}")
    print(f"  Real Label Map Shape:   {real_lbl_b.shape}")
    print(f"  Model Output Logits:    {real_out.shape}")

    if real_out.shape != expected_real_shape:
        raise ValueError(f"Real data output shape mismatch! Expected {expected_real_shape}, got {real_out.shape}")

    if real_out.shape[2:] != real_lbl_b.shape[1:]:
        raise ValueError(f"Spatial dimension mismatch between output {real_out.shape[2:]} and label {real_lbl_b.shape[1:]}!")

    if torch.isnan(real_out).any() or torch.isinf(real_out).any():
        raise ValueError("NaN or Inf values detected in real output tensor!")

    print("[SUCCESS] Real Data Forward Pass Passed Successfully!")

    return {
        "num_classes": num_classes,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dummy_out_shape": list(dummy_out.shape),
        "real_out_shape": list(real_out.shape),
    }


if __name__ == "__main__":
    test_model_architecture()
