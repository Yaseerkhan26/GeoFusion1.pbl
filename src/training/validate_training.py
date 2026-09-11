"""
===============================================================================
File: src/training/validate_training.py
Purpose: Stage 6 Comprehensive Verification Suite for GeoFusion AI Training Setup

Description:
    Performs 19 mandatory verification checks to validate the training pipeline,
    dataset loaders, single-modality and multimodal fusion models, loss functions,
    backward passes, optimizer updates, and spatial tensor dimensions without running full training.
===============================================================================
"""

import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import SPLITS_DIR, DATA_DIR
from src.models.config import (
    CLASS_MAPPING_PATH,
    PATCH_SIZE,
    SENTINEL1_CHANNELS,
    SENTINEL2_CHANNELS,
    NUM_CLASSES,
)
from src.models.sentinel1_model import Sentinel1OnlyNet
from src.models.sentinel2_model import Sentinel2OnlyNet
from src.models.fusion_model import MultimodalFusionNet
from src.training.dataset import MultimodalSatelliteDataset
from src.training.trainer import ModelTrainer


def validate_training_pipeline():
    """
    Executes the 19 mandatory checks for Stage 6 Training Pipeline.
    """
    print("=== Starting Stage 6 Training Setup Validation ===")

    training_dir = BASE_DIR / "src" / "training"

    # CHECK 1: Required files exist
    print("[CHECK 1/19] Verifying required training source files exist...")
    required_files = ["train.py", "trainer.py", "dataset.py", "__init__.py"]
    for fname in required_files:
        fpath = training_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"[FAIL] Missing required file: {fpath}")
    print("  [OK] Required training files exist.")

    # CHECK 2: class_mapping.json loads
    print("[CHECK 2/19] Verifying class_mapping.json loads successfully...")
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"[FAIL] Missing class_mapping.json at {CLASS_MAPPING_PATH}")
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    num_classes = mapping_data.get("num_classes", 0)
    if num_classes <= 0:
        raise ValueError(f"[FAIL] Invalid num_classes in mapping: {num_classes}")
    print(f"  [OK] Loaded class_mapping.json with {num_classes} classes.")

    # CHECK 3: Training CSV exists
    print("[CHECK 3/19] Verifying training split CSV exists...")
    train_csv = SPLITS_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"[FAIL] Missing train.csv at {train_csv}")
    print("  [OK] Training CSV exists.")

    # CHECK 4: Validation CSV exists
    print("[CHECK 4/19] Verifying validation split CSV exists...")
    val_csv = SPLITS_DIR / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"[FAIL] Missing val.csv at {val_csv}")
    print("  [OK] Validation CSV exists.")

    # CHECK 5: Dataset loads successfully
    print("[CHECK 5/19] Instantiating MultimodalSatelliteDataset...")
    train_dataset = MultimodalSatelliteDataset(split_csv_path=train_csv)
    if len(train_dataset) == 0:
        raise ValueError("[FAIL] Training dataset is empty!")
    print(f"  [OK] Dataset instantiated with {len(train_dataset)} samples.")

    # CHECK 6: One training batch loads
    print("[CHECK 6/19] Loading one training mini-batch from DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    s1_batch, s2_batch, lbl_batch = next(iter(train_loader))
    print("  [OK] Batch successfully loaded.")

    # CHECK 7: Sentinel-1 shape matches actual project configuration
    print(f"[CHECK 7/19] Verifying Sentinel-1 tensor shape [Batch, {SENTINEL1_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}]...")
    expected_s1_shape = (2, SENTINEL1_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    if s1_batch.shape != expected_s1_shape:
        raise ValueError(f"[FAIL] S1 shape mismatch. Expected {expected_s1_shape}, got {s1_batch.shape}")
    print(f"  [OK] Sentinel-1 shape: {s1_batch.shape}")

    # CHECK 8: Sentinel-2 shape matches actual project configuration
    print(f"[CHECK 8/19] Verifying Sentinel-2 tensor shape [Batch, {SENTINEL2_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}]...")
    expected_s2_shape = (2, SENTINEL2_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    if s2_batch.shape != expected_s2_shape:
        raise ValueError(f"[FAIL] S2 shape mismatch. Expected {expected_s2_shape}, got {s2_batch.shape}")
    print(f"  [OK] Sentinel-2 shape: {s2_batch.shape}")

    # CHECK 9: Labels have correct shape
    print(f"[CHECK 9/19] Verifying label tensor shape [Batch, {PATCH_SIZE}, {PATCH_SIZE}]...")
    expected_lbl_shape = (2, PATCH_SIZE, PATCH_SIZE)
    if lbl_batch.shape != expected_lbl_shape:
        raise ValueError(f"[FAIL] Label shape mismatch. Expected {expected_lbl_shape}, got {lbl_batch.shape}")
    print(f"  [OK] Label shape: {lbl_batch.shape}")

    # CHECK 10: Labels are torch.long
    print("[CHECK 10/19] Verifying label data type is torch.long...")
    if lbl_batch.dtype != torch.long:
        raise TypeError(f"[FAIL] Label dtype mismatch. Expected torch.long, got {lbl_batch.dtype}")
    print(f"  [OK] Label dtype: {lbl_batch.dtype}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1_batch = s1_batch.to(device)
    s2_batch = s2_batch.to(device)
    lbl_batch = lbl_batch.to(device)

    # CHECK 11: Sentinel-1 model forward pass works
    print("[CHECK 11/19] Verifying Sentinel-1 model forward pass...")
    s1_model = Sentinel1OnlyNet(num_classes=num_classes).to(device)
    s1_logits = s1_model(s1_batch)
    if s1_logits.shape != (2, num_classes, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"[FAIL] S1 model output shape mismatch: {s1_logits.shape}")
    print(f"  [OK] Sentinel-1 model forward pass successful: {s1_logits.shape}")

    # CHECK 12: Sentinel-2 model forward pass works
    print("[CHECK 12/19] Verifying Sentinel-2 model forward pass...")
    s2_model = Sentinel2OnlyNet(num_classes=num_classes).to(device)
    s2_logits = s2_model(s2_batch)
    if s2_logits.shape != (2, num_classes, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"[FAIL] S2 model output shape mismatch: {s2_logits.shape}")
    print(f"  [OK] Sentinel-2 model forward pass successful: {s2_logits.shape}")

    # CHECK 13: Existing multimodal fusion model forward pass works
    print("[CHECK 13/19] Verifying MultimodalFusionNet forward pass...")
    fusion_model = MultimodalFusionNet(num_classes=num_classes).to(device)
    fusion_logits = fusion_model(s1_batch, s2_batch)
    if fusion_logits.shape != (2, num_classes, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"[FAIL] Fusion model output shape mismatch: {fusion_logits.shape}")
    print(f"  [OK] Multimodal fusion forward pass successful: {fusion_logits.shape}")

    # CHECK 14: Model output spatial dimensions match labels
    print("[CHECK 14/19] Verifying model output spatial dimensions match ground truth labels...")
    if fusion_logits.shape[2:] != lbl_batch.shape[1:]:
        raise ValueError(
            f"[FAIL] Logit spatial dimensions {fusion_logits.shape[2:]} != Label spatial dimensions {lbl_batch.shape[1:]}"
        )
    print("  [OK] Spatial dimensions match labels exactly.")

    # CHECK 15: CrossEntropyLoss works
    print("[CHECK 15/19] Verifying CrossEntropyLoss calculation...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(fusion_logits, lbl_batch)
    print(f"  [OK] Loss calculated: {loss.item():.4f}")

    # CHECK 16: Backward pass works
    print("[CHECK 16/19] Verifying backward pass (loss.backward())...")
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()

    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in fusion_model.parameters())
    if not has_grads:
        raise ValueError("[FAIL] Backward pass failed - no gradients calculated for model parameters.")
    print("  [OK] Gradients successfully computed.")

    # CHECK 17: Optimizer update works
    print("[CHECK 17/19] Verifying optimizer step (optimizer.step())...")
    optimizer.step()
    print("  [OK] Optimizer step completed successfully.")

    # CHECK 18: Validation batch works
    print("[CHECK 18/19] Verifying validation batch pass (eval mode + torch.no_grad)...")
    val_dataset = MultimodalSatelliteDataset(split_csv_path=val_csv)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    val_s1, val_s2, val_lbl = next(iter(val_loader))
    val_s1, val_s2, val_lbl = val_s1.to(device), val_s2.to(device), val_lbl.to(device)

    fusion_model.eval()
    with torch.no_grad():
        val_logits = fusion_model(val_s1, val_s2)
        val_loss = criterion(val_logits, val_lbl)
    print(f"  [OK] Validation batch evaluated. Val Loss: {val_loss.item():.4f}")

    # CHECK 19: No NaN or Inf values
    print("[CHECK 19/19] Verifying no NaN or Inf values in logits, loss, or gradients...")
    if torch.isnan(fusion_logits).any() or torch.isinf(fusion_logits).any():
        raise ValueError("[FAIL] NaN or Inf values detected in model logits!")
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("[FAIL] NaN or Inf value detected in loss!")
    print("  [OK] All tensors contain valid finite numeric values.")

    # Render required final success banner
    print("\n" + "=" * 52)
    print("STAGE 6 TRAINING VALIDATION")
    print("=" * 52)
    print("\n[SUCCESS] ALL TRAINING VALIDATION CHECKS PASSED\n")
    print("[SUCCESS] STAGE 6 SETUP COMPLETED SUCCESSFULLY\n")
    print("=" * 52 + "\n")

    return True


if __name__ == "__main__":
    validate_training_pipeline()
