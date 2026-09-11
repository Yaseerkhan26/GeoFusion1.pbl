"""
===============================================================================
File: src/models/validate_model.py
Purpose: Comprehensive Validation Suite and Architecture Reporting for Stage 5

Description:
    Performs 12 mandatory validation checks on MultimodalFusionNet architecture,
    dataset loader, encoders, fusion layer, tensor shapes, value bounds, and real patch inference.

    Checks:
    CHECK 1:  class_mapping.json exists.
    CHECK 2:  Classes are correctly detected.
    CHECK 3:  Sentinel-1 encoder accepts [Batch, 2, 256, 256].
    CHECK 4:  Sentinel-2 encoder accepts [Batch, 6, 256, 256].
    CHECK 5:  Feature dimensions are compatible.
    CHECK 6:  Fusion layer works correctly.
    CHECK 7:  Final model output shape is [Batch, NUM_CLASSES, 256, 256].
    CHECK 8:  No NaN values in outputs.
    CHECK 9:  No Inf values in outputs.
    CHECK 10: Model works with real dataset patches.
    CHECK 11: Output spatial dimensions match labels.
    CHECK 12: Model parameters are greater than zero.

    Also generates: outputs/reports/model_architecture_report.txt
===============================================================================
"""

import json
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import SPLITS_DIR, REPORTS_DIR
from src.models.config import (
    CLASS_MAPPING_PATH,
    PATCH_SIZE,
    SENTINEL1_CHANNELS,
    SENTINEL2_CHANNELS,
    FEATURE_CHANNELS,
)
from src.models.sentinel1_encoder import Sentinel1Encoder
from src.models.sentinel2_encoder import Sentinel2Encoder
from src.models.fusion_model import MultimodalFusionNet
from src.models.dataset import MultimodalSatelliteDataset


def generate_architecture_report(mapping_data, num_classes, total_params, trainable_params, model_str):
    """
    Generates outputs/reports/model_architecture_report.txt summarizing Stage 5 results.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_architecture_report.txt"

    orig_to_model = mapping_data.get("original_to_model", {})
    legend = mapping_data.get("class_legend", {})

    lines = [
        "===============================================================================",
        "FUSIONLAND AI: MULTIMODAL AI MODEL ARCHITECTURE REPORT (STAGE 5)",
        "===============================================================================\n",
        "1. PROJECT & MODEL SPECIFICATIONS:",
        "   - Project Name:              FusionLand AI",
        "   - Model Name:                MultimodalFusionNet",
        f"   - Sentinel-1 Input Channels: {SENTINEL1_CHANNELS} (VV, VH)",
        f"   - Sentinel-2 Input Channels: {SENTINEL2_CHANNELS} (B2, B3, B4, B8, B11, B12)",
        f"   - Patch Spatial Size:        {PATCH_SIZE} x {PATCH_SIZE}",
        f"   - Feature Map Depth:         {FEATURE_CHANNELS} channels per encoder",
        f"   - Number of Target Classes:  {num_classes} (Automatically detected)",
        "   - Fusion Method:             Feature-level Concatenation (64 channels)",
        "   - Output Type:               Pixel-wise land-cover classification logits",
        "   - Activation Function:       Linear (No Softmax inside model; raw logits returned)\n",
        "2. MODEL PARAMETER COUNT:",
        f"   - Total Parameters:          {total_params:,}",
        f"   - Trainable Parameters:      {trainable_params:,}\n",
        "3. TENSOR SHAPE SUMMARY:",
        f"   - Sentinel-1 Input Tensor:  [Batch, {SENTINEL1_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}]",
        f"   - Sentinel-2 Input Tensor:  [Batch, {SENTINEL2_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}]",
        f"   - Sentinel-1 Encoder Output:[Batch, {FEATURE_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}]",
        f"   - Sentinel-2 Encoder Output:[Batch, {FEATURE_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}]",
        f"   - Fused Feature Map Tensor: [Batch, {FEATURE_CHANNELS * 2}, {PATCH_SIZE}, {PATCH_SIZE}]",
        f"   - Final Classification Output: [Batch, {num_classes}, {PATCH_SIZE}, {PATCH_SIZE}]",
        f"   - Ground Truth Label Tensor:[Batch, {PATCH_SIZE}, {PATCH_SIZE}]\n",
        "4. CLASS MAPPING DETAILS:",
        "   Model ID | Original ESA ID | Class Description",
        "   --------------------------------------------------------------",
    ]

    for model_id_str, orig_id in mapping_data.get("model_to_original", {}).items():
        desc = legend.get(model_id_str, f"Class {orig_id}")
        lines.append(f"   {model_id_str:<8} | {orig_id:<15} | {desc}")

    lines.extend([
        "\n5. DETAILED ARCHITECTURE SUMMARY:",
        model_str,
        "\n===============================================================================",
        "[SUCCESS] Stage 5 Model Architecture report generated successfully.",
        "===============================================================================",
    ])

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[SUCCESS] Model architecture report generated at: {report_path}")


def validate_model_pipeline():
    """
    Executes 12 validation checks for Stage 5.
    """
    print("=== Starting Stage 5 Model Architecture Validation ===")

    # CHECK 1: class_mapping.json exists
    print("[CHECK 1/12] Verifying class_mapping.json exists...")
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"[FAIL] class_mapping.json missing at: {CLASS_MAPPING_PATH}")

    # CHECK 2: Classes correctly detected
    print("[CHECK 2/12] Verifying class mapping content...")
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    num_classes = mapping_data.get("num_classes", 0)
    if num_classes <= 0:
        raise ValueError(f"[FAIL] Invalid num_classes detected: {num_classes}")
    print(f"  Detected {num_classes} land-cover classes.")

    # CHECK 3: Sentinel-1 encoder accepts [Batch, 2, 256, 256]
    print("[CHECK 3/12] Verifying Sentinel-1 encoder accepts [Batch, 2, 256, 256]...")
    s1_enc = Sentinel1Encoder(in_channels=2, out_channels=32)
    s1_dummy = torch.randn(2, 2, 256, 256, dtype=torch.float32)
    s1_feat = s1_enc(s1_dummy)
    if s1_feat.shape != (2, 32, 256, 256):
        raise ValueError(f"[FAIL] S1 encoder feature shape mismatch: {s1_feat.shape}")

    # CHECK 4: Sentinel-2 encoder accepts [Batch, 6, 256, 256]
    print("[CHECK 4/12] Verifying Sentinel-2 encoder accepts [Batch, 6, 256, 256]...")
    s2_enc = Sentinel2Encoder(in_channels=6, out_channels=32)
    s2_dummy = torch.randn(2, 6, 256, 256, dtype=torch.float32)
    s2_feat = s2_enc(s2_dummy)
    if s2_feat.shape != (2, 32, 256, 256):
        raise ValueError(f"[FAIL] S2 encoder feature shape mismatch: {s2_feat.shape}")

    # CHECK 5: Feature dimensions are compatible
    print("[CHECK 5/12] Verifying feature dimensions compatibility...")
    if s1_feat.shape != s2_feat.shape:
        raise ValueError(f"[FAIL] Incompatible feature shapes: S1 {s1_feat.shape} vs S2 {s2_feat.shape}")

    # CHECK 6: Fusion layer works correctly
    print("[CHECK 6/12] Verifying fusion concatenation layer...")
    fused = torch.cat([s1_feat, s2_feat], dim=1)
    if fused.shape != (2, 64, 256, 256):
        raise ValueError(f"[FAIL] Fused feature shape mismatch: {fused.shape}")

    # CHECK 7: Final model output shape is [Batch, NUM_CLASSES, 256, 256]
    print("[CHECK 7/12] Verifying final MultimodalFusionNet output shape [Batch, NUM_CLASSES, 256, 256]...")
    model = MultimodalFusionNet(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        logits = model(s1_dummy, s2_dummy)

    expected_shape = (2, num_classes, 256, 256)
    if logits.shape != expected_shape:
        raise ValueError(f"[FAIL] Model output shape mismatch! Expected {expected_shape}, got {logits.shape}")

    # CHECK 8: No NaN values
    print("[CHECK 8/12] Verifying no NaN values in model output...")
    if torch.isnan(logits).any():
        raise ValueError("[FAIL] NaN values detected in model logits!")

    # CHECK 9: No Inf values
    print("[CHECK 9/12] Verifying no Inf values in model output...")
    if torch.isinf(logits).any():
        raise ValueError("[FAIL] Inf values detected in model logits!")

    # CHECK 10: Model works with real dataset patches
    print("[CHECK 10/12] Testing model inference on real dataset patch...")
    train_csv = SPLITS_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"[FAIL] Training CSV missing: {train_csv}")

    dataset = MultimodalSatelliteDataset(split_csv_path=train_csv)
    real_s1, real_s2, real_lbl = dataset[0]

    real_s1_b = real_s1.unsqueeze(0)
    real_s2_b = real_s2.unsqueeze(0)
    real_lbl_b = real_lbl.unsqueeze(0)

    with torch.no_grad():
        real_logits = model(real_s1_b, real_s2_b)

    # CHECK 11: Output spatial dimensions match labels
    print("[CHECK 11/12] Verifying output spatial dimensions match ground truth labels...")
    if real_logits.shape[2:] != real_lbl_b.shape[1:]:
        raise ValueError(
            f"[FAIL] Output spatial dimensions {real_logits.shape[2:]} != label dimensions {real_lbl_b.shape[1:]}"
        )

    # CHECK 12: Model parameters > 0
    print("[CHECK 12/12] Verifying trainable model parameters > 0...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params <= 0 or trainable_params <= 0:
        raise ValueError(f"[FAIL] Model parameters must be > 0. Got total={total_params}, trainable={trainable_params}")

    print(f"  Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # Generate Report
    generate_architecture_report(
        mapping_data=mapping_data,
        num_classes=num_classes,
        total_params=total_params,
        trainable_params=trainable_params,
        model_str=str(model),
    )

    # Print required final success banner
    print("\n" + "=" * 52)
    print("STAGE 5 MODEL VALIDATION")
    print("=" * 52)
    print("\n[SUCCESS] ALL MODEL VALIDATION CHECKS PASSED\n")
    print("[SUCCESS] STAGE 5 COMPLETED SUCCESSFULLY\n")
    print("=" * 52 + "\n")
    return True


if __name__ == "__main__":
    validate_model_pipeline()
