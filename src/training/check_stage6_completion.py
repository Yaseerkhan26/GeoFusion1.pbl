"""
===============================================================================
File: src/training/check_stage6_completion.py
Purpose: Final Completion Verification Suite for GeoFusion AI Stage 6

Description:
    Performs 13 strict verification checks to validate Stage 6 artifacts:
    - Checks existence and loadability of Sentinel-1, Sentinel-2, and Fusion checkpoints.
    - Checks existence of JSON training histories, loss/accuracy data, training graphs, and reports.
    - Tests forward passes for all 3 models and verifies numeric sanity (no NaN/Inf).

    Rules:
    - Does not modify any project data or trained models.
    - Prints exact required completion status banners.
===============================================================================
"""

import json
import math
import sys
import torch
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import MODELS_DIR, REPORTS_DIR, GRAPHS_DIR
from src.models.config import NUM_CLASSES, SENTINEL1_CHANNELS, SENTINEL2_CHANNELS, PATCH_SIZE
from src.models.sentinel1_model import Sentinel1OnlyNet
from src.models.sentinel2_model import Sentinel2OnlyNet
from src.models.fusion_model import MultimodalFusionNet


def run_stage6_completion_check():
    """
    Executes the 13 mandatory Stage 6 verification checks.
    """
    print("=== Starting Stage 6 Final Completion Verification ===\n")

    missing_items = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths setup
    s1_ckpt_path = MODELS_DIR / "sentinel1_best.pth"
    s2_ckpt_path = MODELS_DIR / "sentinel2_best.pth"
    fusion_ckpt_path = MODELS_DIR / "fusion_best.pth"

    full_history_path = REPORTS_DIR / "training_history.json"
    s1_history_path = REPORTS_DIR / "sentinel1_history.json"
    s2_history_path = REPORTS_DIR / "sentinel2_history.json"
    fusion_history_path = REPORTS_DIR / "fusion_history.json"

    report_txt_path = REPORTS_DIR / "training_report.txt"

    # CHECK 1: Sentinel-1 best model exists
    print("[CHECK 1/13] Verifying Sentinel-1 best model exists...")
    if not s1_ckpt_path.exists():
        missing_items.append(f"Missing Sentinel-1 model checkpoint at: {s1_ckpt_path}")
        print("  [FAIL] Sentinel-1 model checkpoint missing.")
    else:
        print(f"  [OK] Found {s1_ckpt_path.name}")

    # CHECK 2: Sentinel-2 best model exists
    print("[CHECK 2/13] Verifying Sentinel-2 best model exists...")
    if not s2_ckpt_path.exists():
        missing_items.append(f"Missing Sentinel-2 model checkpoint at: {s2_ckpt_path}")
        print("  [FAIL] Sentinel-2 model checkpoint missing.")
    else:
        print(f"  [OK] Found {s2_ckpt_path.name}")

    # CHECK 3: Fusion best model exists
    print("[CHECK 3/13] Verifying Fusion best model exists...")
    if not fusion_ckpt_path.exists():
        missing_items.append(f"Missing Fusion model checkpoint at: {fusion_ckpt_path}")
        print("  [FAIL] Fusion model checkpoint missing.")
    else:
        print(f"  [OK] Found {fusion_ckpt_path.name}")

    # CHECK 4: Each model checkpoint can be loaded successfully
    print("[CHECK 4/13] Verifying each model checkpoint can be loaded successfully...")
    s1_model = Sentinel1OnlyNet(num_classes=NUM_CLASSES).to(device)
    s2_model = Sentinel2OnlyNet(num_classes=NUM_CLASSES).to(device)
    fusion_model = MultimodalFusionNet(num_classes=NUM_CLASSES).to(device)

    try:
        if s1_ckpt_path.exists():
            s1_model.load_state_dict(torch.load(s1_ckpt_path, map_location=device))
        if s2_ckpt_path.exists():
            s2_model.load_state_dict(torch.load(s2_ckpt_path, map_location=device))
        if fusion_ckpt_path.exists():
            fusion_model.load_state_dict(torch.load(fusion_ckpt_path, map_location=device))
        print("  [OK] All available model checkpoints loaded into state_dict successfully.")
    except Exception as e:
        missing_items.append(f"Failed to load model checkpoints: {str(e)}")
        print(f"  [FAIL] Exception during model checkpoint loading: {e}")

    # Load history data for checks 5-9
    history_data = {}
    if full_history_path.exists():
        with open(full_history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)

    # CHECK 5: Sentinel-1 training history exists
    print("[CHECK 5/13] Verifying Sentinel-1 training history exists...")
    has_s1_hist = "sentinel1" in history_data or s1_history_path.exists()
    if not has_s1_hist:
        missing_items.append("Sentinel-1 training history missing")
        print("  [FAIL] Sentinel-1 training history missing.")
    else:
        print("  [OK] Sentinel-1 training history present.")

    # CHECK 6: Sentinel-2 training history exists
    print("[CHECK 6/13] Verifying Sentinel-2 training history exists...")
    has_s2_hist = "sentinel2" in history_data or s2_history_path.exists()
    if not has_s2_hist:
        missing_items.append("Sentinel-2 training history missing")
        print("  [FAIL] Sentinel-2 training history missing.")
    else:
        print("  [OK] Sentinel-2 training history present.")

    # CHECK 7: Fusion training history exists
    print("[CHECK 7/13] Verifying Fusion training history exists...")
    has_fusion_hist = "fusion" in history_data or fusion_history_path.exists()
    if not has_fusion_hist:
        missing_items.append("Fusion training history missing")
        print("  [FAIL] Fusion training history missing.")
    else:
        print("  [OK] Fusion training history present.")

    # CHECK 8: Training loss data exists
    print("[CHECK 8/13] Verifying training loss data exists...")
    tr_loss_found = False
    for mkey in ["sentinel1", "sentinel2", "fusion"]:
        hist = history_data.get(mkey, {})
        if "train_loss" in hist and len(hist["train_loss"]) > 0:
            tr_loss_found = True
            break

    if not tr_loss_found and s1_history_path.exists():
        with open(s1_history_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            if "train_loss" in d and len(d["train_loss"]) > 0:
                tr_loss_found = True

    if not tr_loss_found:
        missing_items.append("Training loss data missing in history files")
        print("  [FAIL] Training loss data missing.")
    else:
        print("  [OK] Training loss data verified.")

    # CHECK 9: Validation loss data exists
    print("[CHECK 9/13] Verifying validation loss data exists...")
    val_loss_found = False
    for mkey in ["sentinel1", "sentinel2", "fusion"]:
        hist = history_data.get(mkey, {})
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            val_loss_found = True
            break

    if not val_loss_found and s1_history_path.exists():
        with open(s1_history_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            if "val_loss" in d and len(d["val_loss"]) > 0:
                val_loss_found = True

    if not val_loss_found:
        missing_items.append("Validation loss data missing in history files")
        print("  [FAIL] Validation loss data missing.")
    else:
        print("  [OK] Validation loss data verified.")

    # CHECK 10: Training graphs exist
    print("[CHECK 10/13] Verifying training graphs exist...")
    graph_files = list(GRAPHS_DIR.glob("*.png"))
    if len(graph_files) == 0:
        missing_items.append("No training graph PNG images found in outputs/graphs/")
        print("  [FAIL] Training graphs missing.")
    else:
        print(f"  [OK] Found {len(graph_files)} training graph file(s) in {GRAPHS_DIR.name}.")

    # CHECK 11: The three models can perform a forward pass
    print("[CHECK 11/13] Verifying all three models can perform a forward pass...")
    s1_dummy = torch.randn(2, SENTINEL1_CHANNELS, PATCH_SIZE, PATCH_SIZE, device=device)
    s2_dummy = torch.randn(2, SENTINEL2_CHANNELS, PATCH_SIZE, PATCH_SIZE, device=device)

    s1_model.eval()
    s2_model.eval()
    fusion_model.eval()

    try:
        with torch.no_grad():
            out_s1 = s1_model(s1_dummy)
            out_s2 = s2_model(s2_dummy)
            out_fusion = fusion_model(s1_dummy, s2_dummy)

        expected_shape = (2, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
        if out_s1.shape != expected_shape or out_s2.shape != expected_shape or out_fusion.shape != expected_shape:
            missing_items.append(f"Model forward pass shape mismatch. Expected {expected_shape}")
            print("  [FAIL] Forward pass shape mismatch.")
        else:
            print("  [OK] All 3 models performed forward passes successfully.")
    except Exception as e:
        missing_items.append(f"Forward pass exception: {str(e)}")
        print(f"  [FAIL] Forward pass exception: {e}")

    # CHECK 12: No NaN or Inf values occur
    print("[CHECK 12/13] Verifying no NaN or Inf values occur in outputs or history...")
    has_nan_inf = False
    try:
        if torch.isnan(out_s1).any() or torch.isinf(out_s1).any():
            has_nan_inf = True
        if torch.isnan(out_s2).any() or torch.isinf(out_s2).any():
            has_nan_inf = True
        if torch.isnan(out_fusion).any() or torch.isinf(out_fusion).any():
            has_nan_inf = True

        for mkey, hist in history_data.items():
            for lkey in ["train_loss", "val_loss", "train_acc", "val_acc"]:
                for val in hist.get(lkey, []):
                    if math.isnan(val) or math.isinf(val):
                        has_nan_inf = True
    except Exception as e:
        has_nan_inf = True

    if has_nan_inf:
        missing_items.append("NaN or Inf values detected in forward pass outputs or training history data")
        print("  [FAIL] NaN or Inf values detected.")
    else:
        print("  [OK] All model outputs and history values are valid and finite.")

    # CHECK 13: A training report exists
    print("[CHECK 13/13] Verifying a training report exists...")
    if not report_txt_path.exists() or report_txt_path.stat().st_size == 0:
        missing_items.append(f"Missing or empty training report text file at: {report_txt_path}")
        print("  [FAIL] Training report file missing or empty.")
    else:
        print(f"  [OK] Found training report file: {report_txt_path.name}")

    # Final Summary Banner
    print("\n" + "=" * 52)
    print("STAGE 6 FINAL COMPLETION CHECK")
    print("=" * 52)

    if len(missing_items) == 0:
        print("\n[SUCCESS] ALL STAGE 6 CHECKS PASSED\n")
        print("[SUCCESS] STAGE 6 COMPLETED SUCCESSFULLY\n")
        print("=" * 52 + "\n")
        return True
    else:
        print("\n[FAIL] STAGE 6 NOT COMPLETE\n")
        print("Missing required items:")
        for item in missing_items:
            print(f" - {item}")
        print("\n" + "=" * 52 + "\n")
        return False


if __name__ == "__main__":
    success = run_stage6_completion_check()
    if not success:
        sys.exit(1)
