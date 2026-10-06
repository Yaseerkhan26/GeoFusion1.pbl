"""
===============================================================================
File: src/evaluation/evaluate.py
Purpose: Scientific Evaluation Pipeline with Provenance Tracking
===============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models.config import NUM_CLASSES, IGNORE_INDEX
from src.models.dataset import MultimodalSatelliteDataset
from src.models.fusion_model import MultimodalFusionNet
from src.utils.provenance import (
    compute_dataset_fingerprint,
    compute_file_sha256,
    get_checkpoint_metadata,
    save_evaluation_record,
)


def calculate_per_class_iou(cm: np.ndarray) -> np.ndarray:
    """Calculate IoU for each class from confusion matrix."""
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )
    return iou


def run_evaluation(
    checkpoint_path: Path,
    test_csv: Optional[Path] = None,
    device_name: Optional[str] = None,
    batch_size: int = 4,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates a specific model checkpoint on the test dataset.
    Returns metrics dictionary with provenance metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if test_csv is None:
        test_csv = BASE_DIR / "data" / "splits" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    class_mapping_path = BASE_DIR / "data" / "class_mapping.json"
    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {class_mapping_path}")

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    class_legend = mapping_data["class_legend"]
    model_to_original = mapping_data["model_to_original"]
    class_names = [class_legend[str(i)] for i in range(NUM_CLASSES)]

    # Determine device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[EVALUATION] Starting test evaluation...")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Device:     {device}")
    print(f"  Test split: {test_csv}")

    # Initialize model
    model = MultimodalFusionNet(num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = MultimodalSatelliteDataset(test_csv, class_mapping_path, BASE_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 for safe multi-platform execution
        drop_last=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for s1_x, s2_x, labels in tqdm(test_loader, desc=f"Evaluating {checkpoint_path.name}"):
            s1_x = s1_x.to(device)
            s2_x = s2_x.to(device)

            outputs = model(s1_x, s2_x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(labels.numpy().flatten())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # Exclude invalid/NoData pixels (IGNORE_INDEX = 255)
    valid_mask = (y_true != IGNORE_INDEX)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Core scientific metrics
    pixel_acc = float(accuracy_score(y_true_valid, y_pred_valid))
    precision_w = float(precision_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0))
    recall_w = float(recall_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0))

    # Confusion Matrix & IoU on valid pixels
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(NUM_CLASSES)))
    iou_per_class = calculate_per_class_iou(cm)
    miou = float(np.nanmean(iou_per_class))

    # Per-class precision, recall, F1
    precision_per_class = precision_score(y_true_valid, y_pred_valid, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    recall_per_class = recall_score(y_true_valid, y_pred_valid, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    f1_per_class = f1_score(y_true_valid, y_pred_valid, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)

    # Per-class support & metrics
    per_class_list = []
    for i in range(NUM_CLASSES):
        c_support = int(np.sum(y_true_valid == i))
        c_pred_count = int(np.sum(y_pred_valid == i))
        per_class_list.append({
            "Class_ID": i,
            "Original_Class_ID": model_to_original[str(i)],
            "Class_Name": class_names[i],
            "Precision": float(precision_per_class[i]),
            "Recall": float(recall_per_class[i]),
            "F1": float(f1_per_class[i]),
            "IoU": float(iou_per_class[i]),
            "Ground_Truth_Pixels": c_support,
            "Predicted_Pixels": c_pred_count,
        })

    metrics = {
        "pixel_accuracy": pixel_acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
        "miou": miou,
        "total_test_pixels": int(y_true_valid.size),
        "total_raw_pixels": int(y_true.size),
        "invalid_pixels_ignored": int(np.sum(~valid_mask)),
        "total_test_patches": len(test_dataset),
    }

    if save_artifacts:
        eval_dir = BASE_DIR / "outputs" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Save verifiable evaluation record
        save_evaluation_record(
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            per_class_metrics=per_class_list,
            confusion_matrix=cm.tolist(),
            eval_dir=eval_dir,
        )

        # Plot confusion matrix
        plt.figure(figsize=(10, 8), dpi=200)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
        plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
        plt.title(f"Confusion Matrix — {checkpoint_path.name}\n(Pixel Acc: {pixel_acc*100:.2f}%, mIoU: {miou:.4f})", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = eval_dir / f"{checkpoint_path.stem}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Update legacy evaluation_results.txt if this is the colab best model
        if "colab" in checkpoint_path.name.lower():
            txt_path = eval_dir / "evaluation_results.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Evaluation Results on Test Set (Model: {checkpoint_path.name})\n")
                f.write("============================================================\n")
                f.write(f"Checkpoint SHA-256: {compute_file_sha256(checkpoint_path)}\n")
                f.write(f"Dataset Fingerprint: {compute_dataset_fingerprint()['dataset_fingerprint']}\n")
                f.write(f"Pixel Accuracy:     {pixel_acc:.4f}\n")
                f.write(f"Precision:          {precision_w:.4f}\n")
                f.write(f"Recall:             {recall_w:.4f}\n")
                f.write(f"F1-score:           {f1_w:.4f}\n")
                f.write(f"mIoU:               {miou:.4f}\n")

            # Update per_class_metrics.csv
            pd.DataFrame(per_class_list).to_csv(eval_dir / "per_class_metrics.csv", index=False)
            # Copy confusion matrix to standard location
            import shutil
            shutil.copy2(cm_path, eval_dir / "confusion_matrix.png")

    print(f"\n[SUCCESS] Evaluation complete for {checkpoint_path.name}:")
    print(f"  Pixel Accuracy: {pixel_acc:.4f} ({pixel_acc*100:.2f}%)")
    print(f"  Precision:      {precision_w:.4f}")
    print(f"  Recall:         {recall_w:.4f}")
    print(f"  F1 Score:       {f1_w:.4f}")
    print(f"  mIoU:           {miou:.4f}")

    return {
        "metrics": metrics,
        "per_class": per_class_list,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FusionLand AI Model Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/fusion_colab_best.pth",
        help="Path to checkpoint .pth file",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    ckpt = BASE_DIR / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    run_evaluation(ckpt, device_name=args.device, batch_size=args.batch_size)
