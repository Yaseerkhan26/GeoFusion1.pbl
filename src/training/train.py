"""
===============================================================================
File: src/training/train.py
Purpose: Sequential Multi-Model Training Entry Point for GeoFusion AI (Stage 6)

Description:
    Trains and compares three separate architectures sequentially:
    1. Sentinel-1 SAR-only semantic segmentation network (Sentinel1OnlyNet)
    2. Sentinel-2 Optical-only semantic segmentation network (Sentinel2OnlyNet)
    3. Multimodal SAR-Optical Data Fusion Network (MultimodalFusionNet)

    Outputs & Artifacts Generated:
    - Models: models/sentinel1_best.pth, models/sentinel2_best.pth, models/fusion_best.pth
    - Histories: outputs/reports/training_history.json, outputs/reports/<model>_history.json
    - Reports: outputs/reports/training_report.txt
    - Graphs: outputs/graphs/training_loss_curves.png, outputs/graphs/training_accuracy_curves.png, outputs/graphs/model_comparison.png
===============================================================================
"""

import json
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MODELS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
    SPLITS_DIR,
    RANDOM_SEED,
)
from src.models.config import NUM_CLASSES
from src.training.dataset import MultimodalSatelliteDataset
from src.training.trainer import ModelTrainer


def save_training_artifacts(histories):
    """
    Exports training histories, loss/accuracy curves, and text summary reports.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON histories
    full_history_path = REPORTS_DIR / "training_history.json"
    with open(full_history_path, "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=4)

    for model_name, hist in histories.items():
        individual_hist_path = REPORTS_DIR / f"{model_name}_history.json"
        with open(individual_hist_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=4)

    print(f"[INFO] Exported JSON training histories to {REPORTS_DIR}")

    # 2. Generate Training Loss & Accuracy Graphs
    plt.figure(figsize=(10, 5))
    for model_name, hist in histories.items():
        epochs_range = range(1, len(hist["train_loss"]) + 1)
        plt.plot(epochs_range, hist["train_loss"], label=f"{model_name.capitalize()} Train Loss", linestyle="--")
        plt.plot(epochs_range, hist["val_loss"], label=f"{model_name.capitalize()} Val Loss", linestyle="-")

    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.title("GeoFusion AI — Training & Validation Loss Comparison")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    loss_curve_path = GRAPHS_DIR / "training_loss_curves.png"
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_name, hist in histories.items():
        epochs_range = range(1, len(hist["train_acc"]) + 1)
        plt.plot(epochs_range, [acc * 100 for acc in hist["train_acc"]], label=f"{model_name.capitalize()} Train Acc", linestyle="--")
        plt.plot(epochs_range, [acc * 100 for acc in hist["val_acc"]], label=f"{model_name.capitalize()} Val Acc", linestyle="-")

    plt.xlabel("Epochs")
    plt.ylabel("Pixel Accuracy (%)")
    plt.title("GeoFusion AI — Training & Validation Pixel Accuracy Comparison")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    acc_curve_path = GRAPHS_DIR / "training_accuracy_curves.png"
    plt.savefig(acc_curve_path, dpi=300)
    plt.close()

    # 3. Generate Summary Comparison Chart
    plt.figure(figsize=(8, 5))
    models = list(histories.keys())
    final_val_losses = [hist["val_loss"][-1] if hist["val_loss"] else 0 for hist in histories.values()]
    final_val_accs = [hist["val_acc"][-1] * 100 if hist["val_acc"] else 0 for hist in histories.values()]

    bar_width = 0.35
    x = range(len(models))

    plt.bar([i - bar_width/2 for i in x], final_val_losses, bar_width, label="Final Val Loss", color="#d62728")
    plt.bar([i + bar_width/2 for i in x], [acc / 100 for acc in final_val_accs], bar_width, label="Final Val Acc", color="#2ca02c")
    plt.xticks(x, [m.upper() for m in models])
    plt.ylabel("Metric Scale")
    plt.title("GeoFusion AI — Model Benchmark Summary")
    plt.legend()
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    comparison_path = GRAPHS_DIR / "model_comparison.png"
    plt.savefig(comparison_path, dpi=300)
    plt.close()

    print(f"[INFO] Exported training visualization plots to {GRAPHS_DIR}")

    # 4. Save Text Training Report
    report_path = REPORTS_DIR / "training_report.txt"
    report_lines = [
        "===============================================================================",
        "GEOFUSION AI: STAGE 6 MODEL TRAINING & COMPARISON REPORT",
        "===============================================================================\n",
        "1. OVERVIEW:",
        "   - Project: Multimodal Satellite Imagery Land Cover Classification",
        "   - Stage: Stage 6 Model Training & Modular Trainer Setup",
        f"   - Target Classes: {NUM_CLASSES}",
        "   - Input Modalities: Sentinel-1 (2 SAR channels), Sentinel-2 (6 Optical channels)\n",
        "2. MODEL BENCHMARK PERFORMANCE:",
        "   Model Name      | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc",
        "   -----------------------------------------------------------------------------------",
    ]

    for model_name, hist in histories.items():
        tr_l = hist["train_loss"][-1] if hist["train_loss"] else 0.0
        va_l = hist["val_loss"][-1] if hist["val_loss"] else 0.0
        tr_a = hist["train_acc"][-1] * 100 if hist["train_acc"] else 0.0
        va_a = hist["val_acc"][-1] * 100 if hist["val_acc"] else 0.0
        report_lines.append(
            f"   {model_name.upper():<15} | {tr_l:<16.4f} | {va_l:<14.4f} | {tr_a:<15.2f}% | {va_a:<13.2f}%"
        )

    report_lines.extend([
        "\n3. SAVED MODEL CHECKPOINTS:",
        f"   - Sentinel-1 Model:  {MODELS_DIR / 'sentinel1_best.pth'}",
        f"   - Sentinel-2 Model:  {MODELS_DIR / 'sentinel2_best.pth'}",
        f"   - Multimodal Fusion: {MODELS_DIR / 'fusion_best.pth'}",
        f"   - Initial Baseline:  {MODELS_DIR / 'fusion_model_best.pth'}\n",
        "===============================================================================",
        "[SUCCESS] Stage 6 Training Pipeline executed and reports generated.",
        "===============================================================================",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[INFO] Exported text summary report to {report_path}")


def train_single_model(model_type, train_loader, val_loader, epochs=NUM_EPOCHS, checkpoint_name=None):
    """
    Trains a single model architecture and saves its checkpoint.
    """
    header_title = {
        "sentinel1": "SENTINEL-1 MODEL",
        "sentinel2": "SENTINEL-2 MODEL",
        "fusion": "FUSION MODEL",
    }.get(model_type.lower(), f"{model_type.upper()} MODEL")

    print("\n" + "=" * 52)
    print(f"=== TRAINING {header_title} ===")
    print("=" * 52)

    trainer = ModelTrainer(
        model_type=model_type,
        num_classes=NUM_CLASSES,
        lr=LEARNING_RATE,
        save_dir=MODELS_DIR,
    )

    if checkpoint_name is None:
        checkpoint_name = f"{model_type}_best.pth"

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        checkpoint_name=checkpoint_name,
    )

    return history


def run_training_pipeline(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    Executes sequential training across all three model architectures:
    1. Sentinel-1 SAR-only model
    2. Sentinel-2 Optical-only model
    3. Multimodal Fusion model
    """
    print("=== Starting GeoFusion AI Multi-Model Training Pipeline ===")
    torch.manual_seed(RANDOM_SEED)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training split file missing: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation split file missing: {val_csv}")

    # Load shared PyTorch Datasets & DataLoaders
    train_dataset = MultimodalSatelliteDataset(split_csv_path=train_csv)
    val_dataset = MultimodalSatelliteDataset(split_csv_path=val_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"[INFO] Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
    )

    histories = {}

    # 1. Train Sentinel-1 SAR-only Model
    histories["sentinel1"] = train_single_model(
        model_type="sentinel1",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_name="sentinel1_best.pth",
    )

    # 2. Train Sentinel-2 Optical-only Model
    histories["sentinel2"] = train_single_model(
        model_type="sentinel2",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_name="sentinel2_best.pth",
    )

    # 3. Train Multimodal Fusion Model
    histories["fusion"] = train_single_model(
        model_type="fusion",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_name="fusion_best.pth",
    )

    # Save all training artifacts (histories, graphs, text report)
    save_training_artifacts(histories)

    # Summary of saved model paths
    print("\n" + "=" * 52)
    print("STAGE 6 TRAINING COMPLETE — SAVED MODEL CHECKPOINTS")
    print("=" * 52)
    s1_path = MODELS_DIR / "sentinel1_best.pth"
    s2_path = MODELS_DIR / "sentinel2_best.pth"
    fusion_path = MODELS_DIR / "fusion_best.pth"
    prev_fusion_path = MODELS_DIR / "fusion_model_best.pth"

    print(f"1. Sentinel-1 Model Checkpoint:  {s1_path}")
    print(f"2. Sentinel-2 Model Checkpoint:  {s2_path}")
    print(f"3. Multimodal Fusion Checkpoint: {fusion_path}")
    if prev_fusion_path.exists():
        print(f"4. Preserved Initial Checkpoint: {prev_fusion_path}")
    print("=" * 52 + "\n")

    return histories


if __name__ == "__main__":
    # Execute sequential training across all 3 models
    run_training_pipeline()
