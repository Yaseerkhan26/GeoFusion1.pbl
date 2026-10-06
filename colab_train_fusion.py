"""
===============================================================================
File: colab_train_fusion.py
Purpose: Google Colab specific entry point to train ONLY the Fusion Model.
===============================================================================
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the project root is in the system path so src module imports work
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    SPLITS_DIR,
    RANDOM_SEED,
)
from src.models.config import NUM_CLASSES, IGNORE_INDEX, PATCH_SIZE
from src.training.dataset import MultimodalSatelliteDataset
from src.training.trainer import ModelTrainer
import matplotlib.pyplot as plt

EPOCHS = 20

def run_colab_fusion_training(epochs=EPOCHS, batch_size=BATCH_SIZE):
    print("=== Starting GeoFusion AI Fusion Model Training ===")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")
    
    torch.manual_seed(RANDOM_SEED)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("[ERROR] Dataset splits not found.")

    print("[INFO] Loading datasets...")
    train_dataset = MultimodalSatelliteDataset(split_csv_path=train_csv, is_train=True)
    val_dataset = MultimodalSatelliteDataset(split_csv_path=val_csv, is_train=False)

    print("[INFO] Calculating class frequencies from training set (excluding invalid pixels)...")
    class_counts = np.zeros(NUM_CLASSES)
    patch_class_counts = []
    
    # Simple counting loader without shuffle
    temp_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    for _, _, labels in tqdm(temp_loader, desc="Counting class frequencies"):
        for b in range(labels.size(0)):
            lbl_b = labels[b]
            valid_b = lbl_b[lbl_b != IGNORE_INDEX]
            p_counts = np.zeros(NUM_CLASSES)
            unique, counts = torch.unique(valid_b, return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                if 0 <= u < NUM_CLASSES:
                    class_counts[u] += c
                    p_counts[u] = c
            patch_class_counts.append(p_counts)

    total_pixels = class_counts.sum()
    # Scientifically justified class-weight calculation with max cap = 10.0
    raw_weights = total_pixels / (NUM_CLASSES * np.maximum(class_counts, 1.0))
    capped_weights = np.clip(np.sqrt(raw_weights), 0.5, 10.0)
    class_weights = capped_weights / capped_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"[INFO] Raw Class Weights: {raw_weights.tolist()}")
    print(f"[INFO] Capped Class Weights (max=10.0): {capped_weights.tolist()}")
    print(f"[INFO] Final Normalized Class Loss Weights: {class_weights_tensor.tolist()}")

    # Compute Class-Aware Patch Sampling Weights for WeightedRandomSampler
    # Target Candidate Sampling Vector: [Tree=1.0, Shrub=2.0, Grass=2.0, Crop=1.0, Built=2.0, Bare=1.0, Water=6.0, Wetland=4.0]
    # Reduced Bare boost from 5.0 to 1.0 to eliminate false-positive flood on fallow cropland.
    class_boost = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 6.0, 4.0])
    sample_weights = []
    for p_cnts in patch_class_counts:
        w_sample = np.sum((p_cnts / (PATCH_SIZE * PATCH_SIZE)) * class_boost)
        sample_weights.append(max(w_sample, 0.1))
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Initialized Class-Aware Sampler with Candidate Vector {class_boost.tolist()} for {len(train_dataset)} training samples.")

    # Initialize Trainer for Fusion Model Only
    trainer = ModelTrainer(
        model_type="fusion",
        num_classes=NUM_CLASSES,
        lr=LEARNING_RATE,
        save_dir=MODELS_DIR,
        device=device,
        class_weights=class_weights_tensor
    )

    checkpoint_name = "fusion_colab_best.pth"

    print("\n" + "=" * 52)
    print(f"=== TRAINING FUSION MODEL FOR {epochs} EPOCHS (SELECTING BY VAL mIoU) ===")
    print("=" * 52)

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        checkpoint_name=checkpoint_name,
        select_by="val_miou",
    )

    # Save History and Graphs
    outputs_training_dir = BASE_DIR / "outputs" / "training"
    outputs_training_dir.mkdir(parents=True, exist_ok=True)
    history_path = outputs_training_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", linestyle="--")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], label="Train Acc", linestyle="--")
    plt.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], label="Val Acc", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Pixel Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    graph_path = outputs_training_dir / "loss_accuracy_graph.png"
    plt.savefig(graph_path, dpi=300)
    plt.close()

    best_val_loss = min(history["val_loss"])
    best_val_acc = max(history["val_acc"]) * 100

    print("\n" + "=" * 52)
    print("FINAL STAGE 6 TRAINING COMPLETED")
    print("=" * 52)
    print(f"- GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"- CUDA availability: {torch.cuda.is_available()}")
    print(f"- number of epochs completed: {len(history['train_loss'])}")
    print(f"- best validation loss: {best_val_loss:.4f}")
    print(f"- best validation accuracy: {best_val_acc:.2f}%")
    print(f"- checkpoint path: {MODELS_DIR / checkpoint_name}")
    print(f"- training history path: {history_path}")
    print(f"- graph path: {graph_path}")

if __name__ == "__main__":
    run_colab_fusion_training()
