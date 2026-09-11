"""
===============================================================================
File: colab_train_fusion.py
Purpose: Google Colab specific entry point to train ONLY the Fusion Model.
===============================================================================
"""

import sys
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Ensure the project root is in the system path so src module imports work
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MODELS_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    RANDOM_SEED,
)
from src.models.config import NUM_CLASSES
from src.training.dataset import MultimodalSatelliteDataset
from src.training.trainer import ModelTrainer

def run_colab_fusion_training(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    print("=== Starting GeoFusion AI Fusion Model Training on Google Colab ===")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")
    
    torch.manual_seed(RANDOM_SEED)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("[ERROR] Dataset splits not found. Make sure you mounted Google Drive correctly and paths are intact.")

    print("[INFO] Loading datasets...")
    train_dataset = MultimodalSatelliteDataset(split_csv_path=train_csv)
    val_dataset = MultimodalSatelliteDataset(split_csv_path=val_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Initialize Trainer for Fusion Model Only
    trainer = ModelTrainer(
        model_type="fusion",
        num_classes=NUM_CLASSES,
        lr=LEARNING_RATE,
        save_dir=MODELS_DIR,
        device=device
    )

    checkpoint_name = "fusion_colab_best.pth"
    print("\n" + "=" * 52)
    print("=== TRAINING FUSION MODEL ===")
    print("=" * 52)

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        checkpoint_name=checkpoint_name,
    )

    # Save History
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = REPORTS_DIR / "fusion_colab_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print("\n" + "=" * 52)
    print("STAGE 6 COLAB TRAINING COMPLETE")
    print("=" * 52)
    print(f"[SUCCESS] Best Fusion Model saved to: {MODELS_DIR / checkpoint_name}")
    print(f"[SUCCESS] Training history saved to: {history_path}")

if __name__ == "__main__":
    run_colab_fusion_training()
