"""
===============================================================================
File: src/training/train.py
Purpose: PyTorch Model Training Pipeline for Multimodal Satellite Data Fusion

Description:
    Manages end-to-end model training, validation loops, loss evaluation,
    learning rate scheduling, and model checkpoint saving into models/.
===============================================================================
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Append project root to system path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.fusion_model import MultimodalFusionNetwork
from src.utils.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODELS_DIR,
    NUM_CLASSES, RANDOM_SEED
)


class MultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset skeleton for paired Sentinel-1, Sentinel-2, and label patches.
    """

    def __init__(self, patch_dir=None):
        """
        Args:
            patch_dir (Path/str): Directory containing preprocessed .npy/.npz patch files.
        """
        self.patch_dir = patch_dir
        # TODO: Load index list of patch files from disk

    def __len__(self):
        return 100  # Starter placeholder dataset length

    def __getitem__(self, idx):
        # Starter dummy tensor returns (SAR: 2x128x128, Optical: 10x128x128, Label: 128x128)
        s1_patch = torch.randn(2, 128, 128)
        s2_patch = torch.randn(10, 128, 128)
        label_patch = torch.randint(0, NUM_CLASSES, (128, 128))
        return s1_patch, s2_patch, label_patch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single epoch of training across all mini-batches.
    """
    model.train()
    total_loss = 0.0
    for step, (s1_imgs, s2_imgs, labels) in enumerate(dataloader):
        s1_imgs, s2_imgs, labels = s1_imgs.to(device), s2_imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(s1_imgs, s2_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def run_training_pipeline():
    """
    Executes the full multi-epoch training workflow and saves best weights to models/.
    """
    print("=== Starting Multimodal Model Training Pipeline ===")
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute hardware device: {device}")

    # Dataset & Dataloader
    dataset = MultimodalDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss Function, and Optimizer
    model = MultimodalFusionNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Model initialized. Hyperparameters: Epochs={NUM_EPOCHS}, BatchSize={BATCH_SIZE}, LR={LEARNING_RATE}")

    # Epoch Training Loop Placeholder
    for epoch in range(1, 3):  # Running 2 starter dummy iterations
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"[EPOCH {epoch}/{NUM_EPOCHS}] Average Loss: {loss:.4f}")

    # Save model weights checkpoint
    checkpoint_path = MODELS_DIR / "fusion_model_best.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SUCCESS] Starter checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    run_training_pipeline()
