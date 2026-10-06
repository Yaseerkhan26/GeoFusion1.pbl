"""
===============================================================================
File: src/training/trainer.py
Purpose: Modular Model Trainer for Sentinel-1, Sentinel-2, & Multimodal Fusion Networks

Description:
    Provides a unified training and evaluation engine supporting:
    1. Sentinel-1-only SAR baseline model (Sentinel1OnlyNet)
    2. Sentinel-2-only Optical baseline model (Sentinel2OnlyNet)
    3. Multimodal SAR-Optical Fusion model (MultimodalFusionNet)

    Features:
    - Modular training & validation epoch execution
    - Pixel-wise accuracy evaluation
    - AdamW optimizer & ReduceLROnPlateau scheduler
    - Early stopping and best model checkpoint saving
    - Full training metrics history tracking
===============================================================================
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Append project root to system path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.config import MODELS_DIR, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE
from src.models.config import NUM_CLASSES, IGNORE_INDEX
from src.models.fusion_model import MultimodalFusionNet
from src.models.sentinel1_model import Sentinel1OnlyNet
from src.models.sentinel2_model import Sentinel2OnlyNet


def calculate_pixel_accuracy(preds, labels, ignore_index=IGNORE_INDEX):
    """
    Computes global pixel-wise classification accuracy, excluding invalid pixels (ignore_index).
    """
    valid = (labels != ignore_index)
    correct = ((preds == labels) & valid).sum().item()
    total = valid.sum().item()
    return correct / total if total > 0 else 0.0


class EarlyStopping:
    """
    Tracks validation metric improvement (e.g. val_loss min or val_miou max) and signals early stopping.
    """

    def __init__(self, patience=7, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.early_stop = False

    def __call__(self, val_score):
        if self.mode == "max":
            improved = val_score > self.best_score + self.min_delta
        else:
            improved = val_score < self.best_score - self.min_delta

        if improved:
            self.best_score = val_score
            self.counter = 0
            return True  # Indicates new best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelTrainer:
    """
    Modular engine for training, validating, and saving GeoFusion AI models.
    """

    def __init__(
        self,
        model_type="fusion",
        model=None,
        num_classes=NUM_CLASSES,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=None,
        save_dir=MODELS_DIR,
        **kwargs
    ):
        """
        Args:
            model_type (str): Mode selector - "fusion", "sentinel1", or "sentinel2".
            model (nn.Module, optional): Pre-constructed PyTorch model.
            num_classes (int): Number of target classes.
            lr (float): Initial learning rate for AdamW optimizer.
            weight_decay (float): L2 regularization weight decay.
            device (torch.device or str, optional): Hardware execution target ('cuda' or 'cpu').
            save_dir (Path or str): Target directory for saving checkpoints.
        """
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize requested model architecture if not passed explicitly
        if model is not None:
            self.model = model.to(self.device)
        elif self.model_type in ["sentinel1", "sentinel1_only", "sar"]:
            self.model = Sentinel1OnlyNet(num_classes=self.num_classes).to(self.device)
            self.model_type = "sentinel1"
        elif self.model_type in ["sentinel2", "sentinel2_only", "optical"]:
            self.model = Sentinel2OnlyNet(num_classes=self.num_classes).to(self.device)
            self.model_type = "sentinel2"
        else:
            self.model = MultimodalFusionNet(num_classes=self.num_classes).to(self.device)
            self.model_type = "fusion"

        # Optimization & Loss setup with explicit ignore_index for invalid NoData pixels
        if kwargs.get('class_weights') is not None:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=IGNORE_INDEX,
                weight=kwargs.get('class_weights').to(self.device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_miou": [],
        }

    def _forward_pass(self, s1_batch, s2_batch):
        """
        Executes model forward pass based on active modality configuration.
        """
        if self.model_type == "sentinel1":
            return self.model(s1_batch)
        elif self.model_type == "sentinel2":
            return self.model(s2_batch)
        else:  # fusion
            return self.model(s1_batch, s2_batch)

    def train_epoch(self, dataloader):
        """
        Executes single training epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_pixels = 0
        correct_pixels = 0

        for s1_batch, s2_batch, labels in dataloader:
            s1_batch = s1_batch.to(self.device)
            s2_batch = s2_batch.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self._forward_pass(s1_batch, s2_batch)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * s1_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            valid_mask = (labels != IGNORE_INDEX)
            correct_pixels += ((preds == labels) & valid_mask).sum().item()
            total_pixels += valid_mask.sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(self, dataloader):
        """
        Executes single validation epoch with per-class confusion matrix and mIoU computation.
        """
        import numpy as np
        self.model.eval()
        running_loss = 0.0
        total_pixels = 0
        correct_pixels = 0
        
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        with torch.no_grad():
            for s1_batch, s2_batch, labels in dataloader:
                s1_batch = s1_batch.to(self.device)
                s2_batch = s2_batch.to(self.device)
                labels = labels.to(self.device)

                logits = self._forward_pass(s1_batch, s2_batch)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * s1_batch.size(0)

                preds = torch.argmax(logits, dim=1)
                valid_mask = (labels != IGNORE_INDEX)
                correct_pixels += ((preds == labels) & valid_mask).sum().item()
                total_pixels += valid_mask.sum().item()

                p_valid = preds[valid_mask].cpu().numpy()
                l_valid = labels[valid_mask].cpu().numpy()
                for p_val, l_val in zip(p_valid, l_valid):
                    if 0 <= l_val < self.num_classes and 0 <= p_val < self.num_classes:
                        cm[l_val, p_val] += 1

        val_loss = running_loss / len(dataloader.dataset)
        val_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        # Calculate mIoU across target classes
        intersection = np.diag(cm)
        ground_truth = cm.sum(axis=1)
        predicted = cm.sum(axis=0)
        union = ground_truth + predicted - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        val_miou = float(np.nanmean(iou))

        return val_loss, val_acc, val_miou

    def fit(self, train_loader, val_loader, num_epochs=10, patience=7, checkpoint_name=None, select_by="val_miou"):
        """
        Executes full multi-epoch training pipeline with class-balanced model selection and early stopping.
        select_by: 'val_miou' (default, class-balanced mIoU) or 'val_loss'.
        """
        if checkpoint_name is None:
            checkpoint_name = f"{self.model_type}_best.pth"

        checkpoint_path = self.save_dir / checkpoint_name
        mode = "max" if select_by == "val_miou" else "min"
        early_stopping = EarlyStopping(patience=patience, mode=mode)

        print(f"=== Starting Training for Model: {self.model_type.upper()} ===")
        print(f"Device: {self.device} | Epochs: {num_epochs} | Selection: {select_by} ({mode}) | Checkpoint: {checkpoint_path}")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_miou = self.validate_epoch(val_loader)

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_miou"].append(val_miou)

            score_to_track = val_miou if select_by == "val_miou" else val_loss
            is_best = early_stopping(score_to_track)
            if is_best:
                torch.save(self.model.state_dict(), checkpoint_path)
                saved_flag = " [CHECKPOINT SAVED]"
            else:
                saved_flag = ""

            lr_curr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                f"Val mIoU: {val_miou:.4f} | LR: {lr_curr:.6f}{saved_flag}"
            )

            if early_stopping.early_stop:
                print(f"[INFO] Early stopping triggered after {epoch} epochs.")
                break

        return self.history



if __name__ == "__main__":
    print("=== Testing ModelTrainer Initialization ===")
    for mtype in ["sentinel1", "sentinel2", "fusion"]:
        trainer = ModelTrainer(model_type=mtype)
        print(f"[TEST] Successfully initialized trainer for '{mtype}' on device '{trainer.device}'.")
