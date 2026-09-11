"""
===============================================================================
File: src/models/sentinel1_model.py
Purpose: Single-Modality Sentinel-1 Baseline Architecture (Sentinel1OnlyNet)

Description:
    Processes 2-channel Sentinel-1 SAR inputs (VV, VH) through Sentinel1Encoder
    and a lightweight convolutional classification head to produce pixel-wise
    land cover predictions [Batch, NUM_CLASSES, 256, 256].
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

from src.models.config import SENTINEL1_CHANNELS, FEATURE_CHANNELS, NUM_CLASSES
from src.models.sentinel1_encoder import Sentinel1Encoder


class Sentinel1OnlyNet(nn.Module):
    """
    Single-modality SAR network for semantic segmentation of land cover using Sentinel-1.
    """

    def __init__(
        self,
        in_channels=SENTINEL1_CHANNELS,
        feature_channels=FEATURE_CHANNELS,
        num_classes=NUM_CLASSES,
    ):
        """
        Args:
            in_channels (int): Sentinel-1 SAR input channels (default: 2).
            feature_channels (int): Encoder output channels (default: 32).
            num_classes (int): Number of target land cover classes (default: 8).
        """
        super(Sentinel1OnlyNet, self).__init__()

        self.num_classes = num_classes
        self.encoder = Sentinel1Encoder(in_channels=in_channels, out_channels=feature_channels)

        self.classifier = nn.Sequential(
            nn.Conv2d(feature_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_classes, kernel_size=1),
        )

    def forward(self, s1_x):
        """
        Args:
            s1_x (torch.Tensor): Sentinel-1 tensor of shape [Batch, 2, 256, 256].

        Returns:
            torch.Tensor: Logits tensor of shape [Batch, NUM_CLASSES, 256, 256].
        """
        features = self.encoder(s1_x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = Sentinel1OnlyNet(num_classes=8)
    dummy_input = torch.randn(2, 2, 256, 256)
    out = model(dummy_input)
    print(f"[TEST] Sentinel1OnlyNet Input: {dummy_input.shape} -> Output: {out.shape}")
