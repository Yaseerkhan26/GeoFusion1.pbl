"""
===============================================================================
File: src/models/sentinel2_model.py
Purpose: Single-Modality Sentinel-2 Baseline Architecture (Sentinel2OnlyNet)

Description:
    Processes 6-channel Sentinel-2 Optical inputs (B2, B3, B4, B8, B11, B12) through Sentinel2Encoder
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

from src.models.config import SENTINEL2_CHANNELS, FEATURE_CHANNELS, NUM_CLASSES
from src.models.sentinel2_encoder import Sentinel2Encoder


class Sentinel2OnlyNet(nn.Module):
    """
    Single-modality Optical network for semantic segmentation of land cover using Sentinel-2.
    """

    def __init__(
        self,
        in_channels=SENTINEL2_CHANNELS,
        feature_channels=FEATURE_CHANNELS,
        num_classes=NUM_CLASSES,
    ):
        """
        Args:
            in_channels (int): Sentinel-2 Optical input channels (default: 6).
            feature_channels (int): Encoder output channels (default: 32).
            num_classes (int): Number of target land cover classes (default: 8).
        """
        super(Sentinel2OnlyNet, self).__init__()

        self.num_classes = num_classes
        self.encoder = Sentinel2Encoder(in_channels=in_channels, out_channels=feature_channels)

        self.classifier = nn.Sequential(
            nn.Conv2d(feature_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_classes, kernel_size=1),
        )

    def forward(self, s2_x):
        """
        Args:
            s2_x (torch.Tensor): Sentinel-2 tensor of shape [Batch, 6, 256, 256].

        Returns:
            torch.Tensor: Logits tensor of shape [Batch, NUM_CLASSES, 256, 256].
        """
        features = self.encoder(s2_x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = Sentinel2OnlyNet(num_classes=8)
    dummy_input = torch.randn(2, 6, 256, 256)
    out = model(dummy_input)
    print(f"[TEST] Sentinel2OnlyNet Input: {dummy_input.shape} -> Output: {out.shape}")
