"""
===============================================================================
File: src/models/fusion_model.py
Purpose: Multimodal Data Fusion Network (MultimodalFusionNet) for Land Cover Segmentation

Description:
    Architecture:
    Sentinel-1 (2 SAR Bands)    Sentinel-2 (6 Optical Bands)
               │                           │
       Sentinel1Encoder            Sentinel2Encoder
               │                           │
       S1 Features (32 channels)   S2 Features (32 channels)
               └─────────────┬─────────────┘
                             │
                      Feature Concatenation (64 channels)
                             │
                      Conv2D (64 -> 32) + BN + ReLU
                             │
                      Conv2D (32 -> 16) + BN + ReLU
                             │
                      1x1 Conv2D (16 -> NUM_CLASSES)
                             │
                      Raw Logits [Batch, NUM_CLASSES, 256, 256]
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

from src.models.config import (
    SENTINEL1_CHANNELS,
    SENTINEL2_CHANNELS,
    FEATURE_CHANNELS,
    NUM_CLASSES,
)
from src.models.sentinel1_encoder import Sentinel1Encoder
from src.models.sentinel2_encoder import Sentinel2Encoder


class MultimodalFusionNet(nn.Module):
    """
    Multimodal Data Fusion Deep Learning Architecture for Pixel-wise Land Cover Classification.
    Fuses SAR (Sentinel-1) and Optical (Sentinel-2) feature maps.
    """

    def __init__(
        self,
        s1_channels=SENTINEL1_CHANNELS,
        s2_channels=SENTINEL2_CHANNELS,
        feature_channels=FEATURE_CHANNELS,
        num_classes=NUM_CLASSES,
    ):
        """
        Args:
            s1_channels (int): Sentinel-1 SAR input bands (default: 2).
            s2_channels (int): Sentinel-2 Optical input bands (default: 6).
            feature_channels (int): Per-modality encoder feature channels (default: 32).
            num_classes (int): Number of target land cover classes (default: dynamically loaded).
        """
        super(MultimodalFusionNet, self).__init__()

        self.num_classes = num_classes

        # Dual Encoders
        self.s1_encoder = Sentinel1Encoder(in_channels=s1_channels, out_channels=feature_channels)
        self.s2_encoder = Sentinel2Encoder(in_channels=s2_channels, out_channels=feature_channels)

        # Concatenated feature depth (32 + 32 = 64)
        fused_channels = feature_channels * 2

        # Classification / Fusion Head
        self.classifier = nn.Sequential(
            nn.Conv2d(fused_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_classes, kernel_size=1),  # 1x1 conv to num_classes logits
        )

    def forward(self, s1_x, s2_x):
        """
        Args:
            s1_x (torch.Tensor): Sentinel-1 tensor of shape [Batch, 2, 256, 256].
            s2_x (torch.Tensor): Sentinel-2 tensor of shape [Batch, 6, 256, 256].

        Returns:
            torch.Tensor: Pixel-wise class logits of shape [Batch, NUM_CLASSES, 256, 256].
        """
        # Feature extraction
        s1_features = self.s1_encoder(s1_x)  # [Batch, 32, 256, 256]
        s2_features = self.s2_encoder(s2_x)  # [Batch, 32, 256, 256]

        # Feature fusion via concatenation along channel dimension
        fused_features = torch.cat([s1_features, s2_features], dim=1)  # [Batch, 64, 256, 256]

        # Final pixel-wise class logits (No Softmax activation)
        logits = self.classifier(fused_features)  # [Batch, NUM_CLASSES, 256, 256]

        return logits


if __name__ == "__main__":
    model = MultimodalFusionNet(num_classes=8)
    dummy_s1 = torch.randn(2, 2, 256, 256)
    dummy_s2 = torch.randn(2, 6, 256, 256)

    out = model(dummy_s1, dummy_s2)
    print(f"[TEST] MultimodalFusionNet Forward Pass:")
    print(f"  Sentinel-1 Input: {dummy_s1.shape}")
    print(f"  Sentinel-2 Input: {dummy_s2.shape}")
    print(f"  Output Logits:    {out.shape}")
