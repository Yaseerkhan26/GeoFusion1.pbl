"""
===============================================================================
File: src/models/sentinel1_encoder.py
Purpose: Lightweight CNN Feature Encoder for Sentinel-1 SAR Imagery (VV, VH)

Description:
    Processes 2-channel Sentinel-1 SAR inputs (VV, VH) through sequential
    2D convolutions with batch normalization and ReLU activations.
    Preserves spatial resolution (256x256) for pixel-level feature extraction.
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

from src.models.config import SENTINEL1_CHANNELS, FEATURE_CHANNELS


class Sentinel1Encoder(nn.Module):
    """
    CNN Encoder for Sentinel-1 SAR input tensors of shape [Batch, 2, 256, 256].
    """

    def __init__(self, in_channels=SENTINEL1_CHANNELS, out_channels=FEATURE_CHANNELS):
        """
        Args:
            in_channels (int): Number of Sentinel-1 SAR input bands (default: 2 -> VV, VH).
            out_channels (int): Target feature map channel depth (default: 32).
        """
        super(Sentinel1Encoder, self).__init__()

        mid_channels = out_channels // 2  # 16 channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input SAR tensor of shape [Batch, 2, Height, Width].

        Returns:
            torch.Tensor: Feature map tensor of shape [Batch, out_channels, Height, Width].
        """
        return self.encoder(x)


if __name__ == "__main__":
    model = Sentinel1Encoder()
    dummy_input = torch.randn(2, 2, 256, 256)
    out = model(dummy_input)
    print(f"[TEST] Sentinel-1 Encoder Input: {dummy_input.shape} -> Output: {out.shape}")
