"""
===============================================================================
File: src/models/sentinel2_encoder.py
Purpose: Lightweight CNN Feature Encoder for Sentinel-2 Optical Imagery (6 Bands)

Description:
    Processes 6-channel Sentinel-2 optical inputs (B2, B3, B4, B8, B11, B12)
    through sequential 2D convolutions with batch normalization and ReLU activations.
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

from src.models.config import SENTINEL2_CHANNELS, FEATURE_CHANNELS


class Sentinel2Encoder(nn.Module):
    """
    CNN Encoder for Sentinel-2 Optical input tensors of shape [Batch, 6, 256, 256].
    """

    def __init__(self, in_channels=SENTINEL2_CHANNELS, out_channels=FEATURE_CHANNELS):
        """
        Args:
            in_channels (int): Number of Sentinel-2 optical input bands (default: 6).
            out_channels (int): Target feature map channel depth (default: 32).
        """
        super(Sentinel2Encoder, self).__init__()

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
            x (torch.Tensor): Input Optical tensor of shape [Batch, 6, Height, Width].

        Returns:
            torch.Tensor: Feature map tensor of shape [Batch, out_channels, Height, Width].
        """
        return self.encoder(x)


if __name__ == "__main__":
    model = Sentinel2Encoder()
    dummy_input = torch.randn(2, 6, 256, 256)
    out = model(dummy_input)
    print(f"[TEST] Sentinel-2 Encoder Input: {dummy_input.shape} -> Output: {out.shape}")
