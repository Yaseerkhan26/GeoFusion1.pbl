"""
===============================================================================
File: src/models/sentinel1_model.py
Purpose: PyTorch Feature Extractor Branch for Sentinel-1 Synthetic Aperture Radar (SAR)

Description:
    Implements a dedicated Convolutional Neural Network (CNN) encoder branch
    tailored to process 2-channel Sentinel-1 SAR inputs (VV and VH polarizations).
    Extracts spatial texture and radar backscatter feature representations.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sentinel1Encoder(nn.Module):
    """
    Feature Extraction Backbone for Sentinel-1 SAR imagery (VV + VH bands).
    """

    def __init__(self, in_channels=2, feature_dim=128):
        """
        Args:
            in_channels (int): Number of SAR polarizations (default: 2 -> VV, VH).
            feature_dim (int): Number of feature map channels output by encoder.
        """
        super(Sentinel1Encoder, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Convolutional Encoder Backbone Placeholder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass for SAR feature extraction.

        Args:
            x (torch.Tensor): SAR input tensor of shape (Batch, 2, Height, Width).

        Returns:
            torch.Tensor: Feature map tensor of shape (Batch, feature_dim, H/2, W/2).
        """
        out = self.layer1(x)
        out = self.layer2(out)
        return out


if __name__ == "__main__":
    print("=== Sentinel-1 SAR Model Architecture Placeholder ===")
    dummy_s1 = torch.randn(4, 2, 128, 128)  # Batch of 4, 2 SAR bands, 128x128 image
    model = Sentinel1Encoder()
    output = model(dummy_s1)
    print(f"[TEST] Input shape: {dummy_s1.shape}")
    print(f"[TEST] Output feature shape: {output.shape}")
