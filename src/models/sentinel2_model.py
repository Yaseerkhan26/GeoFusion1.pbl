"""
===============================================================================
File: src/models/sentinel2_model.py
Purpose: PyTorch Feature Extractor Branch for Sentinel-2 Multispectral Optical Data

Description:
    Implements a dedicated Convolutional Neural Network (CNN) encoder branch
    tailored to process 10-channel Sentinel-2 optical/multispectral inputs
    (Visible, Red Edge, NIR, SWIR bands).
    Extracts spectral signatures and spatial contextual features.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sentinel2Encoder(nn.Module):
    """
    Feature Extraction Backbone for Sentinel-2 Multispectral imagery (10 bands).
    """

    def __init__(self, in_channels=10, feature_dim=128):
        """
        Args:
            in_channels (int): Number of optical bands (default: 10).
            feature_dim (int): Number of feature map channels output by encoder.
        """
        super(Sentinel2Encoder, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Convolutional Encoder Backbone Placeholder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass for optical spectral feature extraction.

        Args:
            x (torch.Tensor): Optical input tensor of shape (Batch, 10, Height, Width).

        Returns:
            torch.Tensor: Feature map tensor of shape (Batch, feature_dim, H/2, W/2).
        """
        out = self.layer1(x)
        out = self.layer2(out)
        return out


if __name__ == "__main__":
    print("=== Sentinel-2 Optical Model Architecture Placeholder ===")
    dummy_s2 = torch.randn(4, 10, 128, 128)  # Batch of 4, 10 optical bands, 128x128 image
    model = Sentinel2Encoder()
    output = model(dummy_s2)
    print(f"[TEST] Input shape: {dummy_s2.shape}")
    print(f"[TEST] Output feature shape: {output.shape}")
