"""
===============================================================================
File: src/models/fusion_model.py
Purpose: Multimodal Satellite Data Fusion PyTorch Architecture

Description:
    Combines Sentinel-1 (SAR) and Sentinel-2 (Optical) encoder branches through
    feature-level fusion (concatenation, attention mechanisms, or gating).
    Outputs dense per-pixel land cover classification maps (logits for N classes).
===============================================================================
"""

import sys
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Append project root to system path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.sentinel1_model import Sentinel1Encoder
from src.models.sentinel2_model import Sentinel2Encoder
from src.utils.config import NUM_CLASSES


class MultimodalFusionNetwork(nn.Module):
    """
    Multimodal Data Fusion Architecture combining SAR (Sentinel-1) and
    Multispectral Optical (Sentinel-2) features for Semantic Segmentation.
    """

    def __init__(self, s1_channels=2, s2_channels=10, num_classes=NUM_CLASSES, feature_dim=128):
        """
        Args:
            s1_channels (int): Channels in Sentinel-1 input (default: 2).
            s2_channels (int): Channels in Sentinel-2 input (default: 10).
            num_classes (int): Number of target land cover classes (default: 7).
            feature_dim (int): Intermediate representation dimension.
        """
        super(MultimodalFusionNetwork, self).__init__()

        # Dual Stream Modality Encoders
        self.s1_branch = Sentinel1Encoder(in_channels=s1_channels, feature_dim=feature_dim)
        self.s2_branch = Sentinel2Encoder(in_channels=s2_channels, feature_dim=feature_dim)

        # Feature Fusion Layer (Concatenation of SAR + Optical feature maps)
        fused_channels = feature_dim * 2
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Upsampling Segmentation Decoder Head
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, kernel_size=2, stride=2),  # Upsample back to original resolution
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, s1_x, s2_x):
        """
        Forward pass for multimodal land cover classification.

        Args:
            s1_x (torch.Tensor): SAR image tensor (Batch, 2, H, W).
            s2_x (torch.Tensor): Optical image tensor (Batch, 10, H, W).

        Returns:
            torch.Tensor: Classification logit predictions (Batch, num_classes, H, W).
        """
        # Extract features from each modality branch
        feat_s1 = self.s1_branch(s1_x)
        feat_s2 = self.s2_branch(s2_x)

        # Early-to-Intermediate Multimodal Feature Fusion
        fused_feats = torch.cat([feat_s1, feat_s2], dim=1)
        fused_feats = self.fusion_conv(fused_feats)

        # Final Segmentation Output Logits
        logits = self.decoder(fused_feats)
        return logits


if __name__ == "__main__":
    print("=== Multimodal Fusion Network Model Architecture Placeholder ===")
    dummy_s1 = torch.randn(2, 2, 128, 128)
    dummy_s2 = torch.randn(2, 10, 128, 128)

    model = MultimodalFusionNetwork()
    output_logits = model(dummy_s1, dummy_s2)

    print(f"[TEST] S1 Input shape: {dummy_s1.shape}")
    print(f"[TEST] S2 Input shape: {dummy_s2.shape}")
    print(f"[TEST] Logits Output shape: {output_logits.shape} (Batch, Classes, Height, Width)")
