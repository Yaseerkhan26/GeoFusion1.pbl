"""
===============================================================================
Unit Tests for MultimodalFusionNet Model Architecture and Encoders
===============================================================================
"""

import unittest
import torch
from pathlib import Path

from src.models.config import NUM_CLASSES
from src.models.fusion_model import MultimodalFusionNet
from src.models.sentinel1_encoder import Sentinel1Encoder
from src.models.sentinel2_encoder import Sentinel2Encoder

BASE_DIR = Path(__file__).resolve().parent.parent


class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.s1_encoder = Sentinel1Encoder(in_channels=2, out_channels=32)
        self.s2_encoder = Sentinel2Encoder(in_channels=6, out_channels=32)
        self.fusion_net = MultimodalFusionNet(num_classes=NUM_CLASSES)

    def test_sentinel1_encoder_dimensions(self):
        """Verify S1 SAR encoder maps [B, 2, 256, 256] -> [B, 32, 256, 256]."""
        dummy_s1 = torch.randn(2, 2, 256, 256)
        out = self.s1_encoder(dummy_s1)
        self.assertEqual(out.shape, (2, 32, 256, 256))

    def test_sentinel2_encoder_dimensions(self):
        """Verify S2 Optical encoder maps [B, 6, 256, 256] -> [B, 32, 256, 256]."""
        dummy_s2 = torch.randn(2, 6, 256, 256)
        out = self.s2_encoder(dummy_s2)
        self.assertEqual(out.shape, (2, 32, 256, 256))

    def test_fusion_model_forward(self):
        """Verify MultimodalFusionNet fusion forward pass dimensions."""
        dummy_s1 = torch.randn(2, 2, 256, 256)
        dummy_s2 = torch.randn(2, 6, 256, 256)
        logits = self.fusion_net(dummy_s1, dummy_s2)
        self.assertEqual(logits.shape, (2, 8, 256, 256))

    def test_final_checkpoint_loading(self):
        """Verify primary final checkpoint loads into MultimodalFusionNet without error."""
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        self.assertTrue(ckpt_path.exists(), "Final model checkpoint missing!")

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.fusion_net.load_state_dict(state_dict)
        self.fusion_net.eval()

        dummy_s1 = torch.randn(1, 2, 256, 256)
        dummy_s2 = torch.randn(1, 6, 256, 256)
        with torch.no_grad():
            logits = self.fusion_net(dummy_s1, dummy_s2)
        self.assertEqual(logits.shape, (1, 8, 256, 256))


if __name__ == "__main__":
    unittest.main()
