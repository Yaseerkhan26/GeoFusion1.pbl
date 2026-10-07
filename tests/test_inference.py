"""
===============================================================================
Unit Tests for Multimodal Inference, Uncertainty, and Error Analysis
===============================================================================
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from src.models.fusion_model import MultimodalFusionNet
from src.models.inference import (
    run_model_inference,
    compute_entropy_uncertainty,
    compute_prediction_error_map,
    colorize_categorical_map,
)

BASE_DIR = Path(__file__).resolve().parent.parent


class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MultimodalFusionNet(num_classes=8)
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.model.eval()

    def test_run_model_inference_output_contract(self):
        """Verify inference output dictionary keys, shapes, and value ranges."""
        dummy_s1 = torch.randn(2, 256, 256)
        dummy_s2 = torch.randn(6, 256, 256)

        res = run_model_inference(self.model, dummy_s1, dummy_s2, self.device, confidence_threshold=0.5)

        self.assertIn("predictions", res)
        self.assertIn("confidence_map", res)
        self.assertIn("uncertainty_map", res)
        self.assertIn("uncertainty_tiers", res)

        # Output spatial shape [256, 256]
        self.assertEqual(res["predictions"].shape, (256, 256))
        self.assertEqual(res["confidence_map"].shape, (256, 256))
        self.assertEqual(res["uncertainty_map"].shape, (256, 256))

        # Class bounds [0, 7]
        self.assertTrue(np.all(res["predictions"] >= 0))
        self.assertTrue(np.all(res["predictions"] <= 7))

        # Confidence bounds [0.0, 1.0]
        self.assertTrue(np.all(res["confidence_map"] >= 0.0))
        self.assertTrue(np.all(res["confidence_map"] <= 1.0))

        # Shannon entropy bounds [0.0, 1.0]
        self.assertTrue(np.all(res["uncertainty_map"] >= 0.0))
        self.assertTrue(np.all(res["uncertainty_map"] <= 1.0))

    def test_nan_and_invalid_input_rejection(self):
        """Verify inference safely raises ValueError on NaN tensors or incorrect channel counts."""
        nan_s1 = torch.full((2, 256, 256), float('nan'))
        valid_s2 = torch.randn(6, 256, 256)

        with self.assertRaises(ValueError):
            run_model_inference(self.model, nan_s1, valid_s2, self.device)

        bad_s1_channels = torch.randn(5, 256, 256)
        with self.assertRaises(ValueError):
            run_model_inference(self.model, bad_s1_channels, valid_s2, self.device)

    def test_colorize_categorical_map(self):
        """Verify colorization produces RGB image array [H, W, 3] in range [0, 1]."""
        pred_map = np.array([[0, 1], [3, 6]], dtype=np.int64)
        rgb = colorize_categorical_map(pred_map)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertTrue(np.all(rgb >= 0.0))
        self.assertTrue(np.all(rgb <= 1.0))


if __name__ == "__main__":
    unittest.main()
