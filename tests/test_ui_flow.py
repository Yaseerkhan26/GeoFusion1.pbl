"""
===============================================================================
Comprehensive End-to-End Headless Flow Test for FusionLand AI Application
Executes each page function and verifies zero runtime exceptions or unhandled errors.
===============================================================================
"""

import sys
import unittest
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app
from src.utils.provenance import load_evaluation_for_checkpoint, get_all_checkpoints


class TestUIFlow(unittest.TestCase):

    def setUp(self):
        self.active_ckpt = "fusion_colab_best.pth"
        self.ckpt_path = BASE_DIR / "models" / self.active_ckpt
        self.eval_status, self.eval_data = load_evaluation_for_checkpoint(self.ckpt_path)
        self.model, self.err = app.load_model_from_checkpoint(self.active_ckpt)
        self.assertIsNone(self.err, f"Model failed to load: {self.err}")
        self.assertIsNotNone(self.model)

    def test_01_overview(self):
        """Verify Overview page logic runs without exception."""
        dev_str = "CPU Inference"
        app.page_overview(self.active_ckpt, self.eval_status, self.eval_data, dev_str)

    def test_02_satellite_explorer(self):
        """Verify Satellite Explorer functions run without exception."""
        app.page_satellite_explorer()

    def test_03_ai_classification(self):
        """Verify AI Classification page runs without exception."""
        app.page_ai_classification(self.model, self.active_ckpt)

    def test_04_gis_map(self):
        """Verify GIS map page runs without exception."""
        app.page_gis_map()

    def test_05_analytics(self):
        """Verify Analytics page runs without exception."""
        app.page_analytics(self.eval_data)

    def test_06_model_performance(self):
        """Verify Model Performance page runs without exception."""
        app.page_model_performance(self.active_ckpt, self.eval_status, self.eval_data)

    def test_07_data_quality(self):
        """Verify Data Quality page runs without exception."""
        app.page_data_quality()

    def test_08_experiments(self):
        """Verify Experiments page runs without exception."""
        app.page_experiments()

    def test_09_reports(self):
        """Verify Reports page runs without exception."""
        app.page_reports(self.active_ckpt, self.eval_data)

    def test_10_about(self):
        """Verify About page runs without exception."""
        app.page_about()


if __name__ == "__main__":
    unittest.main()
