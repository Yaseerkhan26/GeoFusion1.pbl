"""
===============================================================================
Unit and Integration Test Suite for FusionLand AI Platform
===============================================================================
"""

import unittest
from pathlib import Path
import torch
import numpy as np

# Project imports
from src.models.config import NUM_CLASSES
from src.models.dataset import MultimodalSatelliteDataset
from src.models.fusion_model import MultimodalFusionNet
from src.models.inference import (
    run_model_inference,
    compute_entropy_uncertainty,
    compute_prediction_error_map,
)
from src.utils.provenance import (
    compute_file_sha256,
    compute_dataset_fingerprint,
    get_checkpoint_metadata,
    load_evaluation_for_checkpoint,
)
from src.geospatial.geo_utils import (
    get_raster_metadata,
    calculate_pixel_area_m2,
    calculate_class_areas,
    load_sentinel1_native_overview,
    load_sentinel2_native_overview,
)

BASE_DIR = Path(__file__).resolve().parent.parent


class TestFusionLandAI(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")

    def test_class_mapping_integrity(self):
        """Verify that class mapping has exactly 8 classes matching ESA WorldCover."""
        self.assertEqual(NUM_CLASSES, 8)
        class_mapping_path = BASE_DIR / "data" / "class_mapping.json"
        self.assertTrue(class_mapping_path.exists())

    def test_dataset_splits_isolation(self):
        """Verify train, val, and test splits are non-empty and disjoint."""
        train_p = BASE_DIR / "data" / "splits" / "train.csv"
        val_p = BASE_DIR / "data" / "splits" / "val.csv"
        test_p = BASE_DIR / "data" / "splits" / "test.csv"

        self.assertTrue(train_p.exists())
        self.assertTrue(val_p.exists())
        self.assertTrue(test_p.exists())

        import pandas as pd
        df_train = pd.read_csv(train_p)
        df_val = pd.read_csv(val_p)
        df_test = pd.read_csv(test_p)

        self.assertEqual(len(df_train), 125)
        self.assertEqual(len(df_val), 25)
        self.assertEqual(len(df_test), 29)

        # Ensure no patch ID overlap
        train_ids = set(df_train["patch_id"])
        val_ids = set(df_val["patch_id"])
        test_ids = set(df_test["patch_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_checkpoint_provenance(self):
        """Verify checkpoint discovery, metadata extraction, and SHA-256."""
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        self.assertTrue(ckpt_path.exists())

        meta = get_checkpoint_metadata(ckpt_path)
        self.assertTrue(meta["compatible"])
        self.assertEqual(meta["architecture"], "MultimodalFusionNet")
        self.assertEqual(meta["num_classes"], 8)
        self.assertEqual(len(meta["sha256"]), 64)

    def test_model_forward_dimensions(self):
        """Verify MultimodalFusionNet input and output tensor shapes."""
        model = MultimodalFusionNet(num_classes=8)
        dummy_s1 = torch.randn(2, 2, 256, 256)
        dummy_s2 = torch.randn(2, 6, 256, 256)
        out = model(dummy_s1, dummy_s2)
        self.assertEqual(out.shape, (2, 8, 256, 256))

    def test_inference_and_uncertainty(self):
        """Verify inference output, probability bounds, and entropy normalization."""
        model = MultimodalFusionNet(num_classes=8)
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        dummy_s1 = torch.randn(2, 256, 256)
        dummy_s2 = torch.randn(6, 256, 256)

        res = run_model_inference(model, dummy_s1, dummy_s2, self.device, confidence_threshold=0.5)

        self.assertEqual(res["predictions"].shape, (256, 256))
        self.assertEqual(res["confidence_map"].shape, (256, 256))
        self.assertEqual(res["uncertainty_map"].shape, (256, 256))

        # Confidence must be in [0.0, 1.0]
        self.assertTrue(np.all(res["confidence_map"] >= 0.0))
        self.assertTrue(np.all(res["confidence_map"] <= 1.0))

        # Normalized entropy uncertainty must be in [0.0, 1.0]
        self.assertTrue(np.all(res["uncertainty_map"] >= 0.0))
        self.assertTrue(np.all(res["uncertainty_map"] <= 1.0))

    def test_error_map_agreement(self):
        """Verify prediction agreement calculation against ground truth."""
        pred = np.array([[1, 2], [3, 4]])
        gt = np.array([[1, 0], [3, 4]])
        err_mask, stats = compute_prediction_error_map(pred, gt)
        self.assertEqual(stats["total_pixels"], 4)
        self.assertEqual(stats["correct_pixels"], 3)
        self.assertEqual(stats["incorrect_pixels"], 1)
        self.assertEqual(stats["patch_accuracy"], 75.0)

    def test_geodesic_area_calculation(self):
        """Verify geodesic pixel area computation on WGS-84 ellipsoid."""
        # At latitude 14.464 degrees, 8.983e-5 deg resolution is ~9.8m x 9.8m -> ~96 m^2
        area_m2 = calculate_pixel_area_m2(14.464, 8.983152841195215e-05, 8.983152841195215e-05)
        self.assertGreater(area_m2, 90.0)
        self.assertLess(area_m2, 105.0)

        # 10,000 pixels is ~96.2 hectares
        areas = calculate_class_areas({"Cropland": 10000}, center_lat_deg=14.464)
        self.assertAlmostEqual(areas["Cropland"]["hectares"], 96.26, delta=1.0)

    def test_geotiff_raster_metadata_and_fallback(self):
        """Verify reading GeoTIFF rasters and S1 graceful fallback."""
        s2_vis, s2_meta = load_sentinel2_native_overview(mode="RGB", max_dim=256)
        self.assertEqual(s2_vis.shape[2], 3)
        self.assertEqual(s2_meta["crs"], "EPSG:4326")

        s1_vis, s1_meta, src_desc = load_sentinel1_native_overview(band="VV", max_dim=256)
        self.assertEqual(s1_vis.ndim, 2)
        self.assertIn("Sentinel-1", src_desc)

    def test_evaluation_freshness(self):
        """Verify evaluation status, provenance, and structural metric integrity for GeoFusion_AI_Final.pth."""
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        status, eval_data = load_evaluation_for_checkpoint(ckpt_path)
        self.assertEqual(status, "VALIDATED")
        self.assertIsNotNone(eval_data)
        
        # Structural & range checks instead of arbitrary accuracy thresholds
        acc = eval_data["metrics"]["pixel_accuracy"]
        miou = eval_data["metrics"]["miou"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(miou, 0.0)
        self.assertLessEqual(miou, 1.0)

        # Provenance verification
        self.assertEqual(eval_data["checkpoint_sha256"], compute_file_sha256(ckpt_path))
        self.assertEqual(eval_data["dataset_fingerprint"], compute_dataset_fingerprint(BASE_DIR / "data")["dataset_fingerprint"])

        # Confusion matrix shape (8 classes)
        cm = eval_data["confusion_matrix"]
        self.assertEqual(len(cm), 8)
        self.assertEqual(len(cm[0]), 8)


    def test_pixel_inspector_affine(self):
        """Verify Affine georeferencing coordinate query and band sampling."""
        from src.geospatial.geo_utils import inspect_pixel, inspect_coordinate
        s2_path = BASE_DIR / "data" / "processed" / "sentinel2" / "sentinel2_processed.tif"
        if not s2_path.exists():
            s2_path = BASE_DIR / "data" / "raw" / "sentinel2" / "Sentinel2_Bands.tif"

        res = inspect_pixel(s2_path, 100, 100)
        self.assertTrue(res["valid"])
        self.assertIn("lat", res)
        self.assertIn("lon", res)
        self.assertEqual(res["crs"], "EPSG:4326")
        self.assertEqual(len(res["bands"]), res["count"])

        # Inverse query using coordinates
        coord_res = inspect_coordinate(s2_path, res["lat"], res["lon"])
        self.assertTrue(coord_res["valid"])
        self.assertAlmostEqual(coord_res["row"], 100, delta=1)
        self.assertAlmostEqual(coord_res["col"], 100, delta=1)

    def test_inference_numerical_safety(self):
        """Verify that inference rejects NaNs and invalid shapes safely."""
        model = MultimodalFusionNet(num_classes=8)
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        # NaN injection
        nan_s1 = torch.full((2, 256, 256), float('nan'))
        valid_s2 = torch.randn(6, 256, 256)
        with self.assertRaises(ValueError):
            run_model_inference(model, nan_s1, valid_s2, self.device)

        # Invalid shape
        bad_s1 = torch.randn(3, 256, 256)
        with self.assertRaises(ValueError):
            run_model_inference(model, bad_s1, valid_s2, self.device)

    def test_uncertainty_tiers(self):
        """Verify uncertainty tier breakdown counts and percentages."""
        model = MultimodalFusionNet(num_classes=8)
        ckpt_path = BASE_DIR / "models" / "GeoFusion_AI_Final.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        s1 = torch.randn(2, 256, 256)
        s2 = torch.randn(6, 256, 256)
        res = run_model_inference(model, s1, s2, self.device)

        self.assertIn("uncertainty_tiers", res)
        tiers = res["uncertainty_tiers"]
        self.assertIn("Low (<0.30)", tiers)
        self.assertIn("Medium (0.30-0.60)", tiers)
        self.assertIn("High (>0.60)", tiers)

        total_counted = sum(t["count"] for t in tiers.values())
        self.assertEqual(total_counted, 256 * 256)

    def test_ui_components_import(self):
        """Verify UI components import and basic helper functions."""
        from src.utils.ui_components import render_metric_card, render_legend
        self.assertTrue(callable(render_metric_card))
        self.assertTrue(callable(render_legend))


if __name__ == "__main__":
    unittest.main()

