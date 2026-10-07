# GeoFusion AI — Project Cleanup & Assembly Plan

**Project Title**: GeoFusion AI — Multimodal Satellite Intelligence  
**Target Root**: `GeoFusion AI`  
**Date**: 2026-10-07  

---

## Executive Overview

This plan details the systematic cleanup, organization, duplicate removal, path refactoring, and assembly of the **GeoFusion AI** repository into a production-grade, highly structured, and fully reproducible scientific software system.

---

## File Disposition Inventory

### 1. Files to KEEP
- `app/app.py`: Primary Streamlit Web Application entry point and view routing.
- `app.py`: Root pass-through entry point (supports `streamlit run app.py` and `streamlit run app/app.py`).
- `src/models/fusion_model.py`: Core `MultimodalFusionNet` architecture.
- `src/models/sentinel1_encoder.py`: Sentinel-1 SAR feature encoder.
- `src/models/sentinel2_encoder.py`: Sentinel-2 Optical feature encoder.
- `src/models/dataset.py`: PyTorch `MultimodalSatelliteDataset` dataset class.
- `src/models/config.py`: Architecture hyperparameter configuration and `NUM_CLASSES` loading.
- `src/models/inference.py`: Inference pipeline, categorical color mapping, error maps, uncertainty calculations.
- `src/training/trainer.py`: `Trainer` module for model training and validation loops.
- `src/evaluation/evaluate.py`: Evaluation suite and test metrics computation.
- `src/geospatial/geo_utils.py`: Raster reading, georeferencing, coordinate inspection, pixel area calculations.
- `src/utils/auth.py`: Authentication, password hashing, validation, session management, user database.
- `src/utils/config.py`: Centralized configuration tokens, color palettes, paths.
- `src/utils/provenance.py`: SHA-256 fingerprinting, dataset integrity checking, evaluation record persistence.
- `src/utils/ui_components.py`: Dynamic dark-theme UI cards, legends, CSS injection.
- `models/GeoFusion_AI_Final.pth`: Final trained PyTorch model checkpoint.
- `data/patches/sentinel1/*.npy`: 179 Sentinel-1 SAR patches.
- `data/patches/sentinel2/*.npy`: 179 Sentinel-2 Optical patches.
- `data/patches/labels/*.npy`: 179 ESA WorldCover ground truth label patches.
- `data/patches/patch_metadata.csv`: Patch coordinate and valid ratio index.
- `data/splits/train.csv`, `val.csv`, `test.csv`: Isolated spatial splits (125 train, 25 val, 29 test).
- `data/class_mapping.json`: 8-Class ESA WorldCover taxonomy definition.
- `data/users.json`: Authentication user database.
- `notebooks/fusion_training_colab.ipynb`: Google Colab training notebook.
- `colab_train_fusion.py`: Colab standalone training script.
- `requirements.txt`: Python package requirements.
- `README.md`: Comprehensive project documentation.
- `PROJECT_AUDIT_REPORT.md`: Audit log and scientific validation report.
- `PROJECT_CLEANUP_PLAN.md`: Cleanup plan document.
- `.gitignore`: Git exclusion patterns.

### 2. Files to MOVE / RELOCATE / RENAME
- `models/GeoFusion_AI_Final.pth` -> Renamed to `models/GeoFusion_AI_Final.pth` (Primary production checkpoint).
- Primary app code consolidated into `app/app.py`, with `app.py` in root acting as forwarding entry point.
- `app/static/css/style.css`: Extracted standalone custom CSS stylesheet.
- `app/static/js/`: JavaScript asset directory.
- `app/static/images/`: Static image assets directory.
- `app/templates/`: HTML/UI template directory.
- Relocate final evaluation artifacts to `outputs/final_results/`:
  - `outputs/evaluation/confusion_matrix.png` -> `outputs/final_results/confusion_matrix.png`
  - `outputs/evaluation/per_class_metrics.csv` -> `outputs/final_results/per_class_metrics.csv`
  - `outputs/graphs/model_comparison.png` -> `outputs/final_results/experiment_comparison.png`
  - `outputs/reports/final_metrics.json` -> `outputs/final_results/final_metrics.json`
  - `outputs/final_results/final_project_results.md`: Summary report of evaluation results.

### 3. Files to MERGE / CONSOLIDATE
- Tests merged & organized into clean test modules:
  - `tests/test_model.py`: Model forward pass & architecture tests.
  - `tests/test_inference.py`: Inference, shape, entropy uncertainty, and error map tests.
  - `tests/test_auth.py`: Authentication, registration, password hashing, lockout tests.
  - `tests/test_platform.py`: Provenance, geospatial area, split isolation, dataset fingerprinting tests.
  - `tests/test_ui_flow.py`: End-to-end headless Streamlit UI flow tests.

### 4. Files to DELETE (Safe Removals)
- `data.zip` (1.02 GB): Duplicate zip archive of `data/` directory.
- `GeoFusion_Dataset.zip` (1.02 GB): Duplicate zip archive of dataset.
- `GeoFusion_Project.zip` (31.7 MB): Redundant zip copy of project.
- `data/raw/labels/worldcover_download.zip` (873 KB): Redundant archive of downloaded label raster.
- `models/GeoFusion_AI_Final.pth` (149 KB): Duplicate copy of final model (SHA-256 identical to `GeoFusion_AI_Final.pth`). Unified to `models/GeoFusion_AI_Final.pth`.
- `models/archive/` (`fusion_best.pth`, `fusion_model_best.pth`): Redundant legacy dry-run checkpoints.
- Temporary standalone scripts at root:
  - `check_data.py`: Redundant debug script (superseded by `src/data/inspect_data.py`).
  - `dataset_summary.py`: Standalone debug snippet.
  - `visualize_dataset.py`: Standalone matplotlib preview script (superseded by app UI).
- Large temporary output image dumps:
  - `outputs/check_data_visualization.png` (16.7 MB debug plot).
  - `outputs/sentinel1_vv_vh.png` (4.6 MB debug export).
  - `outputs/sentinel2_rgb.png` (9.0 MB debug export).

---

## Safety Verification Checklist

- [x] **Model Architecture**: `MultimodalFusionNet` untouched in `src/models/fusion_model.py`.
- [x] **Weights & Checkpoint**: Final trained weights preserved, verified identical SHA-256, saved to `models/GeoFusion_AI_Final.pth`.
- [x] **Dataset & Labels**: All 179 Sentinel-1, Sentinel-2, and label patches preserved. No modification to labels or data values.
- [x] **Authentication & Security**: `src/utils/auth.py` and `data/users.json` fully preserved.
- [x] **Test Verification**: All unit tests run before and after cleanup to ensure zero broken references.
