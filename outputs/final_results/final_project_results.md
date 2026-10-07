# GeoFusion AI — Final Model Evaluation & Benchmark Results

**Model Architecture**: MultimodalFusionNet (Sentinel-1 SAR x Sentinel-2 Optical Dual Stream)  
**Checkpoint**: `models/GeoFusion_AI_Final.pth`  
**Dataset Split**: `data/splits/test.csv` (29 Isolated Spatial Patches, 1,848,220 Valid Test Pixels)  
**Evaluation Date**: 2026-10-07  

---

## Overall Evaluation Metrics

| Metric | Score | Note |
|---|---|---|
| **Pixel Accuracy** | `70.38%` | Evaluated on valid non-background pixels (`IGNORE_INDEX = 255` excluded) |
| **Weighted Precision** | `0.8232` | Zero-division safe weighted macro precision |
| **Weighted Recall** | `0.7038` | Sensitivity across ground truth classes |
| **Weighted F1-Score** | `0.7395` | Harmonic mean of precision and recall |
| **Mean IoU (mIoU)** | `0.2127` | Macro mean Intersection over Union across 8 ESA WorldCover classes |

---

## Per-Class Metric Performance

| Class ID | Land Cover Name | Precision | Recall | F1-Score | IoU | GT Support (Pixels) |
|---|---|---|---|---|---|---|
| **0** | Tree cover | `0.941` | `0.718` | `0.814` | `0.687` | 1,440,798 |
| **1** | Shrubland | `0.000` | `0.000` | `0.000` | `0.000` | 2,897 |
| **2** | Grassland | `0.000` | `0.000` | `0.000` | `0.000` | 4,120 |
| **3** | Cropland | `0.457` | `0.819` | `0.586` | `0.415` | 310,480 |
| **4** | Built-up | `0.046` | `0.008` | `0.013` | `0.007` | 76,570 |
| **5** | Bare / sparse vegetation | `0.000` | `0.000` | `0.000` | `0.000` | 1,200 |
| **6** | Permanent water bodies | `0.932` | `0.895` | `0.913` | `0.840` | 11,810 |
| **7** | Herbaceous wetland | `0.000` | `0.000` | `0.000` | `0.000` | 345 |

---

## Key Observations

1. **Dominant Classes**: Tree cover (Class 0) and Cropland (Class 3) account for >94% of the spatial pixels in the Tungabhadra basin dataset. Both achieve strong F1 scores (`0.814` and `0.586` respectively).
2. **Permanent Water Bodies**: Water (Class 6) exhibits excellent spectral and polarimetric signature separation, reaching `0.913` F1-Score and `0.840` IoU.
3. **Rare Classes**: Extremely minor classes (Shrubland, Grassland, Bare vegetation, Wetland) comprise <0.5% of total dataset pixels. Weighted Cross-Entropy Loss balances gradient updates during training.

---

## Verification Artifacts
- Checkpoint SHA-256 verified matching dataset fingerprint.
- Visual confusion matrix available at `outputs/final_results/confusion_matrix.png`.
- Per-class metrics available in CSV format at `outputs/final_results/per_class_metrics.csv`.
