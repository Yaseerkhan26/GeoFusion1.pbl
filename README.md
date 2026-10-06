# Multimodal Satellite Data Fusion for Enhanced Land Cover Classification Using Sentinel-1 and Sentinel-2

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Data%20Acquisition-2EA44F?style=for-the-badge\&logo=google-earth\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)

## 📌 Project Overview

**GeoFusion AI** is a final-year engineering project focused on **multimodal satellite data fusion for enhanced land cover classification**.

The project combines:

* **Sentinel-1 SAR** data using VV and VH polarizations
* **Sentinel-2 multispectral optical** data
* Deep learning-based feature extraction
* Multimodal fusion for semantic segmentation
* Google Earth Engine for satellite data acquisition
* PyTorch for model development and training

Sentinel-2 provides rich spectral information for identifying vegetation, water bodies, agricultural areas, and urban structures. However, optical imagery can be affected by cloud cover, haze, and illumination conditions.

Sentinel-1 Synthetic Aperture Radar (SAR) can operate under cloudy and rainy conditions and provides complementary information related to surface structure and moisture.

GeoFusion AI combines these complementary modalities to improve land-cover classification.

---

## 🎯 Project Objectives

1. Acquire Sentinel-1 and Sentinel-2 satellite imagery.
2. Preprocess and align multimodal satellite data.
3. Generate spatially aligned training patches.
4. Prepare land-cover labels.
5. Extract features independently from Sentinel-1 and Sentinel-2.
6. Fuse the extracted multimodal features.
7. Train a semantic segmentation model.
8. Evaluate the classification performance using standard metrics.
9. Prepare outputs for visualization and analysis.

---

## 🛰️ Satellite Data

### Sentinel-1

The Sentinel-1 branch uses:

* **VV polarization**
* **VH polarization**

Sentinel-1 provides SAR information that is useful even when optical imagery is affected by clouds.

### Sentinel-2

The Sentinel-2 branch uses multispectral optical information, including the project's selected 10/20 m bands.

Sentinel-2 provides spectral information useful for distinguishing different land-cover classes.

---

## 🧠 Model Architecture

The project uses a multimodal deep-learning architecture consisting of separate feature extraction branches for Sentinel-1 and Sentinel-2 followed by feature fusion and segmentation.

```text
                 Satellite Data
                       │
          ┌────────────┴────────────┐
          │                         │
     Sentinel-1                 Sentinel-2
       SAR Data                 Optical Data
      VV + VH                    Multispectral
          │                         │
          ▼                         ▼
   Sentinel-1 Encoder       Sentinel-2 Encoder
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                Feature Fusion
                       │
                       ▼
              Fusion Segmentation
                    Model
                       │
                       ▼
              Land Cover Map
```

The existing fusion architecture is implemented in:

```text
src/models/fusion_model.py
```

---

## 🛠️ Technology Stack

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| Python 3.10+        | Core programming           |
| PyTorch             | Deep learning              |
| Torchvision         | Computer vision utilities  |
| Google Earth Engine | Satellite data acquisition |
| Rasterio            | Raster processing          |
| GeoPandas           | Geospatial processing      |
| Shapely             | Geometry operations        |
| NumPy               | Numerical processing       |
| SciPy               | Scientific computing       |
| Pandas              | Data processing            |
| Scikit-learn        | Evaluation metrics         |
| Matplotlib          | Visualization              |
| Streamlit           | Web application            |
| Folium              | Interactive maps           |

---

## 📂 Project Structure

```text
GeoFusion AI/
│
├── data/
│   ├── raw/
│   │   ├── sentinel1/
│   │   ├── sentinel2/
│   │   └── labels/
│   │
│   ├── processed/
│   └── splits/
│
├── src/
│   │
│   ├── data/
│   │   ├── create_patches.py
│   │   ├── create_splits.py
│   │   ├── download_worldcover.py
│   │   ├── gee_data_fusion.js
│   │   ├── inspect_data.py
│   │   ├── prepare_labels.py
│   │   ├── preprocess.py
│   │   ├── validate_labels.py
│   │   ├── validate_patches.py
│   │   └── validate_preprocessing.py
│   │
│   ├── models/
│   │   ├── config.py
│   │   ├── create_class_mapping.py
│   │   ├── dataset.py
│   │   ├── fusion_model.py
│   │   ├── sentinel1_encoder.py
│   │   ├── sentinel1_model.py
│   │   ├── sentinel2_encoder.py
│   │   ├── sentinel2_model.py
│   │   ├── test_model.py
│   │   └── validate_model.py
│   │
│   ├── training/
│   │   ├── check_stage6_completion.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── trainer.py
│   │   └── validate_training.py
│   │
│   └── utils/
│       └── config.py
│
├── check_data.py
├── colab_train_fusion.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📄 Important Files

| File                                | Purpose                            |
| ----------------------------------- | ---------------------------------- |
| `src/data/preprocess.py`            | Satellite data preprocessing       |
| `src/data/create_patches.py`        | Creates spatial training patches   |
| `src/data/create_splits.py`         | Creates dataset splits             |
| `src/data/prepare_labels.py`        | Prepares land-cover labels         |
| `src/models/sentinel1_encoder.py`   | Sentinel-1 feature extraction      |
| `src/models/sentinel2_encoder.py`   | Sentinel-2 feature extraction      |
| `src/models/sentinel1_model.py`     | Sentinel-1 model                   |
| `src/models/sentinel2_model.py`     | Sentinel-2 model                   |
| `src/models/fusion_model.py`        | Multimodal fusion architecture     |
| `src/training/train.py`             | Training pipeline                  |
| `src/training/trainer.py`           | Training utilities                 |
| `src/training/validate_training.py` | Training validation                |
| `colab_train_fusion.py`             | Google Colab Fusion model training |
| `check_data.py`                     | Dataset checking and validation    |

---

## 🔄 Project Pipeline

```text
Sentinel-1
   │
   ├── VV
   └── VH
        │
        ▼
   SAR Preprocessing
        │
        ▼
   Sentinel-1 Features
        │
        │
        ├──────────────┐
        │              │
        │              ▼
        │         Feature Fusion
        │              ▲
        │              │
        │        Sentinel-2 Features
        │              ▲
        │              │
        ▼              │
   Sentinel-1      Sentinel-2
                    │
                    ▼
              Optical Processing
                    │
                    ▼
             Multispectral Data

                    │
                    ▼
             Fusion Network
                    │
                    ▼
          Land Cover Classification
                    │
                    ▼
             Prediction Map
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yaseerkhan26/GeoFusion1.pbl.git
cd GeoFusion1.pbl
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🛰️ Google Earth Engine

Google Earth Engine can be used for satellite data acquisition.

Authenticate using:

```bash
earthengine authenticate
```

The project includes:

```text
src/data/gee_data_fusion.js
```

for Google Earth Engine data processing/acquisition workflows.

---

## 🧪 Data Processing

The general processing workflow is:

```text
Satellite Data
      │
      ▼
Data Validation
      │
      ▼
Preprocessing
      │
      ▼
Label Preparation
      │
      ▼
Patch Generation
      │
      ▼
Dataset Splitting
      │
      ▼
Model Training
```

---

## 🧠 Stage 6 — Fusion Model Training

The final fusion model is intended to be trained using **Google Colab GPU** rather than relying on a low-resource local machine.

The repository contains:

```text
colab_train_fusion.py
```

which provides the training workflow for the final multimodal fusion model.

The training process uses the prepared Sentinel-1 and Sentinel-2 datasets and the existing fusion architecture.

---

## 📊 Evaluation

The project can evaluate land-cover classification using metrics such as:

* Overall Accuracy
* Cohen's Kappa
* Intersection over Union (IoU)
* Confusion Matrix

Evaluation and validation utilities are included in:

```text
src/training/
src/models/
```

---

## 🚀 FusionLand AI — Production Frontend Platform

The project includes the **FusionLand AI** satellite intelligence platform, providing a dark-theme Earth-observation dashboard, real-time georeferencing, uncertainty quantification, and strict evaluation provenance.

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Yaseerkhan26/GeoFusion1.git
cd "GeoFusion AI"

# Install production dependencies
pip install -r requirements.txt
```

### 2. Official Frontend Entry Point

Launch the interactive web application directly with:

```bash
streamlit run app.py
```

*(Alternatively: `python -m streamlit run app.py`)*

### 3. Canonical 10 Platform Modules

1. **01 Overview**: Command-center hero dashboard with verified study area bounding coordinates (`EPSG:4326`), natural-color satellite thumbnail, and operational telemetry cards.
2. **02 Satellite Explorer**: Native resolution (`4112 × 4008`) multi-band raster viewer supporting Sentinel-2 RGB, False Color (NIR), SWIR, individual bands, and Sentinel-1 SAR (VV, VH, VV/VH ratio) with interactive percentile stretch controls.
3. **03 AI Classification**: Synchronized 256×256 patch inference workspace displaying input modalities, argmax classification map, maximum class confidence (softmax probability), normalized Shannon entropy uncertainty map, uncertainty tier breakdowns, confidence filtering threshold, and side-by-side ground truth agreement analysis.
4. **04 GIS Map**: Interactive Folium map with Esri Satellite basemap, bounding box polygon, native raster overlay, and an **Affine Pixel Inspector** that converts row/col to real geographic Lat/Lon coordinates and samples band reflectance without hardcoding.
5. **05 Analytics**: Empirical ground truth class distribution across splits, prediction vs ground truth bar charts, and WGS-84 geodesic area calculations.
6. **06 Model Performance**: Provenance-protected benchmarking center displaying validated Pixel Accuracy, Weighted Precision, Weighted Recall, Weighted F1, and mIoU. Features per-class IoU charts, confusion matrix (raw/normalized), 20-epoch Colab training curves, and dynamic Model Cards.
7. **07 Data Quality**: Automated system health matrix with PASS/WARNING/FAIL badges verifying Python/PyTorch/CUDA runtime, GeoTIFF headers, and zero patch ID leakage across dataset splits.
8. **08 Experiments**: Discovered checkpoint registry tracking SHA-256 hashes, parameter shapes, and on-demand test set benchmark execution.
9. **09 Reports**: Multi-format scientific report generation (`.txt`, `.md`, `.json`, `.csv`) and georeferenced GeoTIFF prediction export with strict CRS and affine transform preservation.
10. **10 About**: Remote sensing theoretical rationale (Sentinel-1 C-band microwave vs Sentinel-2 optical), interactive SVG dual-encoder pipeline flow, and real-world sensor limitations.

### 4. Verification & Testing

Execute the automated integration test suite:

```bash
python -m unittest tests/test_platform.py
python -m unittest tests/test_ui_flow.py
```

---


## 📌 Current Project Status

The project repository contains the data-processing, model, validation, and training components required for the GeoFusion multimodal satellite-data-fusion pipeline.

The final Fusion model training workflow is available through:

```text
colab_train_fusion.py
```

---

## 🎓 Academic Project

**Project Title:**
Multimodal Satellite Data Fusion for Enhanced Land Cover Classification Using Sentinel-1 and Sentinel-2

**Domain:**
Remote Sensing · Geospatial Data Science · Deep Learning · Computer Vision

**Project Type:**
Final-Year Engineering Project

---

## 👨‍💻 Author

**Yaseer Khan**

---

## 📜 License

This project is developed for academic and educational purposes.
