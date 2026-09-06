# 🛰️ Multimodal Satellite Data Fusion for Enhanced Land Cover Classification Using Sentinel-1 and Sentinel-2

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Earth Engine](https://img.shields.io/badge/GEE-Data_Acquisition-2EA44F?style=for-the-badge&logo=google-earth&logoColor=white)](https://earthengine.google.com/)

Final-Year Engineering Project for automated land cover and land use (LCLU) semantic segmentation. Combines all-weather Synthetic Aperture Radar (SAR) from **Sentinel-1** (VV & VH polarizations) with multispectral optical imagery from **Sentinel-2** (10m & 20m bands) using deep learning fusion architectures.

---

## 📌 Project Overview

Optical satellite sensors (such as Sentinel-2) provide rich spectral information for distinguishing vegetation, water bodies, and urban structures. However, optical imagery suffers from severe cloud cover, haze, and lighting variations. Conversely, Sentinel-1 Synthetic Aperture Radar (SAR) operates at C-band microwave frequencies, penetrating clouds and precipitation to capture surface geometry and moisture content.

This project implements an **early-to-intermediate multimodal fusion deep neural network** in PyTorch to jointly model SAR backscatter and optical spectral signatures, delivering superior land cover classification accuracy across all weather conditions.

---

## 🛠️ Technology Stack

- **Core Programming**: Python 3.10+
- **Deep Learning Framework**: PyTorch, Torchvision
- **Satellite Data API**: Google Earth Engine (`earthengine-api`)
- **Geospatial & Raster I/O**: Rasterio, GeoPandas, Shapely, PyPROJ
- **Numerical Processing**: NumPy, SciPy, Pandas
- **Machine Learning & Metrics**: Scikit-learn (Overall Accuracy, Cohen's Kappa, Confusion Matrix)
- **Web Application & UI**: Streamlit, Folium (`streamlit-folium`)
- **Data Visualization**: Matplotlib, Seaborn

---

## 📂 Directory Structure

```text
GeoFusion AI/
│
├── data/                         # Local Data Store (Ignored by Git except .gitkeep)
│   ├── raw/
│   │   ├── sentinel1/            # Raw Sentinel-1 SAR GeoTIFF files (VV, VH)
│   │   ├── sentinel2/            # Raw Sentinel-2 Optical GeoTIFF files (10 bands)
│   │   └── labels/               # Ground Truth Land Cover Label GeoTIFFs
│   ├── processed/                # Co-registered & resampled stacked GeoTIFF rasters
│   └── patches/                  # Extracted 128x128 spatial patch triplets (.npz / .npy)
│
├── notebooks/                    # Jupyter notebooks for exploratory data analysis (EDA)
│
├── src/                          # Modular Source Code Package
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py      # Google Earth Engine data query & export script
│   │   ├── preprocess.py        # Speckle filtering, BOA scaling, & grid alignment
│   │   └── create_patches.py     # Sliding window spatial patch extraction algorithm
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sentinel1_model.py    # PyTorch CNN encoder branch for SAR (2 channels)
│   │   ├── sentinel2_model.py    # PyTorch CNN encoder branch for Optical (10 channels)
│   │   └── fusion_model.py       # Multimodal Fusion Network & segmentation decoder
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py              # PyTorch model training pipeline & state saver
│   │   └── evaluate.py           # Metrics calculation (OA, Kappa, IoU) & report exporter
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Global project configuration, band lists, & hyperparams
│
├── app/                          # Interactive Streamlit Web Application
│   ├── pages/                    # Multi-page navigation tabs
│   └── app.py                    # Main web dashboard with Folium map rendering
│
├── models/                       # Saved trained model weight checkpoints (.pth)
│
├── outputs/                      # Generated Outputs & Deliverables
│   ├── maps/                     # Rendered GeoTIFF land cover prediction maps
│   ├── graphs/                   # Exported evaluation plots & confusion matrices
│   └── reports/                  # Metrics summary text and JSON reports
│
├── .gitignore                    # Version control ignore rules for large rasters & weights
├── requirements.txt              # Project dependency list
└── README.md                     # Project documentation
```

---

## 📄 File Purpose Directory Reference

| File Path | Description / Purpose |
| :--- | :--- |
| [`src/utils/config.py`](file:///d:/GeoFusion%20AI/src/utils/config.py) | Centralized configuration for directory paths, band definitions, land cover taxonomy, colors, and training parameters. |
| [`src/data/download_data.py`](file:///d:/GeoFusion%20AI/src/data/download_data.py) | Google Earth Engine script for querying and exporting raw Sentinel-1 and Sentinel-2 imagery. |
| [`src/data/preprocess.py`](file:///d:/GeoFusion%20AI/src/data/preprocess.py) | Preprocessing routines for SAR speckle filtering, optical reflectance scaling, and spatial band stacking. |
| [`src/data/create_patches.py`](file:///d:/GeoFusion%20AI/src/data/create_patches.py) | Spatial window tiling script to slice rasters into uniform 128x128 patches for deep learning. |
| [`src/models/sentinel1_model.py`](file:///d:/GeoFusion%20AI/src/models/sentinel1_model.py) | PyTorch encoder module for Sentinel-1 dual-polarization SAR imagery (VV/VH). |
| [`src/models/sentinel2_model.py`](file:///d:/GeoFusion%20AI/src/models/sentinel2_model.py) | PyTorch encoder module for Sentinel-2 10-band multispectral optical imagery. |
| [`src/models/fusion_model.py`](file:///d:/GeoFusion%20AI/src/models/fusion_model.py) | Multimodal fusion architecture combining SAR and Optical feature maps for semantic segmentation. |
| [`src/training/train.py`](file:///d:/GeoFusion%20AI/src/training/train.py) | Model training loop with loss function, optimizer step, validation logging, and model saving. |
| [`src/training/evaluate.py`](file:///d:/GeoFusion%20AI/src/training/evaluate.py) | Quantitative accuracy assessment script (Overall Accuracy, Cohen's Kappa, IoU, Confusion Matrix). |
| [`app/app.py`](file:///d:/GeoFusion%20AI/app/app.py) | Streamlit dashboard providing interactive Folium map visualization and prediction analytics. |

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
# Create a Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Google Earth Engine Authentication

If downloading data via GEE:

```bash
earthengine authenticate
```

### 3. Data Processing & Pipeline Execution

Run the starter modular pipeline scripts in sequence:

```bash
# Step 1: Download raw satellite data
python src/data/download_data.py

# Step 2: Preprocess and align rasters
python src/data/preprocess.py

# Step 3: Extract spatial patches for training
python src/data/create_patches.py

# Step 4: Train Multimodal Fusion Network
python src/training/train.py

# Step 5: Evaluate model performance & generate reports
python src/training/evaluate.py
```

### 4. Launch Interactive Streamlit Dashboard

Run the Streamlit web application:

```bash
streamlit run app/app.py
```

---

## 🎓 Academic Project Context

- **Project Title**: Multimodal Satellite Data Fusion for Enhanced Land Cover Classification Using Sentinel-1 and Sentinel-2
- **Domain**: Remote Sensing, Geospatial Data Science, Deep Learning, Computer Vision
- **Author**: Final Year Engineering Student
