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

## 📦 Dataset and Large Files

Large satellite datasets and generated model files are intentionally excluded from the Git repository.

Examples include:

```text
*.tif
*.tiff
*.jp2
*.h5
*.nc
*.pth
*.pt
```

These files should be stored separately and are not required to be committed to GitHub.

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
