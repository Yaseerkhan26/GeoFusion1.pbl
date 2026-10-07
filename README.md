# GeoFusion AI — Multimodal Satellite Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Data%20Acquisition-2EA44F?style=for-the-badge&logo=google-earth&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 📌 Project Overview

**GeoFusion AI** is a production-grade Earth-Observation and Land-Cover Segmentation platform that fuses co-registered **Sentinel-1 SAR** (microwave radar) and **Sentinel-2 Optical** (multispectral) satellite imagery for 8-class land cover classification mapped to the ESA WorldCover taxonomy.

### Key Modalities & Fusion Logic
* **Sentinel-1 SAR**: Dual polarizations (`VV`, `VH`) penetrate cloud cover and provide structural and soil moisture backscatter signatures.
* **Sentinel-2 Optical**: 6 multispectral bands (`B02 Blue`, `B03 Green`, `B04 Red`, `B08 NIR`, `B11 SWIR-1`, `B12 SWIR-2`) capture rich vegetation and surface reflectance properties.
* **Multimodal Fusion**: Dual-stream deep neural network (`MultimodalFusionNet`) extracts feature representations independently via custom S1 and S2 encoders before concatenating feature depths (64 channels) for spatial land-cover segmentation with Shannon entropy uncertainty estimation.

---

## 🎯 Target Land-Cover Taxonomy (8 ESA WorldCover Classes)

| Class ID | Land Cover Name | Hex Color | Description |
|---|---|---|---|
| **0** | Tree cover | `#1E5631` | Dense or sparse tree foliage and forest land |
| **1** | Shrubland | `#4C9A2A` | Low woody vegetation and shrubs |
| **2** | Grassland | `#ACD870` | Natural herbaceous vegetation and pasture |
| **3** | Cropland | `#E5B636` | Cultivated agricultural land and seasonal crops |
| **4** | Built-up | `#808080` | Man-made structures, urban infrastructure, roads |
| **5** | Bare / sparse vegetation | `#A0826C` | Unvegetated soil, rocks, and sand |
| **6** | Permanent water bodies | `#0066CC` | Rivers, lakes, reservoirs, and open water |
| **7** | Herbaceous wetland | `#00A896` | Seasonally or permanently flooded vegetation |

---

## 🧠 Model Architecture & Pipeline

```text
               Co-Registered Satellite Inputs
                             │
            ┌────────────────┴────────────────┐
            │                                 │
    Sentinel-1 SAR                    Sentinel-2 Optical
     [Batch, 2, 256, 256]            [Batch, 6, 256, 256]
            │                                 │
            ▼                                 ▼
    Sentinel1Encoder                  Sentinel2Encoder
    (2 -> 16 -> 32)                   (6 -> 16 -> 32)
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                    Concatenated Features
                    [Batch, 64, 256, 256]
                             │
                             ▼
                    Fusion Classifier
                  [Batch, 8, 256, 256]
                             │
                             ▼
            Softmax Probabilities & Uncertainty
```

### Class Imbalance Handling
- Natural ground cover distribution in the Tungabhadra basin study area is heavily dominated by Tree cover and Cropland (>90% of area).
- Gradient update stability during training is achieved using **Weighted Cross-Entropy Loss** calculated on valid non-background pixels (`IGNORE_INDEX = 255` excluded).

---

## 📊 Final Model & Benchmark Performance

- **Final Trained Model**: `models/GeoFusion_AI_Final.pth` (149.8 KB)
- **Evaluation Dataset**: `data/splits/test.csv` (29 isolated test patches, 1,848,220 valid test pixels)

| Metric | Measured Value | Scope / Notes |
|---|---|---|
| **Pixel Accuracy** | `70.38%` | Evaluated on valid non-background test pixels |
| **Weighted Precision** | `0.8232` | Zero-division safe weighted macro precision |
| **Weighted Recall** | `0.7038` | Sensitivity across ground truth classes |
| **Weighted F1-Score** | `0.7395` | Harmonic mean across classes |
| **Mean IoU (mIoU)** | `0.2127` | Macro mean Intersection over Union across 8 classes |

---

## 🚀 Installation & Running the Application

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Yaseerkhan26/GeoFusion1.git
cd "GeoFusion AI"

# Install production dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application

Launch the application directly with either command:

```bash
streamlit run app/app.py
```
*or*
```bash
streamlit run app.py
```

### 3. Run Verification Test Suite

Run all automated unit and integration tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📁 Final Project Structure

```text
GeoFusion AI/
│
├── app/
│   ├── app.py                     # Primary Streamlit Application logic & page router
│   ├── templates/                 # HTML templates
│   └── static/
│       ├── css/
│       │   └── style.css          # Dark satellite theme CSS stylesheet
│       ├── js/                    # JavaScript assets
│       └── images/                # App static images
│
├── src/
│   ├── models/
│   │   ├── fusion_model.py        # MultimodalFusionNet PyTorch architecture
│   │   ├── sentinel1_encoder.py   # Sentinel-1 SAR encoder module
│   │   ├── sentinel2_encoder.py   # Sentinel-2 Optical encoder module
│   │   ├── dataset.py             # MultimodalSatelliteDataset PyTorch loader
│   │   ├── config.py              # Architecture hyperparameters & NUM_CLASSES
│   │   └── inference.py           # Inference, uncertainty & error map functions
│   │
│   ├── training/
│   │   └── trainer.py             # Model trainer class & training loops
│   │
│   ├── evaluation/
│   │   └── evaluate.py            # Evaluation suite & test set metrics calculation
│   │
│   ├── geospatial/
│   │   └── geo_utils.py           # Georeferencing, raster IO, geodesic area calc
│   │
│   └── utils/
│       ├── auth.py                # User authentication, hashing & session manager
│       ├── config.py              # Centralized paths, color palettes & constants
│       ├── provenance.py          # SHA-256 fingerprinting & evaluation loader
│       └── ui_components.py       # Custom dark UI cards, legends & components
│
├── models/
│   └── GeoFusion_AI_Final.pth     # Authoritative trained PyTorch model weights
│
├── data/
│   ├── patches/
│   │   ├── sentinel1/             # 179 Sentinel-1 SAR patch arrays (.npy)
│   │   ├── sentinel2/             # 179 Sentinel-2 Optical patch arrays (.npy)
│   │   └── labels/                # 179 ESA WorldCover ground truth patches (.npy)
│   ├── splits/
│   │   ├── train.csv              # 125 Training patches
│   │   ├── val.csv                # 25 Validation patches
│   │   └── test.csv               # 29 Test patches (strictly isolated)
│   ├── class_mapping.json         # 8-Class ESA taxonomy mapping
│   └── users.json                 # Secure user authentication database
│
├── outputs/
│   ├── final_results/             # Confusion matrix, per-class metrics & summary
│   └── training/                  # Loss/accuracy curves & training history
│
├── tests/                         # Complete automated unit test suite
│   ├── test_model.py
│   ├── test_inference.py
│   ├── test_auth.py
│   ├── test_platform.py
│   └── test_ui_flow.py
│
├── notebooks/
│   └── fusion_training_colab.ipynb# Colab GPU training notebook
│
├── requirements.txt               # Dependency specifications
├── README.md                      # Project documentation
├── PROJECT_AUDIT_REPORT.md        # Scientific audit & verification report
├── PROJECT_CLEANUP_PLAN.md        # Detailed project cleanup plan
├── .gitignore                     # Git tracking exclusions
├── app.py                         # Authoritative pass-through root entry point
└── colab_train_fusion.py          # Google Colab GPU training script
```

---

## 🚀 DEPLOYMENT ON RENDER

GeoFusion AI is fully configured for production web deployment on [Render](https://render.com) as a Streamlit Web Service.

### 1. GitHub Repository Requirements
- Push all project source files (`app/`, `src/`, `models/`, `data/class_mapping.json`, `data/users.json`, `requirements.txt`, `.python-version`, `render.yaml`) to your GitHub repository.
- Ensure `models/GeoFusion_AI_Final.pth` (149.8 KB) is committed to the repository.

### 2. Render Service Type
- **Service Type**: Web Service
- **Environment / Runtime**: Python 3
- **Region**: Any (e.g., Oregon, USA / Frankfurt, Germany)
- **Plan**: Free or Starter (>= 512 MB RAM recommended)

### 3. Build Command
```bash
pip install -r requirements.txt
```

### 4. Start Command
```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port $PORT
```

### 5. Environment Variables
Configure the following in the Render Dashboard (**Environment** section):
| Variable Name | Required | Default / Description |
|---|---|---|
| `PORT` | Auto-provided | Set dynamically by Render |
| `PYTHON_VERSION` | Yes | `3.11.9` |
| `GEOFUSION_SECRET_KEY` | Recommended | Random secret string for session security |
| `GEOFUSION_ADMIN_EMAIL` | Optional | `admin@geofusion.ai` |
| `GEOFUSION_ADMIN_PASSWORD` | Optional | Custom admin password override |

### 6. Python Version
- **Version**: `3.11.9` (Specified via `.python-version` and `render.yaml`).

### 7. Model Location & Architecture
- **Model Checkpoint**: `models/GeoFusion_AI_Final.pth` (149.8 KB)
- **Architecture**: `MultimodalFusionNet(num_classes=8)`
- Fused dual-stream SAR (Sentinel-1) and Optical (Sentinel-2) neural network.

### 8. How to Deploy Step-by-Step
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository containing **GeoFusion AI**.
4. Render will automatically detect `render.yaml` or fill manually:
   - **Name**: `geofusion-ai`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/app.py --server.address 0.0.0.0 --server.port $PORT`
5. Click **Create Web Service**.

### 9. How to Redeploy After GitHub Changes
- **Automatic Redeploy**: Any commit pushed to the `main` branch triggers an automatic rebuild and zero-downtime redeploy on Render.
- **Manual Redeploy**: Click **Manual Deploy** -> **Deploy latest commit** inside the Render Dashboard.

### 10. Troubleshooting
- **Model Loading Error**: Ensure `models/GeoFusion_AI_Final.pth` exists in the repository.
- **Port Binding Failure**: Ensure the start command uses `$PORT` and `--server.address 0.0.0.0`.
- **Memory Limit / Out-of-Memory**: Streamlit resource caching (`@st.cache_resource`) prevents reloading the model across reruns.
- **Login Credentials**: Initial accounts can be registered via the UI or configured using `GEOFUSION_ADMIN_EMAIL` and `GEOFUSION_ADMIN_PASSWORD` environment variables.

---

## 🎓 Academic & Project Metadata

* **Project Title**: Multimodal Satellite Data Fusion for Enhanced Land Cover Classification Using Sentinel-1 and Sentinel-2
* **Domain**: Remote Sensing · Geospatial Data Science · Deep Learning · Computer Vision
* **Author**: Yaseer Khan
* **License**: Academic & Educational Open Source Research

