# Aerobridge — Location-aware PM2.5 Prediction Pipeline (APE 3.0)

## 📌 Overview
**Aerobridge** is a location-aware PM2.5 prediction pipeline that bridges **INSAT satellite imagery** with **CPCB ground-based air quality measurements**.  
It processes, aligns, and enriches multi-year, multi-station datasets, producing machine-learning-ready features for advanced air pollution modeling.

Originally developed for the **Bhartiya Antariksha Hackathon 2025** problem statement from ISRO:  
> “Monitoring Air Pollution from Space using Satellite Observations, Ground-Based Measurements, Reanalysis Data, and AI/ML Techniques.”

---

## 🌍 Project Scope
While a full-scale, all-India implementation would require supercomputing resources (100+ TB imagery), Aerobridge strategically scaled to a **multi-state, multi-station** testbed while retaining the **core technical challenges** of:
- Processing large satellite imagery datasets.
- Cleaning messy, incomplete pollution data.
- Building a robust, automated geospatial data pipeline.

**Current scope:**
- 4 States: Maharashtra, Delhi, Karnataka, Haryana
- 5 Stations per state (20 total)
- Period: 1 Sept – 30 Nov for both 2021 & 2022
- Time: 08:00–16:00 IST (hourly alignment)

---

## 📊 Data Summary
### CPCB Dataset
- Sources: ~450 stations across India (filtered to 20 selected stations for pipeline test)
- Period: 91 days × 9 hours/day = 728 hours/station
- Total: 14,560 hourly PM2.5 measurements
- Preprocessing:
  - Timestamp cleaning & standardization
  - Missing/invalid PM2.5 removal
  - Date/time range filtering

### INSAT Dataset
- Source: ISRO’s MOSDAC platform
- Bands: **TIR1** (Thermal Infrared) & **WV** (Water Vapour)
- Time interval: Half-hourly (two bands per timestamp)
- 2021: ~3,580 images  
- 2022: ~3,230 images  
- Storage: Raw `.tif` format; cropped subsets per station

---

## 🔄 Pipeline Architecture
### **Stage 1: Image Cropping & Merging**
- Crops raw INSAT `.tif` images to a bounding box around each CPCB station.
- Matches each cropped image to nearest CPCB PM2.5 reading (UTC-based).
- Produces **merged CPCB + INSAT dataset**.

### **Stage 2: Feature Extraction & Dataset Creation**
- Extracts statistics from each cropped image:
  - Mean, std, min, max, median, percentiles
  - Pixel threshold counts, skewness, etc.
- Joins satellite-derived features with CPCB PM2.5 readings.
- Produces **station-specific, ML-ready datasets**.

---

## 🖥 Key Achievement — Internal Monitoring App
A lightweight internal application was built to:
- Track pipeline execution in real-time.
- Inspect intermediate outputs before committing long runs.
- Quickly visualize CPCB + INSAT integrations.
- Greatly reduce debugging time during multi-hour batch runs.

---

## 📈 Scale-Up Results
- Full 2021 & 2022 processing completed.
- Pipeline executed successfully after 3 full runs due to earlier errors.
- **Final run stats:**
  - 20 CPCB stations
  - 1,039,390 cropped INSAT images
  - 115,660 merged records
  - 228,973 extracted features
  - Total run time: **5 hours, 26 minutes, 57 seconds**

---

## 🔮 Future Work & Roadmap
- **Atmospheric Reanalysis Data Integration**
  - MERRA-2 variables: wind speed, temperature, humidity, pressure.
- **Advanced ML Models**
  - LightGBM / XGBoost for PM2.5 prediction from satellite-only features.
  - CNN-based spatial feature extraction.
- **End-to-End Prediction**
  - Real-time satellite ingestion.
  - PM2.5 prediction for locations without ground monitoring stations.

---

## 📂 Repository Structure
Aerobridge/  (folder names might have been changed)
├── data/ # CPCB datasets & processed outputs
├── raw/ # INSAT raw images (sample subset in repo)
├── scripts/ # All pipeline scripts
├── notebooks/ # EDA & modeling notebooks
├── docs/ # Documentation
│ ├── PIPELINE_USAGE_GUIDE.md
│ ├── PIPELINE_ANALYSIS_REPORT.md
│ └── AeroBridge_Documentation.pdf
├── README.md
└── requirements.txt

---

## ⚙️ Tech Stack
- **Languages**: Python 3.x
- **Libraries**: Pandas, NumPy, Rasterio, LightGBM, XGBoost, SHAP
- **Data Sources**: CPCB, ISRO MOSDAC
- **Tools**: Git, Jupyter

---

## 🚀 How to Run
1. Place CPCB datasets in `data/` and INSAT `.tif` files in `raw/`.
2. Follow `docs/PIPELINE_USAGE_GUIDE.md` for setup & processing steps.
3. Use provided scripts in `scripts/` to run cropping, merging, and feature extraction.
4. Model training examples available in `notebooks/`.

---

## 👤 Author
**Lalit K Hire**  
Email: LalitHire110@gmail.com

---
