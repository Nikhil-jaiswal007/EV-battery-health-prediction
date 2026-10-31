# ⚡ EV Battery Health Prediction Dashboard & ML Backend

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0+-green.svg)](https://www.djangoproject.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest%20%7C%20MLP-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning web application for **Electric Vehicle (EV) Battery State of Health (SoH) Prediction, Remaining Useful Life (RUL) estimation, and Asset Pack Valuation**. Benchmarked against the **NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Dataset** dynamics and degradation electrochemistryy.

---

## 📌 Problem Statement

Battery degradation directly affects the performance, range, reliability, and safety of electric vehicles. This project delivers production-grade machine learning models to estimate battery State of Health (SoH), predict remaining cycle lifespan, and assess battery pack valuation from real-time and simulated operational telemetry.

---

## 🌟 Key Features

- **Interactive Telemetry Dashboard**: Responsive dark-mode UI built with Tailwind CSS, Chart.js, and FontAwesome icons.
- **Physics-Informed ML Models**:
  - **XGBoost Regressor** ($R^2 = 0.9957$, RMSE = $0.997\%$) - Optimal non-linear degradation estimation.
  - **Random Forest Regressor** ($R^2 = 0.9948$, RMSE = $1.100\%$) - Robust ensemble bagging against telemetry noise.
  - **Neural Network (MLP)** ($R^2 = 0.9973$, RMSE = $0.798\%$) - Deep continuous electrochemical state mapping.
- **Electrochemical Aging Stresses**:
  - Arrhenius thermal penalties for high temperatures ($> 25^\circ\text{C}$) and low temperatures ($< 15^\circ\text{C}$).
  - C-Rate discharge load polarization.
  - DC Fast-charging mechanical stress & accelerated SEI layer growth.
  - Under-voltage / over-voltage degradation warnings.
- **Live Diagnostics**:
  - State of Health (SoH %) gauge and condition badges (`Healthy`, `Moderate`, `Degraded`).
  - Remaining Useful Life (RUL) cycles to the 80% End-of-Life (EOL) threshold.
  - Pack Financial Valuation based on standard \$120/kWh automotive benchmark.
  - Dynamic Degradation Curves and Asset Depreciation Projections.
- **Dual Execution Options**:
  - Run via full-featured **Django Backend**: `python manage.py runserver`
  - Run via standalone single-file **`app.py`**: `python app.py`

---

## 📁 Repository Structure

```
├── app.py                          # Complete standalone all-in-one web app
├── manage.py                       # Django management script
├── requirements.txt                # Python package dependencies
├── test_app.py                     # Integration & API automated test suite
├── CROSS-VALIDATION.ipynb          # Cross-validation exploration notebook
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI workflow
├── ev_battery_health/              # Django core configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── predictor/                      # Battery prediction app
    ├── urls.py
    ├── views.py
    ├── templates/
    │   └── predictor/
    │       └── index.html          # Interactive Dashboard UI
    └── ml/
        ├── dataset.py              # NASA PCoE battery aging simulation generator
        ├── train.py                # ML model training script
        ├── model_service.py        # Model loader, inference engine & RUL estimator
        └── saved_models/           # Serialized models (.joblib) & metrics
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Nikhil-jaiswal007/EV-battery-health-prediction.git
cd EV-battery-health-prediction
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application

#### Option A: Full Django Server (Recommended)
```bash
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

#### Option B: Standalone All-In-One Application
```bash
python app.py
```

Then visit **[http://localhost:8000/](http://localhost:8000/)** in your browser.

---

## 🧪 Running Tests

```bash
python test_app.py
```

---

## 📡 REST API Reference

### `POST /api/predict/`
Evaluates telemetry features and returns predicted SoH and health metrics.

**Payload:**
```json
{
  "cycle": 20,
  "soc": 80,
  "voltage": 17.50,
  "current": 1.00,
  "temperature": 20.0,
  "capacity": 9.50,
  "model": "xgboost",
  "fast_charging": false
}
```

**Response:**
```json
{
  "soh": 98.67,
  "remaining_cycles": 287,
  "pack_value": 5262,
  "curve": [99.27, 93.95, 87.3, 81.2, 74.5, 69.12, 62.82],
  "model_used": "xgboost",
  "status": "Healthy"
}
```

---

## 📜 License
Distributed under the MIT License.
