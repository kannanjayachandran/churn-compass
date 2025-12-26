![Cover image](./assets/Cover_image.png)

<h1 align=center> 🧭 Churn Compass </h1>

> **Production-Grade Customer Churn Prediction Platform for Retail Banking**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
![Status](https://img.shields.io/badge/Status-Active-success)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Overview

Churn Compass predicts **which customers are most likely to churn**, enabling targeted retention campaigns and revenue protection.

### 🔑 Key Capabilities

| Feature | Description |
|--------|------------|
| **XGBoost Model**             | Tuned and optimized for tabular classification       |
| **Optuna Search**             | Automated hyperparameter optimization                |
| **MLflow Registry**           | Model registration, lineage, and experiment tracking |
| **Prefect Workflows**         | Ingestion, training, monitoring, retraining          |
| **FastAPI Serving**           | Real-time scoring + batch scoring APIs               |
| **Evidently**                 | Drift detection, monitoring reports                  |
| **React UI**                  | Dashboard for predictions, SHAP, drift               |
| **DuckDB + Postgres**         | Local development + production RDBMS                 |
| **Synthetic Test Data (SDV)** | Demo and testing without real PII                    |

---

## 📁 Project Structure

```

churn-compass/
├── assets/             # Project assets and images
├── data/
│   ├── raw/            # Ingested CSV / external data
│   ├── interim/        # Preprocessed artifacts
│   ├── processed/      # Feature-ready datasets
│   └── scored/         # Model prediction outputs
│
├── src/churn_compass/
│   ├── api/            # FastAPI app + routers
│   ├── config/         # Settings + environment management
│   ├── features/       # Feature engineering and transformations
│   ├── io/             # File + DB layers (Parquet, CSV, DuckDB, Postgres)
│   ├── logging/        # Structured logging (JSON + rotation)
│   ├── modeling/       # Training, optimization, evaluation
│   ├── monitoring/     # Drift detection + reports
│   ├── orchestration/  # Prefect workflow management
│   ├── pipelines/      # Ingestion and training flows
│   ├── serving/        # Batch scoring and prediction utilities
│   └── validation/     # Pandera schemas + business rule checks
│
├── frontend/           # UI (React + Vite)
├── notebooks/          # Exploration and analysis notebooks
├── scripts/            # Helper scripts (e.g., sample data generator)
├── tests/              # Comprehensive test suite (unit + integration)
├── docker/             # Container deployment configurations
├── docs/               # Project documentation
├── logs/               # Application logs
├── mlruns/             # MLflow experiment tracking store
├── pyproject.toml      # Build system and dependencies (uv)
├── .env                # Local environment variables
├── README.md           # Project documentation
└── setup.sh            # Initialization script

```

---

## ⚙️ Installation & Setup

### **Automated Setup (Recommended)**

```bash
chmod +x setup.sh
./setup.sh
````

Then activate environment:

```bash
source .venv/bin/activate   # if uv created env
# or
source venv/bin/activate
```

### **Manual Setup**

```bash
pip install -r requirements.txt
```

---

## 🌱 Synthetic Demo Data

This project supports **SDV synthetic data generation** for:

- ✔ UI demos
- ✔ Pipeline testing
- ✔ Showing dashboards and drift charts
- ❌ NOT used for training the real model.

Generate sample data:

```bash
python scripts/create_sample_data.py
```

> Do install dev dependencies before running the script. `SDV` installation can take some time.

---

## ⚡ API Serving

Start FastAPI:

```bash
uvicorn churn_compass.api.main:app --reload
```

Test prediction:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore": 650, "Age": 35, "Balance": 15000, ...}'
```

---

## 🚘 Drift Monitoring (Evidently)

```bash
python -m churn_compass.monitoring.check_drift \
  --reference data/processed/train.parquet \
  --current data/processed/new_batch.parquet
```

---

## 📊 React UI Dashboard

Start frontend:

```bash
cd frontend
bun install
bun run dev
```

Open `http://localhost:5173` in your browser.

---

## 🧪 Testing & Code Quality

```bash
pytest -v
ruff check src/
mypy src/
```

---

## 📌 Business Metrics Tracked

* **Precision@K (Top decile targeting)**
* **Recall@K**
* **Lift vs. random strategy**
* **PR-AUC**
* **Cost-based campaign simulation (future)**

---

## 🔒 Security & Privacy

* PII masked in logs
* Environment-based secrets
* DuckDB for local dev — Postgres for production
* Synthetic dataset support to avoid real customer exposure

---

## 📝 License

MIT — free for commercial and academic use.
