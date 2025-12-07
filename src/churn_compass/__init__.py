"""
Churn Compass - Customer Churn Prediction Platform

A production-grade MLOps platform for predicting and preventing customer churn
in retail banking.

Key Features:
- End-to-end ML pipeline (data ingestion, training, serving)
- Hyperparameter optimization with Optuna
- MLflow experiment tracking and model registry
- Prefect orchestration
- FastAPI serving layer
- Data drift monitoring with Evidently
- React dashboard for insights

Author: Kannan Jayachandran
Version: 0.1.0
"""


__version__ = "0.1.0"
__author__ = "Kannan Jayachandran"

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger


__all__ = ["settings", "setup_logger", "__version__", "__author__"]
