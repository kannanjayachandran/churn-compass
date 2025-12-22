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

from churn_compass.config import settings, generate_run_id
from churn_compass.logging import setup_logger, log_execution_time, set_run_context


__all__ = ["settings", "setup_logger", "log_execution_time", "set_run_context", "generate_run_id",  "__version__", "__author__"]
