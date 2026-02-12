"""
Churn Compass

A production-grade MLOps platform for predicting customer churn
in retail banking.

Key Features:
- End-to-end ML pipeline (data ingestion, training, serving)
- XGBoost model
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


from churn_compass.config.settings import get_settings
from churn_compass.logging.logger import (
    clear_run_context,
    default_logger,
    log_execution_time,
    set_run_context,
    setup_logger,
)

__all__ = [
    "get_settings",
    "setup_logger",
    "log_execution_time",
    "set_run_context",
    "default_logger",
    "clear_run_context",
    "__version__",
    "__author__",
]
