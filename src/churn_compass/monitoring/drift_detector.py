from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import pandas as pd
import mlflow

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from churn_compass import settings, setup_logger
from churn_compass.modeling import calculate_all_metrics
from churn_compass.io import FileIO

logger = setup_logger(__name__)

COLUMN_DRIFT_BASELINE = 0.1


class DriftDetector:
    """
    Drift detection using Evidently.

    Supports:
    - Data drift
    - Prediction drift
    - Performance drift
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_col: str = "Exited",
        prediction_col: str = "probability",
    ):
        self.target_col = target_col
        self.prediction_col = prediction_col

        if feature_columns is None:
            self.feature_columns = [
                c for c in reference_data.columns
                if c not in {self.target_col, "prediction", self.prediction_col}
            ]
        else:
            self.feature_columns = feature_columns

        self.reference_data = reference_data[self.feature_columns].copy()

        numeric_cols = self.reference_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = self.reference_data.select_dtypes(include=["object", "category"]).columns.tolist()

        self.reference_schema = DataDefinition(
            numerical_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

        logger.info(
            "DriftDetector initialized",
            extra={
                "reference_samples": len(self.reference_data),
                "feature_columns": len(self.feature_columns),
            },
        )

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        drift_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        logger.info("Detecting data drift", extra={"current_samples": len(current_data)})

        current_data = current_data[self.feature_columns]

        numeric_cols = current_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = current_data.select_dtypes(include=["object", "category"]).columns.tolist()

        current_schema = DataDefinition(
            numerical_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

        reference_ds = Dataset.from_pandas(self.reference_data, self.reference_schema)
        current_ds = Dataset.from_pandas(current_data, current_schema)

        report = Report([
            DataDriftPreset(columns=self.feature_columns, drift_share=drift_threshold)
        ])

        eval_report = report.run(current_data=current_ds, reference_data=reference_ds)
        metrics = eval_report.dict().get("metrics", [])

        # saving report
        try:
            html_path = str(settings.monitoring_report_html_output_path / "report.html")
            json_path = str(settings.monitoring_report_json_output_path / "report.json")
            eval_report.save_html(html_path)
            eval_report.save_json(json_path)

            logger.info("Reports saved successfully.", extra={"json_path": json_path, "html_path": html_path})
        except Exception:
            logger.error("Saving report failed", exc_info=True)

        drifted_columns: Dict[str, float] = {}
        dataset_drift_count = 0
        dataset_drift_share = 0.0

        for m in metrics:
            column = m["config"].get("column")
            value = m.get("value")

            if column is None and isinstance(value, dict):
                dataset_drift_count = value.get("count", 0)
                dataset_drift_share = value.get("share", 0.0)
            elif column and isinstance(value, (float, int)) and value > COLUMN_DRIFT_BASELINE:
                drifted_columns[column] = float(value)

        result = {
            "drift_detected": dataset_drift_count > 0,
            "drift_share": dataset_drift_share,
            "n_drifted_columns": len(drifted_columns),
            "n_total_columns": len(self.feature_columns),
            "drifted_columns": drifted_columns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Data drift detection completed", extra=result)
        return result

    def detect_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
    ) -> Dict[str, Any]:
        ref_mean = reference_predictions.mean()
        cur_mean = current_predictions.mean()

        result = {
            "prediction_drift_detected": abs(cur_mean - ref_mean) > settings.prediction_drift_threshold,
            "reference_mean": float(ref_mean),
            "current_mean": float(cur_mean),
            "mean_shift": float(cur_mean - ref_mean),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Prediction drift detection completed", extra=result)
        return result

    def detect_performance_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        logger.info("Detecting performance drift")

        for col in (self.target_col, self.prediction_col):
            if col not in current_data.columns:
                raise ValueError(f"Missing column '{col}' for performance drift detection")

        ref_metrics = calculate_all_metrics(
            reference_data[self.target_col].to_numpy(),
            reference_data[self.prediction_col].to_numpy(),
        )

        cur_metrics = calculate_all_metrics(
            current_data[self.target_col].to_numpy(),
            current_data[self.prediction_col].to_numpy(),
        )

        drops = {
            "pr_auc": ref_metrics["pr_auc"] - cur_metrics["pr_auc"],
            "precision_at_k": ref_metrics["top_precision_at_k"] - cur_metrics["top_precision_at_k"],
            "recall_at_k": ref_metrics["top_recall_at_k"] - cur_metrics["top_recall_at_k"],
        }

        degraded = any(v > threshold for v in drops.values())

        result = {
            "performance_degraded": degraded,
            "metric_drops": drops,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Performance drift detection completed", extra=result)
        return result

    def log_to_mlflow(self, results: Dict[str, Any], prefix: str):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(f"{settings.mlflow_experiment_name}_monitoring")

        with mlflow.start_run(run_name=f"{prefix}_drift"):
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v)

            mlflow.log_dict(results, f"{prefix}_drift.json")
