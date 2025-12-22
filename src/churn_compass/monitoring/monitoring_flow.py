"""
Churn Compass - Monitoring Orchestration Flow
"""

import argparse
from typing import Optional, Dict
from datetime import datetime, timezone

import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from churn_compass import settings, setup_logger, log_execution_time
from churn_compass.io import FileIO, DatabaseIO
from .drift_detector import DriftDetector
from .alerts import AlertManager, AlertSeverity

logger = setup_logger(__name__)


@task(retries=2)
@log_execution_time(logger)
def load_reference_data(path: str) -> pd.DataFrame:
    return FileIO().read_parquet(path)


@task(retries=2)
@log_execution_time(logger)
def load_current_data(source: str, source_type: str) -> pd.DataFrame:
    if source_type == "parquet":
        return FileIO().read_parquet(source)
    elif source_type == "csv":
        return FileIO().read_csv(source)
    elif source_type == "sql":
        return DatabaseIO().read_query(source)
    raise ValueError(f"Unsupported source type: {source_type}")


@task(retries=1)
@log_execution_time(logger)
def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
    detector = DriftDetector(reference_df)
    return detector.detect_data_drift(current_df)


@task(retries=1)
@log_execution_time(logger)
def detect_prediction_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
    if "probability" not in current_df.columns:
        return {"prediction_drift_detected": False}
    
    ref_probs = (
        reference_df["probability"]
        if "probability" in reference_df.columns
        else pd.Series([0.5] * len(reference_df))
    )

    detector = DriftDetector(reference_df)
    return detector.detect_prediction_drift(ref_probs, current_df["probability"])


@task(retries=1)
@log_execution_time(logger)
def detect_performance_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
    detector = DriftDetector(reference_df)
    return detector.detect_performance_drift(reference_df, current_df)


# Alerts
@task
def generate_alerts(
        data_drift: Dict, 
        prediction_drift: Dict, 
        performance_drift: Optional[Dict], 
): 
    manager = AlertManager()
    alerts = manager.check_all(
        data_drift=data_drift, 
        prediction_drift=prediction_drift, 
        performance=performance_drift, 
    )

    critical_alerts = [
        a for a in alerts if a.severity == AlertSeverity.CRITICAL
    ]

    if critical_alerts:
        manager.send_to_slack(critical_alerts)

    return alerts


@task
def create_monitoring_summary(
    data_drift: Dict, 
    prediction_drift: Dict, 
    performance_drift: Optional[Dict], 
    alerts: list, 
    ) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    md = f"""
    # Monitoring Report
    
    Generated: {ts}
    
    ## Data Drift
    - Drift detected: {data_drift.get("drift_detected", False)}
    - Drift share: {data_drift.get("drift_share", 0):.2%}
    - Drifted columns: {data_drift.get("n_drifted_columns", 0)}
    """

    if data_drift["drifted_columns"]:
        md += "### Drifted Features\n"
        for col, score in data_drift.get("drifted_columns", {}).items():
            md += f"- {col}: {score:.3f}\n"

        
    md += "\n## Prediction Drift\n"
    md += f"- Drift detected: {prediction_drift.get('prediction_drift_detected', False)}\n"
    md += f"- Mean shift: {prediction_drift.get('mean_shift', 0):.4f}\n"

    if performance_drift:
        md += "\n## Performance Drift\n"
        md += f"- Degraded: {performance_drift['performance_degraded']}\n"
        md += f"- PR-AUC drop: {performance_drift['pr_auc_drop']:.4f}\n"

    md += "\n## Alerts\n"
    if alerts:
        for a in alerts:
            md += f"- [{a.severity.value}] {a.alert_type}: {a.message}\n"
    else:
        md += "- No alerts generated\n"

    return md


# flow
@flow(name="monitoring", log_prints=True)
def monitoring_flow(
    reference_path: str, 
    current_source: str, 
    current_type: str = "parquet", 
    check_performance: bool = False, 
    output_path: Optional[str] = None, 
): 
    ref_df = load_reference_data(reference_path)
    curr_df = load_current_data(current_source, current_type)

    data_drift = detect_data_drift(ref_df, curr_df)
    prediction_drift = detect_prediction_drift(ref_df, curr_df)

    performance_drift = None
    if (check_performance and "Exited" in curr_df.columns and "probability" in curr_df.columns):
        performance_drift = detect_performance_drift(ref_df, curr_df)

    alerts = generate_alerts(data_drift, prediction_drift, performance_drift)

    summary = create_monitoring_summary(data_drift, prediction_drift, performance_drift, alerts)

    create_markdown_artifact(
        key="monitoring_report", 
        markdown=summary, 
        description="Drift and performance monitoring report", 
    )

    return {
        "data_drift": data_drift, 
        "prediction_drift": prediction_drift, 
        "performance_drift": performance_drift, 
        "alerts": [a.to_dict() for a in alerts], 
        "timestamp": datetime.now(timezone.utc).isoformat(), 
    }
