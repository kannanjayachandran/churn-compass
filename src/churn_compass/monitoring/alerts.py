"""
Churn Compass Alert system
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from enum import Enum
import json
import requests

from churn_compass import setup_logger, settings

logger = setup_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    def __init__(
            self, 
            alert_type: str, 
            severity: AlertSeverity, 
            message: str, 
            details: Optional[Dict[str, Any]] = None, 
            timestamp: Optional[datetime] = None,
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details
        self.timestamp = timestamp or datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type, 
            "severity": self.severity.value, 
            "message": self.message, 
            "details": self.details, 
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"), 
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def __repr__(self) -> str:
        return f"Alert(type={self.alert_type}, severity={self.severity.value})"
    

class AlertManager:
    def __init__(
            self, 
            data_drift_warning: float = 0.5, 
            data_drift_critical: float = 0.7, 
            prediction_mean_shift_warning: float = 0.10, 
            prediction_mean_shift_critical: float = 0.15, 
            performance_drop_warning: float = 0.05, 
            performance_drop_critical: float = 0.10, 
    ):
        self.alerts: List[Alert] = []

        self.data_drift_warning = data_drift_warning
        self.data_drift_critical = data_drift_critical
        self.prediction_mean_shift_warning =  prediction_mean_shift_warning
        self.prediction_mean_shift_critical =  prediction_mean_shift_critical
        self.performance_drop_warning =  performance_drop_warning
        self.performance_drop_critical = performance_drop_critical

        logger.info("AlertManager initialized")

    def check_data_drift(self, results: Dict[str, Any]) -> Optional[Alert]:
        if not results.get("drift_detected", False):
            return None
        
        drift_share = results["drift_share"]

        if drift_share >= self.data_drift_critical:
            severity = AlertSeverity.CRITICAL
        elif drift_share >= self.data_drift_warning:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        alert = Alert(
            alert_type="data_drift", 
            severity=severity, 
            message=f"Data drift detected: {drift_share:.1%} of features drifted", 
            details=results, 
        )

        self.alerts.append(alert)
        logger.warning("Data drift alert generated", extra=alert.to_dict())

        return alert
    
    def check_prediction_drift(self, results: Dict[str, Any]) -> Optional[Alert]:
        if not results.get("prediction_drift_detected", False):
            return None
        
        mean_shift = abs(results["mean_shift"])

        if mean_shift >= self.prediction_mean_shift_critical:
            severity = AlertSeverity.CRITICAL
        elif mean_shift >= self.prediction_mean_shift_warning:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        alert = Alert(
            alert_type="prediction_drift", 
            severity=severity, 
            message=f"Prediction mean shift detected: {mean_shift:.3f}", 
            details=results, 
        )

        self.alerts.append(alert)
        logger.warning("Prediction drift alert generated", extra=alert.to_dict())
        
        return alert
    
    def check_performance_degradation(self, results: Dict[str, Any]) -> Optional[Alert]: 
        if not results.get("performance_degraded", False):
            return None
        
        pr_auc_drop = results.get("pr_auc_drop", 0.0)

        if pr_auc_drop >= self.performance_drop_critical:
            severity = AlertSeverity.CRITICAL
        elif pr_auc_drop >= self.performance_drop_warning:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        alert = Alert(
            alert_type="performance_degradation", 
            severity=severity, 
            message=f"Model PR_AUC dropped by {pr_auc_drop:.3f}", 
            details=results, 
        )

        self.alerts.append(alert)
        logger.warning("Performance degradation alert generated", extra=alert.to_dict())

        return alert
    
    # Orchestration
    def check_all(
            self, 
            data_drift: Optional[Dict[str, Any]] = None, 
            prediction_drift: Optional[Dict[str, Any]] = None, 
            performance: Optional[Dict[str, Any]] = None, 
    ) -> List[Alert]:
        alerts = []

        if data_drift:
            a = self.check_data_drift(data_drift)
            if a:
                alerts.append(a)
        
        if prediction_drift:
            a = self.check_prediction_drift(prediction_drift)
            if a:
                alerts.append(a)
        
        if performance:
            a = self.check_performance_degradation(performance)
            if a:
                alerts.append(a)
        
        return alerts
    

    def clear_alerts(self):
        self.alerts.clear()

    def send_to_slack(self, alerts: List[Alert]) -> None:
        """
        Send alerts to slack via incoming webhook.
        
        :param alerts: Description
        :type alerts: List[Alert]
        """
        if not settings.slack_webhook_url:
            logger.info("Slack webhook not configured, skipping alert delivery")
            return
        
        for alert in alerts:
            payload = {
                "text": self._format_slack_message(alert)
            }

            try:
                response = requests.post(
                    settings.slack_webhook_url, 
                    json=payload, 
                    timeout=5, 
                )
                response.raise_for_status()

                logger.info(
                    "Alert sent to Slack", 
                    extra={"alert_type": alert.alert_type, "severity": alert.severity.value}, 
                )

            except Exception as e:
                logger.exception(
                    "Failed to send alert to slack", 
                    extra={"alert_type": alert.alert_type}, 
                )
    
    def _format_slack_message(self, alert: Alert) -> str:
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            }
        
        header = f"{severity_emoji[alert.severity]} * {alert.severity.value.upper()} * - '{alert.alert_type}'"
        body = alert.message

        details = ""
        if alert.details:
            details = "\n```" + json.dumps(alert.details, indent=2)[:1500] + "```"  # Slack has a ~4000 char limit
        
        return f"{header}\n{body}{details}"