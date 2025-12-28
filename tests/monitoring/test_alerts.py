"""
Tests for alert system.

Tests AlertManager severity classification and alert generation contracts.
Does NOT test Slack integration or external services.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from churn_compass.monitoring.alerts import Alert, AlertSeverity, AlertManager


@pytest.fixture
def alert_manager(mocker):
    """Default AlertManager instance with mocked logger to avoid LogRecord conflicts."""
    # Patch the logger to avoid 'message' key conflict in LogRecord
    mocker.patch("churn_compass.monitoring.alerts.logger")
    return AlertManager()


# Alert class tests
def test_alert_to_dict_structure():
    """Alert.to_dict should return expected structure."""
    alert = Alert(
        alert_type="test_alert",
        severity=AlertSeverity.WARNING,
        message="Test message",
        details={"key": "value"},
    )
    
    result = alert.to_dict()
    
    assert result["alert_type"] == "test_alert"
    assert result["severity"] == "warning"
    assert result["message"] == "Test message"
    assert result["details"] == {"key": "value"}
    assert "timestamp" in result


def test_alert_timestamp_auto_generated():
    """Alert timestamp should be auto-generated if not provided."""
    alert = Alert(
        alert_type="test",
        severity=AlertSeverity.INFO,
        message="Test",
    )
    
    assert alert.timestamp is not None
    assert isinstance(alert.timestamp, datetime)


def test_alert_custom_timestamp():
    """Custom timestamp should be preserved."""
    custom_ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    alert = Alert(
        alert_type="test",
        severity=AlertSeverity.INFO,
        message="Test",
        timestamp=custom_ts,
    )
    
    assert alert.timestamp == custom_ts


def test_alert_to_json():
    """Alert.to_json should return valid JSON string."""
    import json
    
    alert = Alert(
        alert_type="test",
        severity=AlertSeverity.INFO,
        message="Test",
    )
    
    json_str = alert.to_json()
    parsed = json.loads(json_str)
    
    assert parsed["alert_type"] == "test"


# AlertManager - Data Drift tests
def test_check_data_drift_returns_none_when_no_drift(alert_manager):
    """No alert should be generated when drift_detected is False."""
    results = {"drift_detected": False, "drift_share": 0.0}
    
    alert = alert_manager.check_data_drift(results)
    
    assert alert is None


def test_check_data_drift_info_severity(alert_manager):
    """Low drift share should generate INFO severity alert."""
    results = {"drift_detected": True, "drift_share": 0.3}
    
    alert = alert_manager.check_data_drift(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.INFO


def test_check_data_drift_warning_severity(alert_manager):
    """Medium drift share should generate WARNING severity alert."""
    results = {"drift_detected": True, "drift_share": 0.55}
    
    alert = alert_manager.check_data_drift(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING


def test_check_data_drift_critical_severity(alert_manager):
    """High drift share should generate CRITICAL severity alert."""
    results = {"drift_detected": True, "drift_share": 0.75}
    
    alert = alert_manager.check_data_drift(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL


# AlertManager - Prediction Drift tests
def test_check_prediction_drift_returns_none_when_no_drift(alert_manager):
    """No alert should be generated when prediction_drift_detected is False."""
    results = {"prediction_drift_detected": False, "mean_shift": 0.0}
    
    alert = alert_manager.check_prediction_drift(results)
    
    assert alert is None


def test_check_prediction_drift_warning_severity(alert_manager):
    """Mean shift above warning threshold should generate WARNING alert."""
    results = {"prediction_drift_detected": True, "mean_shift": 0.12}
    
    alert = alert_manager.check_prediction_drift(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING


def test_check_prediction_drift_critical_severity(alert_manager):
    """Mean shift above critical threshold should generate CRITICAL alert."""
    results = {"prediction_drift_detected": True, "mean_shift": 0.20}
    
    alert = alert_manager.check_prediction_drift(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL


# AlertManager - Performance Degradation tests
def test_check_performance_degradation_returns_none_when_no_degradation(alert_manager):
    """No alert should be generated when performance_degraded is False."""
    results = {"performance_degraded": False, "pr_auc_drop": 0.0}
    
    alert = alert_manager.check_performance_degradation(results)
    
    assert alert is None


def test_check_performance_degradation_warning_severity(alert_manager):
    """PR-AUC drop above warning threshold should generate WARNING alert."""
    results = {"performance_degraded": True, "pr_auc_drop": 0.07}
    
    alert = alert_manager.check_performance_degradation(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING


def test_check_performance_degradation_critical_severity(alert_manager):
    """PR-AUC drop above critical threshold should generate CRITICAL alert."""
    results = {"performance_degraded": True, "pr_auc_drop": 0.12}
    
    alert = alert_manager.check_performance_degradation(results)
    
    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL


# AlertManager - check_all tests
def test_check_all_accumulates_alerts(alert_manager):
    """check_all should return all generated alerts."""
    data_drift = {"drift_detected": True, "drift_share": 0.6}
    pred_drift = {"prediction_drift_detected": True, "mean_shift": 0.12}
    
    alerts = alert_manager.check_all(data_drift=data_drift, prediction_drift=pred_drift)
    
    assert len(alerts) == 2


def test_check_all_stores_in_alerts_list(alert_manager):
    """check_all should accumulate alerts in manager's alerts list."""
    data_drift = {"drift_detected": True, "drift_share": 0.6}
    
    alert_manager.check_all(data_drift=data_drift)
    
    assert len(alert_manager.alerts) == 1


def test_clear_alerts(alert_manager):
    """clear_alerts should empty the alerts list."""
    alert_manager.alerts.append(
        Alert("test", AlertSeverity.INFO, "Test")
    )
    
    alert_manager.clear_alerts()
    
    assert len(alert_manager.alerts) == 0


# AlertManager - Custom thresholds
def test_custom_thresholds(mocker):
    """Custom thresholds should be respected."""
    mocker.patch("churn_compass.monitoring.alerts.logger")
    manager = AlertManager(
        data_drift_warning=0.3,
        data_drift_critical=0.5,
    )
    
    # 0.4 should be WARNING with new threshold (0.3 < 0.4 < 0.5)
    results = {"drift_detected": True, "drift_share": 0.4}
    alert = manager.check_data_drift(results)
    
    assert alert.severity == AlertSeverity.WARNING
