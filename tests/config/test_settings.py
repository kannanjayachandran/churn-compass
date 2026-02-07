from pathlib import Path
from datetime import datetime, timezone


from churn_compass.config import (
    settings,
    PROJECT_ROOT,
    MLFLOW_DIR,
    DATA_DIR,
    generate_run_id,
)


def test_settings_importable():
    """
    Settings should import without raising
    """
    assert settings is not None


def test_project_path_exists():
    """Core project directories must exist or be creatable"""
    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists()

    assert isinstance(DATA_DIR, Path)
    assert DATA_DIR.exists

    assert isinstance(MLFLOW_DIR, Path)


def test_mlflow_uri_configured():
    """
    MLFlow tracking URI must be a non-empty string.
    File based URI is acceptable for local dev.
    """
    uri = settings.mlflow_tracking_uri

    assert isinstance(uri, str)
    assert uri.strip() != ""

    # allow file:./mlruns ro sqlite://mlflow.db
    assert uri.startswith(("file:", "sqlite:", "http"))


def test_top_k_percent_range():
    """
    top-k percent must be in (0, 1].
    """
    k = settings.top_k_percent

    assert isinstance(k, float)
    assert 0.0 < k <= 1.0


def test_prediction_threshold_range():
    """
    Prediction threshold must be a valid probability.
    """
    t = settings.prediction_threshold

    assert isinstance(t, float)
    assert 0.0 < t < 1.0


def test_prediction_drift_threshold_range():
    """
    Prediction drift threshold must be non-negative and same.
    """
    t = settings.prediction_drift_threshold

    assert isinstance(t, float)
    assert 0.0 <= t <= 1.0


def test_database_type_support():
    """Only supported DB backends should be allowed"""
    assert settings.db_type in {"duckdb", "postgres"}


def test_duckdb_path_when_enabled():
    """
    DuckDB path must be defined when using duckdb.
    """
    if settings.db_type == "duckdb":
        assert settings.duckdb_path is not None
        assert isinstance(settings.duckdb_path, Path)


def test_auto_retrain_flag_is_boolean():
    """
    Auto-retrain flag must always be boolean.
    """
    assert isinstance(settings.auto_retrain_enabled, bool)


def test_generate_run_id(mocker):
    """"""
    fixed_now = datetime(2025, 12, 26, 15, 30, 0, tzinfo=timezone.utc)
    mock_datetime = mocker.patch("churn_compass.config.run_id.datetime")
    mock_datetime.now.return_value = fixed_now

    prefix = "experiment"
    expected_id = "experiment_20251226_153000"

    result = generate_run_id(prefix)

    assert result == expected_id
    mock_datetime.now.assert_called_once_with(timezone.utc)
