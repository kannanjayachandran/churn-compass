"""
Churn Compass - Configuration Management

Central configuration for all project settings.
Environment variables override defaults via pydantic BaseSettings.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Prefix: CHURN_COMPASS_
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="CHURN_COMPASS_",
        extra="ignore",
    )

    # Project Metadata
    project_name: str = "churn_compass"
    version: str = "0.1.0"
    environment: str = Field(default="local")  # local | dev | prod

    # Data Paths
    data_raw_dir: Path = DATA_DIR / "raw"
    data_processed_dir: Path = DATA_DIR / "processed"

    # MLflow
    mlflow_tracking_uri: str = f"file:{PROJECT_ROOT}/mlruns"
    mlflow_experiment_name: str = "churn_compass"
    mlflow_model_name: str = "churn_compass_xgb"

    # Database Configuration
    db_type: str = Field(default="duckdb")  # duckdb | postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "churn_compass"
    postgres_user: str = "churn_user"
    postgres_password: str = ""

    # DuckDB local path
    duckdb_path: Path = DATA_DIR / "churn_compass.duckdb"

    # Train/val/Test Splits
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2  # applied to remaining training set

    # Serving
    prediction_threshold: float = 0.5
    high_risk_threshold: float = 0.5

    # Monitoring
    prediction_drift_threshold: float = 0.5
    monitoring_report_json_output_path: Path = DATA_DIR / "artifacts"
    monitoring_report_html_output_path: Path = DATA_DIR / "artifacts"
    slack_webhook_url: Optional[str] = None

    @model_validator(mode="after")
    def validate_splits(self):
        """Ensure splits makes sense."""
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
        return self

    # Business metrics Configuration
    top_k_percent: float = 0.15

    # Feature flags
    enable_shap_explanations: bool = True
    enable_monitoring: bool = True
    auto_retrain_enabled: bool = False  # Gated - requires manual approval

    # Schema-related columns
    leakage_columns: list[str] = Field(
        default_factory=lambda: ["Complain", "Satisfaction Score"]
    )
    id_columns: list[str] = Field(
        default_factory=lambda: ["RowNumber", "CustomerId", "Surname"]
    )

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_file: Optional[Path] = PROJECT_ROOT / "logs" / "churn_compass.log"

    # helper functions
    def get_postgres_uri(self) -> str:
        """Generate PostgresSQL connection URI"""
        pw = f":{self.postgres_password}" if self.postgres_password else ""

        return (
            f"postgresql://{self.postgres_user}{pw}"
            f"@{self.postgres_host}:{self.postgres_database}"
        )

    def setup(self):
        """Initialize application state (create directories, etc.)"""
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.data_raw_dir,
            self.data_processed_dir,
            MLFLOW_DIR,
            self.log_file.parent if self.log_file else None,
        ]
        for d in dirs:
            if d:
                d.mkdir(parents=True, exist_ok=True)


# Global singleton instance
settings = Settings()
