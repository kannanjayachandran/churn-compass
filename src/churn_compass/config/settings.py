"""
Churn Compass - Configuration Management

Central configuration for all project settings.
Environment variables override defaults via pydantic BaseSettings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Env Prefix: CHURN_COMPASS_
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="CHURN_COMPASS_",
        extra="ignore",
    )

    # Project Metadata
    project_name: str = "churn_compass"
    version: str = "0.1.0"
    environment: Literal["local", "dev", "prod"] = "local"

    # Data Paths
    data_raw_dir: Path = DATA_DIR / "raw"
    data_processed_dir: Path = DATA_DIR / "processed"

    # MLflow
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "churn_compass"
    mlflow_model_name: str = "churn_compass_xgb"

    # Database Configuration
    db_type: Literal["duckdb", "postgres"] = "postgres"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "churn_compass"
    postgres_user: str = "churn_user"
    postgres_password: str = Field(default="")

    # DuckDB
    duckdb_path: Path = DATA_DIR / "churn_compass.duckdb"

    # Train/val/Test Splits
    random_seed: int = 529
    test_size: float = 0.2
    val_size: float = 0.2  # applied to remaining training set

    # Serving
    prediction_threshold: float = 0.5
    high_risk_threshold: float = 0.7

    # Monitoring
    prediction_drift_threshold: float = 0.6
    monitoring_report_json_output_path: Path = ARTIFACT_DIR
    monitoring_report_html_output_path: Path = ARTIFACT_DIR
    slack_webhook_url: Optional[AnyHttpUrl] = None

    # Business metrics
    top_k_percent: float = 0.15

    # Feature flags
    enable_shap_explanations: bool = True
    enable_monitoring: bool = True
    auto_retrain_enabled: bool = False  # Gated
    enable_docs_in_prod: bool = False

    # Schema / Data
    leakage_columns: list[str] = Field(
        default_factory=lambda: ["Complain", "Satisfaction Score"]
    )
    id_columns: list[str] = Field(
        default_factory=lambda: ["RowNumber", "CustomerId", "Surname"]
    )

    min_minority_class_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Minimum acceptable minority class percentage for imbalance check",
    )

    max_missing_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed missing value percentage per column",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    csv_chunk_size: int = 2000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # logging
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "text"] = "text"
    log_file: Optional[Path] = PROJECT_ROOT / "logs" / "churn_compass.log"

    # Validators
    @model_validator(mode="after")
    def validate_configs(self):
        """Validate cross-field constraints"""
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")

        for name, value in {
            "prediction_threshold": self.prediction_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "prediction_drift_threshold": self.prediction_drift_threshold,
            "top_k_percent": self.top_k_percent,
        }.items():
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 1 and 0")
        return self

    @model_validator(mode="after")
    def mlflow_defaults(self):
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = f"file://{MLFLOW_DIR}"
        return self

    @model_validator(mode="after")
    def ensure_directories(self):
        """Auto-create necessary directories on settings initialization."""
        directories = [
            self.data_raw_dir,
            self.data_processed_dir,
            ARTIFACT_DIR,
            MLFLOW_DIR,
            self.log_file.parent if self.log_file else None,
        ]

        for directory in filter(None, directories):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_production_security(self):
        """Enforce security requirements in production"""
        if self.environment == "prod":
            if not self.postgres_password:
                raise ValueError(
                    "Postgres password is required in production."
                    "Set password in envrionment"
                )
        return self

    # helpers
    def get_postgres_uri(self) -> str:
        """Generate PostgresSQL connection URI"""
        pw = f":{self.postgres_password}" if self.postgres_password else ""

        return (
            f"postgresql://{self.postgres_user}{pw}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Lazy loaded singleton settings instance.

    settings = get_settings()

    """
    return Settings()
