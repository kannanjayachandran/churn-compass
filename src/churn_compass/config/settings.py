"""
Churn Compass - Configuration Management

Central configuration for all project settings. Uses environment variables
with sensible defaults for local development.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Project Root Directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Environment variables override defaults and can be set in .env file.
    Prefix: CHURN_COMPASS_    
    """

    # Project Metadata
    project_name: str = "churn_compass"
    version: str = "0.1.0"
    environment: str = Field(default="local", 
                             json_schema_extra={"env": "CHURN_COMPASS_ENV"}
                            )
    
    # Paths
    data_raw_dir: Path = DATA_DIR / "raw"
    data_interim_dir: Path = DATA_DIR / "interim"
    data_processed_dir: Path = DATA_DIR / "processed"

    # MLflow Configuration
    mlflow_tracking_uri: str = f"file://{MLFLOW_DIR}"
    mlflow_experiment_name: str = "churn_compass"
    mlflow_model_name: str = "churn_compass_xgb"

    # Database Configuration (Postgres for Production)
    db_type: str = Field(default="duckdb",
                         json_schema_extra={"env": "CHURN_COMPASS_DB_TYPE"})
    postgres_host: str = Field(default="localhost", 
    json_schema_extra={"env": "CHURN_COMPASS_POSTGRES_HOST"})
    postgres_port: int = Field(default=5432, 
    json_schema_extra={"env": "CHURN_COMPASS_POSTGRES_PORT"})
    postgres_database: str = Field(default="churn_compass", 
    json_schema_extra={"env": "CHURN_COMPASS_POSTGRES_DB"})
    postgres_user: str = Field(default="churn_user", 
    json_schema_extra={"env": "CHURN_COMPASS_POSTGRES_USER"})
    postgres_password: str = Field(default="", 
    json_schema_extra={"env": "CHURN_COMPASS_POSTGRES_PASSWORD"})
    
    # DuckDB Configuration (for Local Development)
    duckdb_path: Path = DATA_DIR / "churn_compass.duckdb"

    # Model training Configuration
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2 # from remaining training data

    # Business metrics Configuration
    top_k_percent: float = 0.10

    # feature flags
    enable_shap_explanations: bool = True
    enable_monitoring: bool  = True
    enable_auto_retraining: bool = True # Gated - requires manual approval

    # columns to drop
    leakage_columns: list[str] = Field(
        default_factory=lambda: ["Complain", "Satisfaction Score"]
    )
    id_columns: list[str] = Field(
        default_factory=lambda: ["RowNumber", "CustomerId", "Surname"]
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", 
                          json_schema_extra={"env": "CHURN_COMPASS_API_HOST"})
    api_port: int = Field(default=8000, 
                          json_schema_extra={"env": "CHURN_COMPASS_API_PORT"})
    
    # logging Configuration
    log_level: str = Field(default="INFO", 
                           json_schema_extra={"env": "CHURN_COMPASS_LOG_LEVEL"})
    log_format: str = "json"
    log_file: Optional[Path] = PROJECT_ROOT / "logs" / "churn_compass.log"

    # S3 Configuration (for production)
    s3_enables: bool = Field(default=False, 
                             json_schema_extra={"env": "CHURN_COMPASS_S3_ENABLED"})
    s3_bucket: str = Field(default="", 
                           json_schema_extra={"env": "CHURN_COMPASS_S3_BUCKET"})
    s3_prefix: str = Field(default="churn-compass", 
                           json_schema_extra={"env": "CHURN_COMPASS_S3_PREFIX"})
    aws_region: str = Field(default="us-east-1", 
                            json_schema_extra={"env": "AWS_REGION"})

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    def get_postgres_uri(self) -> str:
        """Generate PostgresSQL connection URI"""
        pw = f":{self.postgres_password}" if self.postgres_password else ""
        return f"postgresql://{self.postgres_user}{pw}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [
            self.data_raw_dir,
            self.data_interim_dir,
            self.data_processed_dir,
            MLFLOW_DIR,
            self.log_file.parent if self.log_file else None
        ]:
            if dir_path:
                dir_path.mkdir(parents=True, exist_ok=True)


# Global Settings Instance
settings = Settings()

# Ensure all directories exists
settings.ensure_directories()


# Export commonly used paths
__all__ = ["settings", "PROJECT_ROOT", "DATA_DIR", "MLFLOW_DIR"]