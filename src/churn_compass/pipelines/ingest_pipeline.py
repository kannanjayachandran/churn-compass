"""
Churn Compass - Data Ingestion Pipeline

Prefect-orchestrated ETL pipeline for ingesting and validating customer data.

Pipeline steps:
1. Extract: Read raw CSV data
2. Validate: Apply Pandera schema validation
3. Clean: Remove leakage columns, duplicates and nulls
4. Load: Write validated data to Parquet

Usage:
    from churn_compass.pipelines.ingest_pipeline import data_ingestion_flow
    data_ingestion_flow(input_path="data/raw/customers.csv", output_path="output.parquet")
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
from prefect import flow, task

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger, log_execution_time
from churn_compass.io.file_io import FileIO
from churn_compass.validation.schema_training import (
    validate_raw_data,
    detect_leakage_columns,
)


logger = setup_logger(__name__)


@task(name="extract_raw_data", retries=2)
@log_execution_time(logger)
def extract_raw_data(input_path: str) -> pd.DataFrame:
    """
    Extract raw data from CSV file

    :param input_path: Path to input CSV file
    :type input_path: str
    :return: Raw DataFrame
    :rtype: DataFrame
    """
    logger.info(f"Extracting raw data from: {input_path}")

    try:
        df = FileIO().read_csv(input_path)

        logger.info(
            "Data extraction completed",
            extra={
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
            },
        )

        return df
    except Exception as e:
        logger.error(
            f"Failed to extract data from {input_path}",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


@task(name="validate_data", retries=1)
@log_execution_time(logger)
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw data against schema.

    :param df: Raw DataFrame
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    """
    logger.info("Validating data schema...")

    try:
        validated_df = validate_raw_data(df)

        # Report validation results
        logger.info(
            "Data validation passed",
            extra={
                "validated_rows": len(validated_df),
                "validated_columns": len(validated_df.columns),
            },
        )

        return validated_df

    except Exception as e:
        logger.error(
            "Data validation failed",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


@task(name="clean_data", retries=1)
@log_execution_time(logger)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing leakage columns, duplicates and nulls.

    :param df: Validated DataFrame
    :type df: pd.DataFrame
    :return: Cleaned DataFrame
    :rtype: DataFrame
    """
    logger.info("Cleaning data...")

    try:
        original_row_count = len(df)
        original_col_count = len(df.columns)

        # Detect and remove leakage columns
        leakage_cols = detect_leakage_columns(df)
        if leakage_cols:
            logger.warning(f"Dropping leakage columns: {leakage_cols}")
            df = df.drop(columns=leakage_cols)

        # Remove duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Removing {n_duplicates} duplicate rows")
            df = df.drop_duplicates()

        # Remove any remaining null values
        # Our data agreement guarantees that there won't be any null values
        # Otherwise this can cause issues (not a good idea to drop null values in data ingestion)
        n_nulls = df.isnull().sum().sum()
        if n_nulls > 0:
            logger.warning(f"Dropping {n_nulls} null values")
            df = df.dropna()

        logger.info(
            "Data cleaning completed",
            extra={
                "original_row_count": original_row_count,
                "cleaned_row_count": len(df),
                "rows_removed": original_row_count - len(df),
                "original_column_count": original_col_count,
                "cleaned_column_count": len(df.columns),
                "columns_removed": original_col_count - len(df.columns),
            },
        )

        return df

    except Exception as e:
        logger.error(
            "Data cleaning failed",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


@task(name="load_processed_data", retries=2)
@log_execution_time(logger)
def load_processed_data(df: pd.DataFrame, output_path: str) -> str:
    """
    Write processed data to Parquet file

    :param df: Cleaned DataFrame
    :type df: pd.DataFrame
    :param output_path: Destination path for Parquet file
    :type output_path: str
    :return: Path to written file
    :rtype: str
    """
    logger.info(f"Loading data to: {output_path}")

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to Parquet with Snappy compression
        FileIO().write_parquet(df, output_path, compression="snappy")

        # verify file was written
        output_path_obj = Path(output_path).absolute()
        file_size_mb = output_path_obj.stat().st_size / (1024**2)

        logger.info(
            "Data loading completed",
            extra={
                "output_path_parameter": output_path,
                "output_path_saved": output_path_obj,
                "file_size_mb": round(file_size_mb, 2),
                "rows_written": len(df),
                "columns_written": len(df.columns),
            },
        )

        return str(output_path_obj)

    except Exception as e:
        logger.error(
            f"Failed to load (write) data to {output_path}",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


# Main Pipeline
@flow(
    name="ingest_customer_data",
    description="ETL pipeline for ingesting and validating customer churn data",
    log_prints=True,
)
def data_ingestion_flow(
    input_path: Optional[str], output_path: Optional[str]
) -> Optional[str]:
    """
    Main ingestion flow: Extract -> Validate -> Clean -> Load

    :param input_path: Path to input CSV file (default: data/raw/sample.csv)
    :type input_path: Optional[str]
    :param output_path: Path to output Parquet file (default: data/processed/customers_YYYYMMDD.parquet). If output not provided → auto-create timestamped file.
    :type output_path: Optional[str]
    :return: Path to processed Parquet file
    :rtype: Optional[str]
    Example:
        >>> from churn_compass.pipelines.ingest_pipeline import data_ingestion_flow
        >>> result = data_ingestion_flow(input_path="data/raw/customers.csv", output_path="data/processed/customers_YYYYMMDD.parquet")
    """
    try:
        # set default paths
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                settings.data_processed_dir / f"customers_{timestamp}.parquet"
            )

        if input_path is None:
            input_path = str(settings.data_raw_dir / "sample.csv")

        logger.info(
            "starting ingestion flow",
            extra={
                "input_path": input_path,
                "output_path": output_path,
                "environment": settings.environment,
            },
        )

        # Execute pipeline
        df_raw = extract_raw_data(input_path)
        df_valid = validate_data(df_raw)
        df_clean = clean_data(df_valid)
        result_path = load_processed_data(df_clean, output_path)

        logger.info(
            "Ingestion flow completed successfully", extra={"result_path": result_path}
        )

        return result_path

    except Exception as e:
        logger.error(
            "Failed to run data ingestion pipeline",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise
