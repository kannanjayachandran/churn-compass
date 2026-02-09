"""
Churn Compass - Data Ingestion Pipeline

Prefect-orchestrated ETL pipeline for ingesting and validating customer data.

Pipeline steps:
1. Extract: Read raw CSV data
2. Validate: Apply Pandera schema validation
3. Load: Write validated data to Parquet
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from prefect import flow, task

from churn_compass import (
    get_settings,
    setup_logger,
    log_execution_time,
    set_run_context,
    generate_run_id,
)
from churn_compass.io import FileIO, DatabaseIO
from churn_compass.validation import validate_raw_data, detect_leakage_columns
from churn_compass.utils import normalize_dataframe


logger = setup_logger(__name__)
file_io = FileIO()


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
    logger.info("Extracting raw data", extra={"input_path": input_path})

    suffix: str = Path(input_path).suffix.lower()

    if suffix == ".parquet":
        df = file_io.read_parquet(input_path)
    elif suffix == ".csv":
        df = file_io.read_csv(input_path)
    elif suffix == ".sql":
        db = DatabaseIO()
        df = db.read_query(input_path)
    else:
        raise ValueError(f"Unsupported source type: {suffix}")

    logger.info(
        "Data extraction completed",
        extra={
            "rows": len(df),
            "columns": len(df.columns),
        },
    )
    return df


@task(name="validate_raw_data", retries=1)
@log_execution_time(logger)
def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw data against schema.

    :param df: Raw DataFrame
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    """
    logger.info("Validating raw data schema")
    validated = validate_raw_data(df)

    logger.info(
        "Raw validation passed",
        extra={"rows": len(validated), "columns": len(validated.columns)},
    )

    return validated


@task(name="clean_data", retries=1)
@log_execution_time(logger)
def clean_data(df: pd.DataFrame, drop_nulls: bool = True) -> pd.DataFrame:
    """
    Clean data by removing leakage columns, duplicates and nulls.

    :param df: Validated DataFrame
    :type df: pd.DataFrame
    :param drop_nulls: Should we drop null values or not
    :type drop_nulls: bool
    :return: Cleaned DataFrame
    :rtype: DataFrame
    """
    logger.info("Cleaning data")

    original_rows = len(df)
    original_cols = len(df.columns)

    leakage_cols = detect_leakage_columns(df)
    if leakage_cols:
        logger.warning(f"Dropping leakage columns: {leakage_cols}")
        df = df.drop(columns=leakage_cols)

    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Removing {n_duplicates} duplicate rows")
        df = df.drop_duplicates()

    if drop_nulls:
        n_nulls = df.isnull().sum().sum()
        if n_nulls:
            logger.warning("Dropping rows with nulls", extra={"null_cells": n_nulls})
            df = df.dropna()

    logger.info(
        "Data cleaning completed",
        extra={
            "rows_removed": original_rows - len(df),
            "columns_removed": original_cols - len(df.columns),
        },
    )

    return df


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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to Parquet with Snappy compression
    file_io.write_parquet(df, output_path)

    output_path_obj = Path(output_path).absolute()
    file_size_mb = output_path_obj.stat().st_size / (1024**2)

    logger.info(
        "Data written",
        extra={
            "output_path": str(output_path_obj),
            "rows": len(df),
            "columns": len(df.columns),
            "file_size_mb": round(file_size_mb, 2),
        },
    )

    return str(output_path_obj.resolve())


# utility
def normalize_card_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CardType column and handle mapping from Card Type"""
    return normalize_dataframe(
        df, column_mapping={"Card Type": "CardType", "Point Earned": "PointEarned"}
    )


@flow(
    name="ingest_customer_data",
    description="ETL pipeline for ingesting and validating customer churn data",
    log_prints=True,
)
def data_ingestion_flow(
    input_path: Optional[str] = None, output_path: Optional[str] = None
) -> Optional[str]:
    """
    Main ingestion flow: Extract -> Validate -> Clean -> Load

    :param input_path: Path to input CSV file (default: data/raw/sample.csv)
    :type input_path: Optional[str]
    :param output_path: Path to output Parquet file (default: data/processed/customers_YYYYMMDD.parquet). If output not provided → auto-create timestamped file.
    :type output_path: Optional[str]
    :return: Path to processed Parquet file
    :rtype: Optional[str]
    """
    settings = get_settings()
    run_id = generate_run_id("ingestion")
    set_run_context(run_id, stage="ingestion")
    settings.setup()

    if input_path is None:
        input_path = str(settings.data_raw_dir / "Customer-Churn-Records.csv")

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(settings.data_processed_dir / f"customers_{ts}.parquet")

    logger.info(
        "Starting ingestion flow",
        extra={
            "input_path": input_path,
            "output_path": output_path,
            "environment": settings.environment,
        },
    )

    # Pipeline
    df_raw = extract_raw_data(input_path)
    df_normalized = normalize_card_type(df_raw)
    df_valid = validate_raw(df_normalized)
    df_clean = clean_data(df_valid)
    result = load_processed_data(df_clean, output_path)

    logger.info("Ingestion flow completed successfully", extra={"result_path": result})

    return result
