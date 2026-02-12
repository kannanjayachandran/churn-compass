"""
Churn Compass - Data Ingestion Pipeline

Prefect-orchestrated ETL pipeline for ingestion.

Pipeline steps:
1. Extract: Read raw CSV data
2. Generate UUIDs: Create deterministic customer_uuid for each record
3. Validate: Apply Pandera schema validation
4. Store: Write to PostgreSQL (customers table) and MinIO (raw_data bucket)
5. Load: Write validated data to Parquet (processed bucket)
6. Track: Log job metadata to batch_jobs table
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import pandas as pd
from prefect import flow, task
from sqlalchemy import text

from churn_compass import (
    get_settings,
    log_execution_time,
    set_run_context,
    setup_logger,
)
from churn_compass.config import (
    generate_customer_uuid,
    generate_batch_job_id,
    generate_run_id,
)
from churn_compass.io import DatabaseIO, FileIO
from churn_compass.validation import detect_leakage_columns, validate_raw_data

logger = setup_logger(__name__)


@task(name="extract_raw_data", retries=2)
@log_execution_time(logger)
def extract_raw_data(input_path: str) -> pd.DataFrame:
    """
    Extract raw data from CSV/Parquet/SQL source.

    :param input_path: Path to input file or SQL query
    :type input_path: str
    :return: Raw DataFrame
    :rtype: pd.DataFrame
    :raises ValueError: If unsupported source type
    """
    logger.info("Extracting raw data", extra={"input_path": input_path})

    file_io = FileIO()
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


@task(name="generate_customer_uuids", retries=1)
@log_execution_time(logger)
def generate_customer_uuids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate deterministic customer_uuid for each record.

    :param df: Raw DataFrame with CustomerId column
    :type df: pd.DataFrame
    :return: DataFrame with customer_uuid column
    :rtype: pd.DataFrame
    :raises ValueError: If CustomerId column missing
    """
    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId column required for UUID generation")

    logger.info("Generating customer UUIDs")
    df = df.copy()
    customer_ids = df["CustomerId"]
    df["customer_uuid"] = customer_ids.map(
        lambda customer_id: generate_customer_uuid(int(customer_id))
    )

    logger.info(
        "Customer UUIDs generated",
        extra={"unique_customers": df["customer_uuid"].nunique()},
    )
    return df


@task(name="validate_raw_data", retries=1)
@log_execution_time(logger)
def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw data against Pandera schema.

    :param df: Raw DataFrame
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: pd.DataFrame
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

    df = df.drop_duplicates(subset=["customer_uuid"])
    n_duplicates = original_rows - len(df)
    if n_duplicates > 0:
        logger.warning(
            f"Removing {n_duplicates} duplicate customer_uuid records",
            extra={"duplicates_removed": n_duplicates},
        )

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


@task(name="store_to_postgres", retries=2)
@log_execution_time(logger)
def store_to_postgres(df: pd.DataFrame, run_id: str) -> None:
    """
    Store customer records to PostgreSQL customers table (bulk upsert).

    :param df: Cleaned DataFrame with customer_uuid
    :type df: pd.DataFrame
    :param run_id: Current ingestion run ID
    :type run_id: str
    """
    if df.empty:
        logger.warning("Received empty DataFrame, skipping insert")
        return

    logger.info("Storing customer data to PostgreSQL")

    db = DatabaseIO(db_type="postgres")
    now = datetime.now().isoformat()

    # Convert PyArrow dtypes to native Python before dict serialization.
    # PyArrow scalars (Int64Scalar, StringScalar etc.) are not JSON-serializable.
    if any("pyarrow" in str(dtype) for dtype in df.dtypes):
        df_native = df.convert_dtypes(dtype_backend="numpy_nullable")
    else:
        df_native = df

    records = df_native.to_dict("records")

    customer_records = [
        {
            "customer_uuid": r["customer_uuid"],
            "bank_customer_id": r["CustomerId"],
            "surname": r.get("Surname", ""),
            "raw_data": json.dumps(
                {k: v for k, v in r.items() if k != "customer_uuid"},
                default=str,  # safety net for any remaining non-serializable types
            ),
            "ingestion_run_id": run_id,
            "created_at": now,
            "updated_at": now,
        }
        for r in records
    ]

    insert_query = text("""
        INSERT INTO customers (
            customer_uuid,
            bank_customer_id,
            surname,
            raw_data,
            ingestion_run_id,
            created_at,
            updated_at
        )
        VALUES (
            :customer_uuid,
            :bank_customer_id,
            :surname,
            :raw_data::jsonb,
            :ingestion_run_id,
            :created_at,
            :updated_at
        )
        ON CONFLICT (customer_uuid) DO UPDATE SET
            raw_data        = EXCLUDED.raw_data,
            updated_at      = EXCLUDED.updated_at,
            ingestion_run_id = EXCLUDED.ingestion_run_id
    """)

    with db.get_connection() as conn:
        conn.execute(insert_query, customer_records)

    logger.info(
        "Customer data stored to PostgreSQL",
        extra={"records_written": len(customer_records)},
    )


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

    file_io = FileIO()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    file_io.write_parquet(df, output_path)

    output_path_obj = Path(output_path).absolute()
    file_size_mb = output_path_obj.stat().st_size / (1024**2)

    logger.info(
        "Data written to Parquet",
        extra={
            "output_path": str(output_path_obj),
            "rows": len(df),
            "columns": len(df.columns),
            "file_size_mb": round(file_size_mb, 2),
        },
    )

    return str(output_path_obj.resolve())


@task(name="create_batch_job", retries=1)
def create_batch_job(run_id: str, input_path: str) -> UUID:
    """
    Create batch job record in PostgreSQL.

    :param run_id: Current ingestion run ID
    :type run_id: str
    :param input_path: Source data path
    :type input_path: str
    :return: Job ID
    :rtype: str
    """
    job_id = generate_batch_job_id()
    db = DatabaseIO(db_type="postgres")

    insert_query = text("""
    INSERT INTO batch_jobs (job_id, run_id, job_type, status, input_path, created_at)
    VALUES (:job_id, :run_id, :job_type, :status, :input_path, :created_at)
    """)

    with db.get_connection() as conn:
        conn.execute(
            insert_query,
            {
                "job_id": job_id,
                "run_id": run_id,
                "job_type": "ingestion",
                "status": "running",
                "input_path": input_path,
                "created_at": datetime.now(),
            },
        )

    logger.info("Batch job created", extra={"job_id": job_id, "run_id": run_id})

    return job_id


@task(name="update_batch_job", retries=1)
def update_batch_job(
    job_id: str | UUID,
    status: str,
    rows_processed: int,
    output_path: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Update batch job status in PostgreSQL.

    :param job_id: Batch job ID
    :type job_id: str
    :param status: Job status (running/completed/failed)
    :type status: str
    :param rows_processed: Number of rows processed
    :type rows_processed: int
    :param output_path: Output file path
    :type output_path: Optional[str]
    :param error_message: Error message if failed
    :type error_message: Optional[str]
    """
    db = DatabaseIO(db_type="postgres")

    update_query = text("""
    UPDATE batch_jobs
    SET status = :status,
        rows_processed = :rows_processed,
        output_path = :output_path,
        error_message = :error_message,
        completed_at = :completed_at
    WHERE job_id = :job_id
    """)

    with db.get_connection() as conn:
        conn.execute(
            update_query,
            {
                "job_id": job_id,
                "status": status,
                "rows_processed": rows_processed,
                "output_path": output_path,
                "error_message": error_message,
                "completed_at": datetime.now().isoformat()
                if status in ["completed", "failed"]
                else None,
            },
        )

    logger.info("Batch job updated", extra={"job_id": job_id, "status": status})


# utility
def normalize_column_name(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names."""
    df = df.copy()

    df = df.rename(
        columns={
            "Card Type": "CardType",
            "Point Earned": "PointEarned",
        }
    )

    categorical_cols = ["CardType", "Geography", "Gender"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().str.capitalize()

    return df


@flow(
    name="ingest_customer_data",
    description="ETL pipeline for ingesting and validating customer churn data",
    log_prints=True,
)
def data_ingestion_flow(
    input_path: Optional[str] = None, output_path: Optional[str] = None
) -> Optional[str]:
    """
    Main ingestion flow: Extract → Generate UUIDs → Validate → Clean → Store → Load

    :param input_path: Path to input CSV file
    :type input_path: Optional[str]
    :param output_path: Path to output Parquet file
    :type output_path: Optional[str]
    :return: Path to processed Parquet file
    :rtype: Optional[str]"""
    settings = get_settings()
    run_id = generate_run_id("ingestion")
    set_run_context(run_id, stage="ingestion")

    if input_path is None:
        input_path = str(settings.data_raw_dir / "Customer-Churn-Records.csv")

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(settings.data_processed_dir / f"customers_{ts}.parquet")

    logger.info(
        "Starting ingestion flow",
        extra={
            "run_id": run_id,
            "input_path": input_path,
            "output_path": output_path,
            "environment": settings.environment,
        },
    )

    job_id = create_batch_job(run_id, input_path)
    try:
        df_raw = extract_raw_data(input_path)
        df_normalized = normalize_column_name(df_raw)
        df_with_uuids = generate_customer_uuids(df_normalized)
        df_valid = validate_raw(df_with_uuids)
        df_clean = clean_data(df_valid)

        store_to_postgres(df_clean, run_id)

        result = load_processed_data(df_clean, output_path)

        update_batch_job(job_id, "completed", len(df_clean), result)

        logger.info(
            "Ingestion flow completed successfully",
            extra={"result_path": result, "job_id": job_id},
        )

        return result

    except Exception as e:
        logger.error(
            "Ingestion flow failed",
            extra={"job_id": job_id, "error": str(e)},
            exc_info=True,
        )
        update_batch_job(job_id, "failed", 0, error_message=str(e))
        raise
