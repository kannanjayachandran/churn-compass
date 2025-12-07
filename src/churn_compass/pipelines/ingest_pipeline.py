"""
Churn Compass - Data Ingestion Pipeline

Prefect-orchestrated ETL pipeline for ingesting and validating customer data.

Pipeline steps:
1. Extract: Read raw CSV data
2. Validate: Apply Pandera schema validation
3. Clean: Remove leakage columns and duplicates
4. Load: Write validated data to Parquet

Usage:
    # Run as Prefect flow
    python -m src.bank_churn.pipelines.ingest_pipeline --demo
    
    # Or programmatically
    from src.bank_churn.pipelines.ingest_pipeline import ingest_flow
    ingest_flow(input_path="data/raw/customers.csv")
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Union
import pandas as pd
from prefect import flow, task

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger, log_execution_time
from churn_compass.io.file_io import FileIO
from churn_compass.validation.schema_training import(
    validate_raw_data, 
    detect_leakage_columns
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
    logger.info(f"Extracting data from: {input_path}")

    try:
        df =FileIO().read_csv(input_path)

        logger.info(
            "Data extraction completed", 
            extra={
                "rows": len(df), 
                "columns": len(df.columns), 
                "column_names": list(df.columns)
            }
        )

        return df
    except Exception as e:
        logger.error(f"Failed to extract data from {input_path}", exc_info=True)
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
                "validated_columns": len(validated_df.columns)
            }
        )

        return validated_df
    
    except Exception as e:
        logger.error("Data validation failed", exc_info=True)
        raise

@task(name="clean_data", retries=1)
@log_execution_time(logger)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing leakage columns and duplicates.
    
    :param df: Validated DataFrame
    :type df: pd.DataFrame
    :return: Cleaned DataFrame
    :rtype: DataFrame
    """
    logger.info("Cleaning data...")

    original_rows = len(df)
    original_cols = len(df.columns)

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
    # This is fine in our dataset, as it won't have any important values missing
    n_nulls = df.isnull().sum().sum()
    if n_nulls > 0:
        logger.warning(f"Dropping {n_nulls} null values")
        df = df.dropna()

    logger.info(
        "Data cleaning completed", 
        extra={
            "original_rows": original_rows, 
            "cleaned_rows": len(df), 
            "rows_removed": original_rows - len(df), 
            "original_columns": original_cols, 
            "cleaned_columns": len(df.columns), 
            "columns_removed": original_cols - len(df.columns)
        }
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

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to Parquet with Snappy compression
        FileIO().write_parquet(df, output_path, compression="snappy")

        # verify file was written
        output_path_obj = Path(output_path)
        file_size_mb = output_path_obj.stat().st_size / (1024 ** 2)

        logger.info(
            "Data loading completed", 
            extra={
                "output_pat": output_path, 
                "file_size_mb": round(file_size_mb, 2), 
                "rows_written": len(df), 
                "columns_written": len(df.columns)
            }
        )

        return str(output_path_obj.absolute())
    
    except Exception as e:
        logger.error(f"Failed to load data to {output_path}", exc_info=True)
        raise

@flow(
    name="ingest_customer_data", 
    description="ETL pipeline for ingesting and validating customer churn data", 
    log_prints=True
)
def ingest_flow(
    input_path: Union[str, Path], 
    output_path: Union[str, Path]
) -> str:
    """
    Main ingestion flow: Extract -> Validate -> Clean -> Load
    
    :param input_path: Path to input CSV file (default: data/raw/sample.csv)
    :type input_path: Optional[str]
    :param output_path: Path to output Parquet file (default: data/processed/customers_YYYYMMDD.parquet)
    :type output_path: Optional[str]
    :return: Path to processed Parquet file
    :rtype: str
    Example:
        >>> from churn_compass.pipelines.ingest_pipeline import ingest_flow
        >>> result = ingest_flow(input_path="data/raw/customers.csv")
    """
    # set default paths
    if input_path is None:
        input_path = settings.data_raw_dir / "sample.csv"

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = settings.data_processed_dir / f"customers_{timestamp}.parquet"

    logger.info(
        "starting ingestion flow", 
        extra={
            "input_path": input_path, 
            "output_path": output_path, 
            "environment": settings.environment
        }
    )

    # Execute pipeline
    raw_data = extract_raw_data(str(input_path))
    validated_data = validate_data(raw_data)
    cleaned_data = clean_data(validated_data)
    result_path = load_processed_data(cleaned_data, str(output_path))

    logger.info(
        "Ingestion flow completed successfully", 
        extra={"result_path": result_path}
    )

    return result_path


def main():
    """CLI entry point for running the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Run Churn Compass ingestion pipeline")
    parser.add_argument(
        "--input", 
        type=str, 
        help="Path to input CSV file", 
        default=None
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to output Parquet file", 
        default=None
    )
    parser.add_argument(
        "--demo", 
        action="store-true", 
        help="Run demo with sample data"
    )

    args = parser.parse_args()

    if args.demo:
        print("=" * 70)
        print("Running Churn Ingestion Pipeline - DEMO MODE")
        print("=" * 70)

        # use sample file
        input_path = str(settings.data_raw_dir / "sample.csv")
        output_path = str(settings.data_processed_dir / "sample.parquet")
    else:
        input_path = args.input
        output_path = args.output
    
    try:
        result = ingest_flow(input_path=input_path, output_path=output_path)

        print("\n" + "=" * 80)
        print("✅ INGESTION PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Processed data written to: {result}")

        # Display file info
        result_path = Path(result)
        if result_path.exists():
            size_mb = result_path.stat().st_size / (1024 ** 2)
            print(f"File size: {size_mb:.2f} MB")

            # Read and display summary
            df = FileIO().read_parquet(result)
            print(f"\nData Summary:")
            print(f"    Rows: {len(df):,}")
            print(f"    Columns: {len(df.columns)}")
            print(f"    Churn rate: {df['Exited'].mean():.2%}")
            print(f"    Columns: {', '.join(df.columns)}")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ INGESTION PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        logger.error("Pipeline execution failed", exc_info=True)
        raise

if __name__ == "__main__":
    main()
