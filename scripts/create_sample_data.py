"""
Churn Compass - Integrated Synthetic Data Generator

Generates synthetic data AND processes it through the ingestion pipeline.

Purpose:
- Generate synthetic churn dataset for pipeline & UI demonstration.
- Base: 2,500 rows
- Drifted: 1,000 rows (Age shift + churn probability increase)
- Quality evaluation using SDV metrics

Outputs:
- data/raw/sample.csv (raw synthetic data)
- data/processed/sample_reference.parquet (base synthetic, pipeline-processed)
- data/processed/sample_current_drifted.parquet (drifted synthetic, pipeline-processed)
- quality_report.json (SDV quality metrics)

Usage:
    python synthetic_data_generator.py --input data/raw/real_data.csv --model gaussian_copula
    python synthetic_data_generator.py --input data/raw/real_data.csv --model copulagan --base-rows 5000
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from churn_compass.config.settings import settings
from churn_compass.io.file_io import FileIO
from churn_compass.logging.logger import log_execution_time, setup_logger
from churn_compass.pipelines.ingest_pipeline import data_ingestion_flow
from churn_compass.validation.schema_training import detect_leakage_columns


logger = setup_logger("create_sample_data")


# CLI arguments
def parse_arguments():
    """parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic churn dataset using SDV + Pipeline", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Generating data using GaussianCopula
                python synthetic_data_generator.py --input data/raw/real_data.csv

                # Custom row counts (DO NOT USE THIS DATA FOR TRAINING)
                python synthetic_data_generator.py --input data/raw/real_data.csv --base-rows 5000 --drift-rows 2000

                # Skip pipeline processing (faster, but doesn't validate!)
                python synthetic_data_generator.py --input data/raw/real_data.csv --skip-pipeline
                """,
    )

    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to real dataset CSV for training the synthetic model", 
    )

    parser.add_argument(
        "--base-rows", 
        type=int, 
        default=2000, 
        help="Number of rows for base synthetic dataset (default: 2000)", 
    )

    parser.add_argument(
        "--skip-pipeline", 
        action="store_true", 
        help="Skip running synthetic data through ingestion pipeline (Faster but no validation)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: uses settings.data_raw_dir and settings.data_processed_dir)",
    )

    return parser.parse_args()

# Data loading & cleaning
@log_execution_time(logger)
def load_and_clean(input_path: str) -> pd.DataFrame:
    """
    Load real reference data and apply basic cleaning
    
    :param input_path: Path to real dataset CSV
    :type input_path: str
    :return: Cleaned DataFrame
    :rtype: DataFrame
    :raise: FileNotFoundError, If input file doesn't exist
    """

    logger.info(f"Started loading and cleaning data from: {input_path}")

    try:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = FileIO().read_csv(input_path)

        logger.info(
            "Real data loaded for cleaning", 
            extra={
                "shape": df.shape, 
                "columns": df.columns.tolist()
            }, 
        )

        #same leakage detection as our pipeline
        leakage_cols = detect_leakage_columns(df)
        if leakage_cols:
            logger.warning(f"Data leakage found: removing leakage columns: {leakage_cols}")
            df = df.drop(columns=leakage_cols)
        
        # remove duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Removing {n_duplicates} duplicate rows")
            df = df.drop_duplicates()
        
        logger.info(f"Cleaned dataset successfully. Final shape: {df.shape}")

        return df
    
    except Exception as e:
        logger.error(
            "Failed to load and clean real data", 
            extra={
                "status": "error", 
                "error_type": type(e).__name__, 
                "error_message": str(e)
            }
        )
        raise

# Metadata configuration
@log_execution_time(logger)
def configure_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """
    Configure SDV metadata for the dataset. This ensures synthetic data has the same
    structure and types as data processed through the pipeline.
    
    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Configured metadata object
    :rtype: SingleTableMetadata
    """
    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # define column types
        numerical = [
            "CreditScore", 
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary",
            "Point Earned",
            "Satisfaction Score",
        ]

        categorical  = ["Geography", "Gender", "Card Type", "Surname"]

        binary = ["HasCrCard", "IsActiveMember", "Exited", "Complain"]

        for col in numerical:
            if col in df.columns:
                metadata.update_column(col, sdtype="numerical")

        for col in categorical:
            if col in df.columns:
                metadata.update_column(col, sdtype="categorical")
            
        for col in binary:
            if col in df.columns:
                metadata.update_column(col, sdtype="categorical")
        
        # Handle ID columns if present
        if "RowNumber" in df.columns:
            metadata.update_column("RowNumber", sdtype="id")
            metadata.set_primary_key("RowNumber")

        if "CustomerId" in df.columns:
            metadata.update_column("RowNumber", sdtype="id")
        
        logger.info("Metadata configured successfully")

        return metadata
    
    except Exception as e:
        logger.error(
            "Failed to configure metadata", 
            extra={
                "status": "error", 
                "error_type": type(e).__name__, 
                "error_message": str(e), 
            }, 
        )
        raise

# data gen
@log_execution_time(logger)
def fit_synthesizer(metadata: SingleTableMetadata) -> GaussianCopulaSynthesizer:
    """
    Configure gaussian_copula model for data synthesis
    
    :param metadata: SDv metadata object
    :type metadata: SingleTableMetadata
    :return: GaussianCopulaSynthesizer
    """

    logger.info("Initializing GaussianCopula Synthesizer")
    return GaussianCopulaSynthesizer(
        metadata=metadata, 
        enforce_min_max_values=True, 
        enforce_rounding=True, 
        numerical_distributions={
            "CreditScore": "beta", 
            "Age": "gama", 
            "Balance": "beta", 
            "EstimatedSalary": "beta"
        }
    )

# Drift application
@log_execution_time(logger)
def apply_drift(df: pd.DataFrame, Original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply drift transformation to synthetic data.

    Drift specification:
        - Age: +5 years average shift
        - Churn rate: Increase from ~21% to 27%
        - Other numerical columns ±2.5% random noise
    
    :param df: Synthetic Dataframe to apply drift to
    :type df: pd.DataFrame
    :param Original_df: Original reference DataFrame
    :type Original_df: pd.DataFrame
    :return: Drifted DataFrame
    :rtype: DataFrame
    """

    logger.info("Applying drift transformations")
    df_drift = df.copy()

    # Age drift +5 years
    if "Age" in df_drift.columns:
        original_age_mean = Original_df["Age"].mean()
        df_drift["Age"] = df_drift["Age"] + 5
        df_drift["Age"] = df_drift["Age"].clip(
            lower=Original_df["Age"].min(), 
            upper=Original_df["Age"].max() + 10
        )
        new_age_mean = df_drift["Age"].mean()

        logger.info(
            f"Age drift applied: {original_age_mean:.2f} -> {new_age_mean:.2f}"
            f"(+{new_age_mean - original_age_mean:.2f} Years)"
        )
    
    # churn rate drift: -> 27
    if "Exited" in df_drift.columns:
        target_churn = 0.28
        current_churn = df_drift["Exited"].mean()
        additional_churns = int(len(df_drift) * target_churn) - df_drift["Exited"].sum()

        if additional_churns > 0: