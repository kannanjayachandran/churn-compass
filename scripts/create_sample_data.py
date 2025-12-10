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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from churn_compass.config.settings import settings
from churn_compass.io.file_io import FileIO
from churn_compass.logging.logger import log_execution_time, setup_logger
from churn_compass.pipelines.ingest_pipeline import data_ingestion_flow


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
        "--drift-rows",
        type=int,
        default=1000,
        help="Number of rows for drifted synthetic dataset (default: 1000)",
    )

    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip running synthetic data through ingestion pipeline (Faster but no validation)",
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
            extra={"shape": df.shape, "columns": df.columns.tolist()},
        )

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
                "error_message": str(e),
            },
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

        categorical = ["Geography", "Gender", "Card Type", "Surname"]

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


# Drift application
@log_execution_time(logger)
def apply_drift(df: pd.DataFrame, Original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply drift transformation to synthetic data.

    Drift specification:
        - Age: +5 years average shift
        - Churn rate: Increase from ~21% to 27%
        - Other numerical columns ±2% random noise

    :param df: Synthetic Dataframe to apply drift to
    :type df: pd.DataFrame
    :param Original_df: Original reference DataFrame
    :type Original_df: pd.DataFrame
    :return: Drifted DataFrame
    :rtype: DataFrame
    """

    logger.info("Applying drift transformations")

    try:
        df_drift = df.copy()

        # Age drift +5 years
        if "Age" in df_drift.columns:
            original_age_mean = Original_df["Age"].mean()
            df_drift["Age"] = df_drift["Age"] + 5
            df_drift["Age"] = df_drift["Age"].clip(
                lower=Original_df["Age"].min(), upper=Original_df["Age"].max() + 10
            )
            new_age_mean = df_drift["Age"].mean()

            logger.info(
                f"Age drift applied: {original_age_mean:.2f} -> {new_age_mean:.2f}"
                f"(+{new_age_mean - original_age_mean:.2f} Years)"
            )

        # churn rate drift: -> 27%
        if "Exited" in df_drift.columns:
            target_churn = 0.28
            current_churn = df_drift["Exited"].mean()
            additional_churns = (
                int(len(df_drift) * target_churn) - df_drift["Exited"].sum()
            )

            if additional_churns > 0:
                non_churned_idx = df_drift[df_drift["Exited"] == 0].index
                if len(non_churned_idx) >= additional_churns:
                    flip_idx = np.random.choice(
                        non_churned_idx, size=additional_churns, replace=False
                    )
                    df_drift.loc[flip_idx, "Exited"] = 1

            new_churn = df_drift["Exited"].mean()
            logger.info(
                f"Churn rate drift applied: {current_churn:.2%} -> {new_churn:.2%}"
                f"(target: {target_churn:.2%})"
            )

        # Minimal drift to other numerical columns
        drift_cols = ["CreditScore", "Balance", "EstimatedSalary"]
        for col in drift_cols:
            if col in df_drift.columns and pd.api.types.is_numeric_dtype(df_drift[col]):
                noise = np.random.normal(0, 0.020, len(df_drift))
                df_drift[col] = df_drift[col] * (1 + noise)
                df_drift[col] = df_drift[col].clip(
                    lower=Original_df[col].min(), upper=Original_df[col].max()
                )

        logger.info(f"Drift applied to {len(drift_cols)} additional columns")
        return df_drift

    except Exception as e:
        logger.error(
            "Failed to apply data drift",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise


# Quality evaluation
@log_execution_time(logger)
def evaluate_synthetic_data_quality(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: SingleTableMetadata
) -> Dict:
    """
    Evaluate synthetic data quality using SDV metrics

    :param real_data: Original dataset
    :type real_data: pd.DataFrame
    :param synthetic_data: Generated synthetic dataset
    :type synthetic_data: pd.DataFrame
    :param metadata: SDV metadata
    :type metadata: SingleTableMetadata
    :return: Quality report dictionary
    :rtype: Dict[Any, Any]
    """
    logger.info("Running SDV quality evaluation...")

    try:
        # Run quality evaluation
        quality_report = evaluate_quality(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )

        # Run diagnostic
        diagnostics_report = run_diagnostic(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )

        quality_score = quality_report.get_score()
        properties = quality_report.get_properties()

        logger.info(
            "Quality evaluation completed",
            extra={
                "overall_score": f"{quality_score:.3f}",
                "column_shapes_score": f"{properties.get('Column Shapes', {}).get('Score', 0):.3f}",
                "column_pair_trend_score": f"{properties.get('Column Pair Trends', {}).get('Score', 0):.3f}",
            },
        )

        return {
            "quality_score": quality_score,
            "properties": properties.to_dict(),
            "diagnostic": str(diagnostics_report),
        }

    except Exception as e:
        logger.warning(f"Quality evaluation failed: {e}")
        return {"overall_quality_score": None, "error": str(e)}


# Data generation
@log_execution_time(logger)
def generate_synthetic_data(
    input_path: str,
    base_rows: int = 2500,
    drift_rows: int = 1000,
    skip_pipeline: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Synthetic data generation workflow

    :param input_path: Path to real dataset CSV
    :type input_path: str
    :param base_rows: Number of base synthetic rows
    :type base_rows: int
    :param drift_rows: Number of drifted synthetic rows
    :type drift_rows: int
    :param skip_pipeline: Skip pipeline processing
    :type skip_pipeline: bool
    :param output_dir: Output director (None = use settings)
    :type output_dir: str
    :return: Dictionary of output file paths
    :rtype: Dict[str, str]
    """
    logger.info("Synthetic data generator Started")

    try:
        if output_dir:
            raw_dir = Path(output_dir) / "raw"
            processed_dir = Path(output_dir) / "processed"
        else:
            raw_dir = Path(settings.data_raw_dir)
            processed_dir = Path(settings.data_processed_dir)

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        raw_csv_path = raw_dir / "sample.csv"
        base_parquet_path = processed_dir / "sample_reference.parquet"
        drift_parquet_path = processed_dir / "sample_current_drifted.parquet"
        quality_report_path = processed_dir / "quality_report.json"

        # Step: 1 - Load data
        real_df = load_and_clean(input_path)

        # Step: 2 - Configure metadata
        metadata = configure_metadata(real_df)

        # Step: 3 - Train synthesizer
        synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            numerical_distributions={
                "CreditScore": "beta",
                "Age": "gamma",
                "Balance": "beta",
                "EstimatedSalary": "beta",
            },
        )
        synthesizer.fit(real_df)

        # Step: 4 - Generate synthetic data
        synthetic_base = synthesizer.sample(num_rows=base_rows)

        # Step: 5 - Generate drifted synthetic data
        synthetic_drift_raw = synthesizer.sample(num_rows=drift_rows)
        synthetic_drift = apply_drift(synthetic_drift_raw, real_df)

        # Step: 6 - Save raw synthetic data
        logger.info(f"Saving raw synthetic data to {raw_csv_path}")
        FileIO().write_csv(synthetic_base, raw_csv_path)

        # Step: 7 - Process through pipeline (Optional)
        if skip_pipeline:
            logger.warning("Skipping pipeline processing (--skip-pipeline flag set)")
            logger.info(
                "Writing synthetic data directly to processed directory. This might cause data inconsistency..."
            )

            FileIO().write_parquet(synthetic_base, base_parquet_path)
            FileIO().write_parquet(synthetic_drift, drift_parquet_path)

            base_processed_path = str(base_parquet_path)
            drift_processed_path = str(drift_parquet_path)

        else:
            logger.info("Processing synthetic data through ingestion pipeline")
            base_processed_path = data_ingestion_flow(
                input_path=str(raw_csv_path), output_path=str(base_parquet_path)
            )

            # for drifted data, save temp CSV and process through pipeline
            temp_drift_csv = raw_dir / "sample_drift_temp.csv"
            FileIO().write_csv(synthetic_drift, temp_drift_csv)

            logger.info("-> Processing Drifted Synthetic data...")
            drift_processed_path = data_ingestion_flow(
                input_path=str(temp_drift_csv), output_path=str(drift_parquet_path)
            )

            # clean up temp csv file
            if temp_drift_csv.exists():
                temp_drift_csv.unlink()

            logger.info("Pipeline processing completed successfully")

        # Step: 8 - Quality evaluation
        quality_metrics = evaluate_synthetic_data_quality(
            real_df, synthetic_base, metadata
        )

        # save quality report
        with open(quality_report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "base_rows": base_rows,
                    "drift_rows": drift_rows,
                    "real_data_shape": real_df.shape,
                    "quality_metrics": quality_metrics,
                },
                f,
                indent=2,
            )

        logger.info(f"Quality report saved to {quality_report_path}")

        logger.info(
            f"Quality score: {quality_metrics.get('overall_quality_score', 'N/A')}"
        )

        return {
            "raw_csv": str(raw_csv_path),
            "base_parquet": str(base_processed_path),
            "drift_parquet": str(drift_processed_path),
            "quality_report": str(quality_report_path),
        }

    except Exception as e:
        logger.error(
            "Synthetic data generation failed",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


# CLI
def main():
    """CLI entry point"""
    args = parse_arguments()

    logger.info(
        "Starting synthetic data generation with arguments:\n"
        f"  Input: {args.input}\n"
        f"  Base rows: {args.base_rows}"
        f"  Drift rows: {args.drift_rows}"
        f"  Skip pipeline: {args.skip_pipeline}"
        f"  Output dir: {args.output_dir or 'default (from settings)'}"
    )

    output_paths = generate_synthetic_data(
        input_path=args.input,
        base_rows=args.base_rows,
        drift_rows=args.drift_rows,
        skip_pipeline=args.skip_pipeline,
        output_dir=args.output_dir,
    )

    print("\n", "=" * 80)
    print("Synthetic Data Generation Successful")
    print("\nGenerated Files:")
    for file_type, path in output_paths.items():
        print(f"    . {file_type}: {path}")
    print("\n", "=" * 80)


if __name__ == "__main__":
    main()
