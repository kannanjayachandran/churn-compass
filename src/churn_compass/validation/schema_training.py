"""
Churn Compass - Data Validation Schema

Pandera schemas for validating datasets at various pipeline stages.

Features:
- Type checking and constraints
- Missing value validation
- Business rule enforcement
- Automatic leakage column detection
- Custom validation checks
"""

from typing import List
import pandas as pd
from pandera.pandas import Column, DataFrameSchema, Check, errors

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger


logger = setup_logger(__name__)


# Raw data schema - validates incoming data before cleaning
RAW_SCHEMA = DataFrameSchema(
    columns={
        "RowNumber": Column(int, nullable=False),
        "CustomerId": Column(int, nullable=False),
        "Surname": Column(str, nullable=False),
        "CreditScore": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(300),
                Check.less_than_or_equal_to(850),
            ],
        ),
        "Geography": Column(
            str, nullable=False, checks=Check.isin(["France", "Spain", "Germany"])
        ),
        "Gender": Column(str, nullable=False, checks=Check.isin(["Male", "Female"])),
        "Age": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(18),
                Check.less_than_or_equal_to(100),
            ],
        ),
        "Tenure": Column(
            int,
            nullable=False,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(10)],
        ),
        "Balance": Column(
            float, nullable=False, checks=Check.greater_than_or_equal_to(0)
        ),
        "NumOfProducts": Column(
            int,
            nullable=False,
            checks=[Check.greater_than_or_equal_to(1), Check.less_than_or_equal_to(4)],
        ),
        "HasCrCard": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "IsActiveMember": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "EstimatedSalary": Column(
            float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(200000),
            ],
        ),
        "Exited": Column(
            int,
            nullable=False,
            checks=Check.isin([0, 1]),
            description="Target variable: 1 = customer churned, 0 = retained",
        ),
    },
    strict=False,  # Allow extra columns (Complain, Satisfaction Score) in raw data
    coerce=True,  # Attempt type coercion
)

# Training data schema - after cleaning and feature engineering
TRAINING_SCHEMA = DataFrameSchema(
    columns={
        "CreditScore": Column(
            int,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(300),
                Check.less_than_or_equal_to(850),
            ],
        ),
        "Geography": Column(str, nullable=False),
        "Gender": Column(str, nullable=False),
        "Age": Column(int, nullable=False),
        "Tenure": Column(int, nullable=False),
        "Balance": Column(float, nullable=False),
        "NumOfProducts": Column(int, nullable=False),
        "HasCrCard": Column(int, nullable=False),
        "IsActiveMember": Column(int, nullable=False),
        "EstimatedSalary": Column(float, nullable=False),
        "Exited": Column(int, nullable=False),
    },
    strict=True,  # Here don't allow extra columns
    coerce=True,
)


def detect_leakage_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect potential data leakage columns in DataFrame

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: List of column names that may cause leakage
    :rtype: List[str]
    """
    leakage_cols = []

    # check against configured leakage columns
    for col in settings.leakage_columns:
        if col in df.columns:
            leakage_cols.append(col)

    # check against ID columns
    for col in settings.id_columns:
        if col in df.columns:
            leakage_cols.append(col)

    # Block engineered prediction-like columns
    for col in df.columns:
        if col.startswith("pred") or col.startswith("shap"):
            leakage_cols.append(col)

    if leakage_cols:
        logger.warning(
            f"Detected potential leakage columns: {leakage_cols}",
            extra={"leakage_columns": leakage_cols},
        )

    return leakage_cols


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw input data against RAW_SCHEMA.

    :param df: Raw DataFrame
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    :raise: Pandera.pandas.errors.SchemaError: If validation fails
    """
    try:
        logger.info("Validating raw data...")
        validated_df = RAW_SCHEMA.validate(df, lazy=True)

        logger.info(
            "Raw data validation passed",
            extra={"rows": len(validated_df), "columns": len(validated_df.columns)},
        )

        return validated_df
    except errors.SchemaError as e:
        logger.error("Raw data validation failed", exc_info=True)
        logger.error(f"Schema errors:\n{e}")
        raise


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate cleaned training data against TRAINING_SCHEMA

    :param df: Cleaned DataFrame ready for training
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    :raise: Pandera.pandas.errors.SchemaError: If validation fails
    """
    try:
        logger.info("Validating training data...")

        # Check for leakage columns
        leakage_cols = detect_leakage_columns(df)
        if leakage_cols:
            raise ValueError(
                f"Found leakage columns in training data: {leakage_cols}"
                f"These must be removed before training."
            )

        # validate against schema
        validated_df = TRAINING_SCHEMA.validate(df, lazy=True)

        # Additional business logic checks
        _check_class_imbalance(validated_df)
        _check_data_quality(validated_df)

        logger.info(
            "Training data validation passed",
            extra={
                "rows": len(validated_df),
                "columns": len(validated_df.columns),
                "churn_rate": validated_df["Exited"].mean(),
            },
        )

        return validated_df

    except errors.SchemaError as e:
        logger.error(
            "Training data validation failed",
            extra={"error_type": type(e).__name__},
            exc_info=True,
        )
        raise


def _check_class_imbalance(df: pd.DataFrame, min_minority_pct: float = 0.05) -> None:
    """
    Check if target variable has sufficient minority class representation.

    :param df: DataFrame with 'Exited' target column
    :type df: pd.DataFrame
    :param min_minority_pct: Minimum percentage for minority class
    :type min_minority_pct: float
    :raise: ValueError: If minority class is below threshold
    """
    churn_rate = df["Exited"].mean()
    minority_pct = min(churn_rate, 1 - churn_rate)

    if minority_pct < min_minority_pct:
        logger.warning(
            f"Severe class imbalance detected: minority class = {minority_pct:.2%}",
            extra={"churn_rate": churn_rate, "minority_pct": minority_pct},
        )
        raise ValueError(
            f"Minority class too small: {minority_pct:.2%} < {min_minority_pct:.2%}. "
            f"Consider collecting more data or adjusting thresholds"
        )

    logger.info(
        "Class balance check passed",
        extra={
            "churn_rate": f"{churn_rate:.2%}",
            "minority_pct": f"{minority_pct:.2%}",
        },
    )


def _check_data_quality(df: pd.DataFrame, max_missing_pct: float = 0.05) -> None:
    """
    Check data quality metrics.

    :param df: DataFrame to check
    :type df: pd.DataFrame
    :param max_missing_pct: Maximum allowed percentage of missing values per column
    :type max_missing_pct: float
    :raise: ValueError: If data quality issues detected
    """
    # check for missing values
    missing_pct = df.isnull().mean()
    problematic_cols = missing_pct[missing_pct > max_missing_pct]

    if not problematic_cols.empty:
        logger.error(
            f"Columns with excessive missing values: {problematic_cols.to_dict()}",
            extra={"problematic_columns": problematic_cols.to_dict()},
        )
        raise ValueError(
            f"Found columns with >{max_missing_pct:.1%} missing values: "
            f"{list(problematic_cols.index)}"
        )

    # check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(
            f"Found {n_duplicates} duplicate rows",
            extra={
                "n_duplicates": n_duplicates,
                "pct_duplicates": n_duplicates / len(df),
            },
        )
    logger.info(
        "Data quality checks passed",
        extra={"max_missing_pct": missing_pct.max(), "n_duplicates": n_duplicates},
    )
