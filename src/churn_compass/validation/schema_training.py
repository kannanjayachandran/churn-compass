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

from typing import List, Set
import pandas as pd
from pandera.pandas import Column, DataFrameSchema, Check, errors

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger


logger = setup_logger(__name__)


# Raw data schema
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

# Training data schema
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
        "Geography": Column(str, checks=Check.isin(["France", "Spain", "Germany"])),
        "Gender": Column(str, checks=Check.isin(["Male", "Female"])),
        "Age": Column(int, nullable=False),
        "Tenure": Column(int, nullable=False),
        "Balance": Column(float, nullable=False),
        "NumOfProducts": Column(int, nullable=False),
        "HasCrCard": Column(int, nullable=False),
        "IsActiveMember": Column(int, nullable=False),
        "EstimatedSalary": Column(float, nullable=False),
        "Exited": Column(int, checks=Check.isin([0, 1])),
    },
    strict=True,  # Extra columns not allowed
    coerce=True,
)


# Leakage detection
def detect_leakage_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect potential data leakage columns in DataFrame

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: List of column names that may cause leakage
    :rtype: List[str]
    """
    leakage_cols: Set[str] = set()

    leakage_cols.update(col for col in settings.leakage_columns if col in df.columns)
    leakage_cols.update(col for col in settings.id_columns if col in df.columns)

    # # check against configured leakage columns
    # for col in settings.leakage_columns:
    #     if col in df.columns:
    #         leakage_cols.append(col)

    # # check against ID columns
    # for col in settings.id_columns:
    #     if col in df.columns:
    #         leakage_cols.append(col)

    # Block engineered prediction-like columns
    for col in df.columns:
        if col.startswith(("pred", "shap")):
            leakage_cols.add(col)

    if leakage_cols:
        logger.warning(
            f"Detected potential leakage columns: {leakage_cols}",
            extra={"leakage_columns": sorted(leakage_cols)},
        )

    return sorted(leakage_cols)


# Validation functions
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
        logger.info("Validating raw data")
        validated = RAW_SCHEMA.validate(df, lazy=True)

        logger.info(
            "Raw data validation passed",
            extra={"rows": len(validated), "columns": len(validated.columns)},
        )

        return validated
    
    except errors.SchemaError:
        logger.error("Raw data validation failed", exc_info=True)
        raise


def validate_training_data(df: pd.DataFrame, fail_on_imbalance: bool = True) -> pd.DataFrame:
    """
    Validate cleaned training data against TRAINING_SCHEMA

    :param df: Cleaned DataFrame ready for training
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    :raise: Pandera.pandas.errors.SchemaError: If validation fails
    """
    try:
        logger.info("Validating training data")

        leakage_cols = detect_leakage_columns(df)
        if leakage_cols:
            raise ValueError(
                f"Found leakage columns in training data: {leakage_cols}"
            )

        # validate against training schema
        validated = TRAINING_SCHEMA.validate(df, lazy=True)

        # Additional business logic checks
        _check_class_imbalance(validated, fail=fail_on_imbalance)
        _check_data_quality(validated)

        logger.info(
            "Training data validation passed",
            extra={
                "rows": len(validated),
                "columns": len(validated.columns),
                "churn_rate": validated["Exited"].mean(),
            },
        )
        return validated

    except Exception as e:
        logger.error(
            "Training data validation failed",
            extra={"error_type": type(e).__name__},
            exc_info=True,
        )
        raise


# Business Checks
def _check_class_imbalance(df: pd.DataFrame, min_minority_pct: float = 0.05, fail: bool = True) -> None:
    """
    Check if target variable has sufficient minority class representation.

    :param df: DataFrame with 'Exited' target column
    :type df: pd.DataFrame
    :param min_minority_pct: Minimum percentage for minority class
    :type min_minority_pct: float
    :param fail: whether to fail if data imbalance is higher than set threshold
    :type fail: bool
    :raise: ValueError: If minority class is below threshold
    """
    churn_rate = df["Exited"].mean()
    minority_pct = min(churn_rate, 1 - churn_rate)

    if minority_pct < min_minority_pct:
        msg = (
            f"Severe class imbalance detected: minority={minority_pct:.2%}, "
            f"threshold={min_minority_pct:.2%}"
        )
        logger.warning(
            msg,
            extra={"churn_rate": churn_rate,},
        )
        if fail:
            raise ValueError

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
    problematic = missing_pct[missing_pct > max_missing_pct]

    if not problematic.empty:
        logger.error(
            f"Excessive missing values detected",
            extra={"problematic_columns": problematic.to_dict()},
        )
        raise ValueError(f"Columns exceed missing threshold: {list(problematic.index)}")

    # check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(
            "Duplicate rows detected",
            extra={
                "n_duplicates": n_duplicates,
            },
        )
    logger.info(
        "Data quality checks passed",
        extra={"max_missing_pct": missing_pct.max(), "n_duplicates": int(n_duplicates)},
    )
