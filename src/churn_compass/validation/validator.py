"""
Churn Compass - Data validator

Validates datasets at different pipeline stages using Pandera schemas.
"""

import pandas as pd
from pandera.pandas import errors

from churn_compass import setup_logger

from .checks import check_class_imbalance, check_data_quality
from .leakage import detect_leakage_columns
from .schemas import RAW_SCHEMA, TRAINING_SCHEMA

logger = setup_logger(__name__)


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw input data against RAW_SCHEMA.

    Applied after CSV ingestion, before or after customer_uuid generation.

    :param df: Raw DataFrame
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    :raises pandera.errors.SchemaError: If validation fails
    """
    try:
        logger.info("Validating raw data schema")
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Raw validation passed", extra={"rows": len(validated)})
        return validated

    except errors.SchemaError:
        logger.exception("Raw data validation failed", exc_info=True)
        raise


def validate_training_data(
    df: pd.DataFrame,
    fail_on_imbalance: bool = True,
) -> pd.DataFrame:
    """
    Validate cleaned training data against TRAINING_SCHEMA

    Applied before model training. Ensures:
        - Schema compliance (correct columns, types, ranges)
        - customer_uuid present and unique
        - No leakage columns
        - Acceptable class balance
        - Data quality threshold met

    :param df: Cleaned DataFrame ready for training
    :type df: pd.DataFrame
    :param fail_on_imbalance: Whether to raise error on severe class imbalance
    :type fail_on_imbalance: bool
    :return: Validated DataFrame
    :rtype: DataFrame
    :raises ValueError: If leakage columns detected or validation fails
    :raises pandera.errors.SchemaError: If schema validation fails
    """
    logger.info("Validating training data")

    if "customer_uuid" not in df.columns:
        raise ValueError("customer_uuid column required for training data validation")

    leakage = detect_leakage_columns(df)
    if leakage:
        raise ValueError(f"Leakage columns detected: {leakage}")

    try:
        validated = TRAINING_SCHEMA.validate(df, lazy=True)
    except errors.SchemaError:
        logger.exception("Training schema validation failed")
        raise

    check_class_imbalance(validated, fail=fail_on_imbalance)
    check_data_quality(validated)

    logger.info(
        "Training validation passed",
        extra={
            "rows": len(validated),
            "churn_rate": validated["Exited"].mean(),
        },
    )
    return validated
