"""
Churn Compass - Data validator
"""

import pandas as pd
from pandera.pandas import errors

from churn_compass.logging.logger import setup_logger
from .schemas import RAW_SCHEMA, TRAINING_SCHEMA
from .leakage import detect_leakage_columns
from .checks import check_class_imbalance, check_data_quality

logger = setup_logger(__name__)

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw input data against RAW_SCHEMA.

    :param df: Raw DataFrame
    :type df: pd.DataFrame
    :return: Validated DataFrame
    :rtype: DataFrame
    """
    try:
        logger.info("Validating raw data")
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

    :param df: Cleaned DataFrame ready for training
    :type df: pd.DataFrame
    :param fail_on_imbalance: Whether to raise error on severe class imbalance or not
    :type fail_on_imbalance: bool
    :return: Validated DataFrame
    :rtype: DataFrame
    """
    logger.info("Validating training data")

    leakage = detect_leakage_columns(df)
    if leakage:
        raise ValueError(f"Leakage columns present: {leakage}")

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
