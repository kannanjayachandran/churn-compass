"""
Churn Compass - Business logic checks

Validates dataset quality and class distribution
"""

from typing import Optional

import pandas as pd

from churn_compass import get_settings, setup_logger

logger = setup_logger(__name__)


def check_class_imbalance(
    df: pd.DataFrame,
    min_minority_pct: Optional[float] = None,
    fail: bool = True,
) -> None:
    """
    Check for severe class imbalance in target variable.

    :param df: Input DataFrame with 'Exited' column
    :type df: pd.DataFrame
    :param min_minority_pct: Minimum acceptable minority class percentage (default: from settings)
    :type min_minority_pct: float
    :param fail: Whether to raise ValueError on threshold violation
    :type fail: bool
    :raises ValueError: If minority class below threshold and fail=True
    """
    settings = get_settings()
    threshold = (
        min_minority_pct
        if min_minority_pct is not None
        else settings.min_minority_class_pct
    )

    churn_rate = df["Exited"].mean()
    minority_pct = min(churn_rate, 1 - churn_rate)

    if minority_pct < threshold:
        msg = (
            f"Severe class imbalance: minority={minority_pct:.2%}, "
            f"threshold={min_minority_pct:.2%}"
        )
        logger.warning(msg, extra={"churn_rate": churn_rate})
        if fail:
            raise ValueError(msg)

    logger.info(
        "Class balance OK",
        extra={"churn_rate": churn_rate, "minority_pct": minority_pct},
    )


def check_data_quality(
    df: pd.DataFrame, max_missing_pct: Optional[float] = None
) -> None:
    """
    Check data quality metrics (missing values, duplicates)

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param max_missing_pct: Maximum allowed percentage of missing values per column (default: from settings)
    :type max_missing_pct: float
    :raises ValueError: If any column exceeds missing value threshold
    """
    settings = get_settings()
    threshold = (
        max_missing_pct if max_missing_pct is not None else settings.max_missing_pct
    )

    missing_pct = df.isnull().mean()
    problematic = missing_pct[missing_pct > threshold]

    if not problematic.empty:
        logger.error(
            "Excessive missing values",
            extra={"problematic_columns": problematic.to_dict()},
        )
        raise ValueError(
            f"Missing values threshold exceeded: {list(problematic.index)}"
        )

    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning("Duplicate rows detected", extra={"count": n_duplicates})

    logger.info(
        "Data quality OK",
        extra={"max_missing_pct": float(missing_pct.max())},
    )
