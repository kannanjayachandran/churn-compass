"""
Churn Compass - Business logic checks

Checks dataset for class imbalance and duplicate rows
"""

import pandas as pd
from churn_compass.logging.logger import setup_logger

logger = setup_logger(__name__)

def check_class_imbalance(
    df: pd.DataFrame,
    min_minority_pct: float = 0.05,
    fail: bool = True,
) -> None:
    """
    Check class imbalance
    
    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param min_minority_pct: Minimum percentage for minority class
    :type min_minority_pct: float
    :param fail: Whether to fail if data imbalance is higher than set threshold
    :type fail: bool
    """
    churn_rate = df["Exited"].mean()
    minority_pct = min(churn_rate, 1 - churn_rate)

    if minority_pct < min_minority_pct:
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


def check_data_quality(df: pd.DataFrame, max_missing_pct: float = 0.05) -> None:
    """
    Check data quality metrics.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param max_missing_pct: Maximum allowed percentage of missing values per column
    :type max_missing_pct: float
    """
    missing_pct = df.isnull().mean()
    problematic = missing_pct[missing_pct > max_missing_pct]

    if not problematic.empty:
        logger.error(
            "Excessive missing values",
            extra={"problematic_columns": problematic.to_dict()},
        )
        raise ValueError(f"Missing threshold exceeded: {list(problematic.index)}")

    n_duplicates = df.duplicated().sum()
    if n_duplicates:
        logger.warning("Duplicate rows detected", extra={"count": n_duplicates})

    logger.info(
        "Data quality OK",
        extra={"max_missing_pct": float(missing_pct.max())},
    )
