"""
Churn Compass - Leakage detection

Identifies columns that would cause data leakage if used as features.
"""

from typing import List, Set

import pandas as pd

from churn_compass import get_settings, setup_logger

logger = setup_logger(__name__)

# Identity columns that are preserved for mapping but never used as features
IDENTITY_COLUMNS = {"customer_uuid"}


def detect_leakage_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect potential data leakage columns in DataFrame

    Leakage sources:
        - Configured leakage columns (`Complain`, `Satisfaction Score`)
        - PII columns (CustomerId, Surname, RowNumber)
        - Prediction artifacts (columns starting with 'pred', 'shap')

    Note: Identity column (customer_uuid) is NOT leakage - they're for mapping, not features.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: List of column names that may cause leakage
    :rtype: List[str]
    """
    settings = get_settings()
    leakage: Set[str] = set()

    leakage.update(c for c in settings.leakage_columns if c in df.columns)

    pii_cols = set(settings.id_columns) - IDENTITY_COLUMNS
    leakage.update(c for c in pii_cols if c in df.columns)

    for col in df.columns:
        if col.startswith(("pred", "shap")):
            leakage.add(col)

    if leakage:
        logger.warning(
            "Potential leakage columns detected",
            extra={"leakage_columns": sorted(leakage)},
        )

    return sorted(leakage)
