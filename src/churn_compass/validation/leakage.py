"""
Churn Compass - Leakage detection

Detecting leakage columns (No mutation. No validation.)
"""

from typing import List, Set
import pandas as pd
from churn_compass import settings, setup_logger


logger = setup_logger(__name__)


def detect_leakage_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect potential data leakage columns in DataFrame

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: List of column names that may cause leakage
    :rtype: List[str]
    """
    leakage: Set[str] = set()

    leakage.update(c for c in settings.leakage_columns if c in df.columns)
    leakage.update(c for c in settings.id_columns if c in df.columns)
    leakage.update(c for c in settings.misleading_columns if c in df.columns)

    for col in df.columns:
        if col.startswith(("pred", "shap")):
            leakage.add(col)

    if leakage:
        logger.warning(
            "Potential leakage columns detected",
            extra={"leakage_columns": sorted(leakage)},
        )

    return sorted(leakage)
