"""
Churn Compass - Data Normalization Utilities

Centralized logic for normalizing categorical feature values and column names.
"""

import pandas as pd
from typing import List, Optional


def normalize_string(value: str) -> str:
    """Normalize a single string to Title Case after stripping whitespace."""
    if not isinstance(value, str):
        return value
    return value.strip().lower().capitalize()


def normalize_dataframe(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    column_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Normalize a DataFrame's columns and categorical values.

    :param df: Input DataFrame
    :param categorical_cols: List of columns to apply string normalization to
    :param column_mapping: Dictionary for renaming columns (e.g., {"Card Type": "CardType"})
    :return: Normalized DataFrame
    """
    df = df.copy()

    # 1. Column renaming
    if column_mapping:
        df = df.rename(columns=column_mapping)

    # 2. Categorical value normalization
    if categorical_cols is None:
        categorical_cols = ["CardType", "Geography", "Gender"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().str.capitalize()

    return df
