"""
Tests for normalization utilities.

Tests string and DataFrame normalization contracts.
"""

import pandas as pd

from churn_compass.utils.normalization import normalize_string, normalize_dataframe


# normalize_string tests
def test_normalize_string_title_case():
    """Strings should be converted to title case."""
    assert normalize_string("GOLD") == "Gold"
    assert normalize_string("silver") == "Silver"
    assert normalize_string("DiAmOnD") == "Diamond"


def test_normalize_string_strips_whitespace():
    """Leading and trailing whitespace should be stripped."""
    assert normalize_string("  gold  ") == "Gold"
    assert normalize_string("\tsilver\n") == "Silver"


def test_normalize_string_preserves_non_strings():
    """Non-string values should be returned unchanged."""
    assert normalize_string(123) == 123
    assert normalize_string(None) is None


def test_normalize_string_empty():
    """Empty string should return empty string."""
    assert normalize_string("") == ""


# normalize_dataframe tests
def test_normalize_dataframe_column_rename():
    """Column mapping should rename columns correctly."""
    df = pd.DataFrame(
        {
            "Card Type": ["Gold", "Silver"],
            "Other": [1, 2],
        }
    )

    result = normalize_dataframe(df, column_mapping={"Card Type": "CardType"})

    assert "CardType" in result.columns
    assert "Card Type" not in result.columns


def test_normalize_dataframe_categorical_values():
    """Default categorical columns should be normalized."""
    df = pd.DataFrame(
        {
            "CardType": ["GOLD", "silver", "  Diamond  "],
            "Geography": ["france", "GERMANY", "spain"],
            "Gender": ["MALE", "female", "Male"],
        }
    )

    result = normalize_dataframe(df)

    assert list(result["CardType"]) == ["Gold", "Silver", "Diamond"]
    assert list(result["Geography"]) == ["France", "Germany", "Spain"]
    assert list(result["Gender"]) == ["Male", "Female", "Male"]


def test_normalize_dataframe_preserves_other_columns():
    """Non-categorical columns should be unchanged."""
    df = pd.DataFrame(
        {
            "CardType": ["GOLD"],
            "CreditScore": [650],
            "Balance": [50000.0],
        }
    )

    result = normalize_dataframe(df)

    assert result["CreditScore"].iloc[0] == 650
    assert result["Balance"].iloc[0] == 50000.0


def test_normalize_dataframe_custom_categorical_cols():
    """Custom categorical columns list should be respected."""
    df = pd.DataFrame(
        {
            "CardType": ["GOLD"],
            "CustomField": ["VALUE"],
            "Geography": ["FRANCE"],
        }
    )

    # Only normalize CustomField
    result = normalize_dataframe(df, categorical_cols=["CustomField"])

    assert result["CustomField"].iloc[0] == "Value"  # Normalized
    assert result["CardType"].iloc[0] == "GOLD"  # Not normalized
    assert result["Geography"].iloc[0] == "FRANCE"  # Not normalized


def test_normalize_dataframe_missing_columns_handled():
    """Missing categorical columns should be handled gracefully."""
    df = pd.DataFrame(
        {
            "CardType": ["GOLD"],
        }
    )

    # Request normalization including non-existent columns
    result = normalize_dataframe(df, categorical_cols=["CardType", "NonExistent"])

    assert result["CardType"].iloc[0] == "Gold"


def test_normalize_dataframe_returns_copy():
    """normalize_dataframe should not mutate the original DataFrame."""
    df = pd.DataFrame(
        {
            "CardType": ["GOLD"],
        }
    )
    original_value = df["CardType"].iloc[0]

    normalize_dataframe(df)

    assert df["CardType"].iloc[0] == original_value  # Still "GOLD"
