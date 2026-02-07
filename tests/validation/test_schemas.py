"""
Tests for validation schemas.

Tests behavioral contracts of Pandera schema validation, not Pandera internals.
"""

import pytest
import pandas as pd

from churn_compass.validation.schemas import RAW_SCHEMA, TRAINING_SCHEMA


@pytest.fixture
def valid_raw_customer():
    """Minimal valid raw customer record."""
    return pd.DataFrame(
        [
            {
                "RowNumber": 1,
                "CustomerId": 12345678,
                "Surname": "Smith",
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0,
                "CardType": "Gold",
                "Exited": 0,
            }
        ]
    )


def test_raw_schema_accepts_valid_data(valid_raw_customer):
    """Valid data should pass raw schema validation."""
    validated = RAW_SCHEMA.validate(valid_raw_customer)
    assert len(validated) == 1


def test_raw_schema_rejects_invalid_creditscore(valid_raw_customer):
    """Credit score outside 300-850 range should fail."""
    valid_raw_customer.loc[0, "CreditScore"] = 250  # Below 300

    with pytest.raises(Exception):  # SchemaError
        RAW_SCHEMA.validate(valid_raw_customer)


def test_raw_schema_rejects_creditscore_above_max(valid_raw_customer):
    """Credit score above 850 should fail."""
    valid_raw_customer.loc[0, "CreditScore"] = 900  # Above 850

    with pytest.raises(Exception):
        RAW_SCHEMA.validate(valid_raw_customer)


def test_raw_schema_rejects_invalid_geography(valid_raw_customer):
    """Geography must be France, Spain, or Germany."""
    valid_raw_customer.loc[0, "Geography"] = "Italy"

    with pytest.raises(Exception):
        RAW_SCHEMA.validate(valid_raw_customer)


def test_raw_schema_rejects_invalid_gender(valid_raw_customer):
    """Gender must be Male or Female."""
    valid_raw_customer.loc[0, "Gender"] = "Other"

    with pytest.raises(Exception):
        RAW_SCHEMA.validate(valid_raw_customer)


def test_raw_schema_rejects_invalid_cardtype(valid_raw_customer):
    """CardType must be Silver, Gold, Diamond, or Platinum."""
    valid_raw_customer.loc[0, "CardType"] = "Bronze"

    with pytest.raises(Exception):
        RAW_SCHEMA.validate(valid_raw_customer)


def test_raw_schema_rejects_negative_balance(valid_raw_customer):
    """Balance must be non-negative."""
    valid_raw_customer.loc[0, "Balance"] = -1000.0

    with pytest.raises(Exception):
        RAW_SCHEMA.validate(valid_raw_customer)


def test_raw_schema_rejects_invalid_age(valid_raw_customer):
    """Age must be between 18 and 100."""
    valid_raw_customer.loc[0, "Age"] = 5  # Too young

    with pytest.raises(Exception):
        RAW_SCHEMA.validate(valid_raw_customer)


def test_training_schema_excludes_id_columns():
    """Training schema should not include RowNumber, CustomerId, Surname."""
    assert "RowNumber" not in TRAINING_SCHEMA.columns
    assert "CustomerId" not in TRAINING_SCHEMA.columns
    assert "Surname" not in TRAINING_SCHEMA.columns


def test_training_schema_includes_feature_columns():
    """Training schema should include all feature columns."""
    expected_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "CardType",
        "Exited",
    ]

    for col in expected_cols:
        assert col in TRAINING_SCHEMA.columns, f"Missing column: {col}"
