"""
Tests for feature engineering pipeline.

Tests behavioral contracts of feature transformations, not sklearn internals.
"""

import pytest
import numpy as np
import pandas as pd

from churn_compass.features.feature_engineering_pipeline import (
    FeatureEngineer,
    build_preprocessing_pipeline,
    prepare_data_for_training,
)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for feature engineering tests."""
    return pd.DataFrame(
        {
            "CreditScore": [650, 700, 750],
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Male", "Female", "Male"],
            "Age": [30, 45, 60],
            "Tenure": [3, 5, 8],
            "Balance": [50000.0, 100000.0, 0.0],
            "NumOfProducts": [2, 1, 0],
            "HasCrCard": [1, 0, 1],
            "IsActiveMember": [1, 1, 0],
            "EstimatedSalary": [75000.0, 150000.0, 50000.0],
            "CardType": ["Gold", "Platinum", "Silver"],
            "Exited": [0, 1, 0],
        }
    )


@pytest.fixture
def feature_engineer():
    """FeatureEngineer instance."""
    return FeatureEngineer()


# FeatureEngineer tests
def test_feature_engineer_creates_expected_columns(
    feature_engineer, sample_customer_data
):
    """FeatureEngineer should create all expected derived columns."""
    df = sample_customer_data.drop(columns=["Exited"])
    transformed = feature_engineer.fit_transform(df)

    expected_new_cols = [
        "balance_per_product",
        "tenure_age_ratio",
        "is_zero_balance",
        "high_value_customer",
        "age_group",
    ]

    for col in expected_new_cols:
        assert col in transformed.columns, f"Missing column: {col}"


def test_balance_per_product_zero_when_no_products(feature_engineer):
    """Balance per product should be 0 when NumOfProducts is 0 (avoid division by zero)."""
    df = pd.DataFrame(
        {
            "Balance": [50000.0],
            "NumOfProducts": [0],
            "Age": [30],
            "Tenure": [5],
            "EstimatedSalary": [75000.0],
        }
    )

    transformed = feature_engineer.transform(df)

    assert transformed["balance_per_product"].iloc[0] == 0.0


def test_balance_per_product_calculated_correctly(feature_engineer):
    """Balance per product should be Balance / NumOfProducts."""
    df = pd.DataFrame(
        {
            "Balance": [100000.0],
            "NumOfProducts": [2],
            "Age": [30],
            "Tenure": [5],
            "EstimatedSalary": [75000.0],
        }
    )

    transformed = feature_engineer.transform(df)

    assert transformed["balance_per_product"].iloc[0] == 50000.0


def test_tenure_age_ratio_zero_when_age_zero(feature_engineer):
    """Tenure/Age ratio should be 0 when Age is 0 (avoid division by zero)."""
    df = pd.DataFrame(
        {
            "Balance": [50000.0],
            "NumOfProducts": [1],
            "Age": [0],
            "Tenure": [5],
            "EstimatedSalary": [75000.0],
        }
    )

    transformed = feature_engineer.transform(df)

    assert transformed["tenure_age_ratio"].iloc[0] == 0.0


def test_is_zero_balance_flag(feature_engineer):
    """is_zero_balance should be 1 when Balance is 0, else 0."""
    df = pd.DataFrame(
        {
            "Balance": [0.0, 50000.0],
            "NumOfProducts": [1, 1],
            "Age": [30, 30],
            "Tenure": [5, 5],
            "EstimatedSalary": [75000.0, 75000.0],
        }
    )

    transformed = feature_engineer.transform(df)

    assert transformed["is_zero_balance"].iloc[0] == 1
    assert transformed["is_zero_balance"].iloc[1] == 0


def test_high_value_customer_flag(feature_engineer):
    """high_value_customer should be 1 when Balance > 100k AND EstimatedSalary > 100k."""
    df = pd.DataFrame(
        {
            "Balance": [150000.0, 50000.0, 150000.0],
            "NumOfProducts": [1, 1, 1],
            "Age": [30, 30, 30],
            "Tenure": [5, 5, 5],
            "EstimatedSalary": [150000.0, 150000.0, 50000.0],  # First is high value
        }
    )

    transformed = feature_engineer.transform(df)

    assert transformed["high_value_customer"].iloc[0] == 1  # Both > 100k
    assert transformed["high_value_customer"].iloc[1] == 0  # Balance < 100k
    assert transformed["high_value_customer"].iloc[2] == 0  # Salary < 100k


def test_age_group_binning_correct(feature_engineer):
    """Age groups should be binned correctly: young (<35), middle (35-55), senior (>55)."""
    df = pd.DataFrame(
        {
            "Balance": [50000.0, 50000.0, 50000.0],
            "NumOfProducts": [1, 1, 1],
            "Age": [30, 45, 60],
            "Tenure": [5, 5, 5],
            "EstimatedSalary": [75000.0, 75000.0, 75000.0],
        }
    )

    transformed = feature_engineer.transform(df)

    assert transformed["age_group"].iloc[0] == "young"
    assert transformed["age_group"].iloc[1] == "middle"
    assert transformed["age_group"].iloc[2] == "senior"


# Pipeline tests
def test_build_preprocessing_pipeline_returns_pipeline():
    """build_preprocessing_pipeline should return a sklearn Pipeline."""
    from sklearn.pipeline import Pipeline

    pipeline = build_preprocessing_pipeline()

    assert isinstance(pipeline, Pipeline)


def test_build_preprocessing_pipeline_has_feature_engineering():
    """Pipeline with feature engineering should have feature_engineering step."""
    pipeline = build_preprocessing_pipeline(include_feature_engineering=True)

    assert "feature_engineering" in pipeline.named_steps


def test_build_preprocessing_pipeline_without_feature_engineering():
    """Pipeline without feature engineering should not have feature_engineering step."""
    pipeline = build_preprocessing_pipeline(include_feature_engineering=False)

    assert "feature_engineering" not in pipeline.named_steps


def test_prepare_data_for_training_raises_on_missing_target(sample_customer_data):
    """prepare_data_for_training should raise ValueError if target column is missing."""
    df = sample_customer_data.drop(columns=["Exited"])

    with pytest.raises(ValueError, match="Target column .* missing"):
        prepare_data_for_training(df, target_col="Exited")


def test_prepare_data_for_training_returns_correct_shape(sample_customer_data):
    """prepare_data_for_training should return transformed arrays with correct shape."""
    X, y, pipeline, feature_names = prepare_data_for_training(sample_customer_data)

    assert len(X) == len(sample_customer_data)
    assert len(y) == len(sample_customer_data)
    assert len(feature_names) == X.shape[1]


def test_prepare_data_for_training_y_is_binary(sample_customer_data):
    """Target array y should contain only 0s and 1s."""
    X, y, pipeline, feature_names = prepare_data_for_training(sample_customer_data)

    assert set(np.unique(y)).issubset({0, 1})
