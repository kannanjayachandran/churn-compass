"""
Tests for business logic checks.

Tests class imbalance detection and data quality validation contracts.
"""

import pytest
import pandas as pd

from churn_compass.validation.checks import check_class_imbalance, check_data_quality


@pytest.fixture
def balanced_data():
    """DataFrame with ~20% churn rate (balanced)."""
    # 80 retained, 20 churned = 20% churn
    return pd.DataFrame(
        {
            "Exited": [0] * 80 + [1] * 20,
            "CreditScore": [650] * 100,
        }
    )


@pytest.fixture
def imbalanced_data():
    """DataFrame with 1% churn rate (severe imbalance)."""
    # 99 retained, 1 churned = 1% churn
    return pd.DataFrame(
        {
            "Exited": [0] * 99 + [1] * 1,
            "CreditScore": [650] * 100,
        }
    )


def test_check_class_imbalance_passes_balanced_data(balanced_data):
    """Balanced data (20% churn) should pass default threshold (5%)."""
    # Should not raise
    check_class_imbalance(balanced_data, min_minority_pct=0.05, fail=True)


def test_check_class_imbalance_fails_extreme_imbalance(imbalanced_data):
    """Severely imbalanced data (1% churn) should fail with default threshold."""
    with pytest.raises(ValueError, match="Severe class imbalance"):
        check_class_imbalance(imbalanced_data, min_minority_pct=0.05, fail=True)


def test_check_class_imbalance_warn_only_mode(imbalanced_data):
    """Warning-only mode should not raise even with severe imbalance."""
    # Should not raise when fail=False
    check_class_imbalance(imbalanced_data, min_minority_pct=0.05, fail=False)


def test_check_class_imbalance_custom_threshold(balanced_data):
    """Custom threshold should be respected."""
    # 20% minority is below 25% threshold - should fail
    with pytest.raises(ValueError):
        check_class_imbalance(balanced_data, min_minority_pct=0.25, fail=True)


def test_check_data_quality_passes_clean_data():
    """Clean data with no missing values should pass."""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    # Should not raise
    check_data_quality(df, max_missing_pct=0.05)


def test_check_data_quality_fails_excessive_nulls():
    """Data with >5% nulls should fail."""
    df = pd.DataFrame(
        {
            "A": [1, None, None, None, 5],  # 60% missing
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    with pytest.raises(ValueError, match="Missing threshold exceeded"):
        check_data_quality(df, max_missing_pct=0.05)


def test_check_data_quality_custom_threshold():
    """Custom missing threshold should be respected."""
    df = pd.DataFrame(
        {
            "A": [1, None, 3, 4, 5],  # 20% missing
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # Should pass at 25% threshold
    check_data_quality(df, max_missing_pct=0.25)

    # Should fail at 10% threshold
    with pytest.raises(ValueError):
        check_data_quality(df, max_missing_pct=0.10)


def test_check_data_quality_handles_duplicates():
    """Data with duplicates should pass quality check (with warning logged)."""
    df = pd.DataFrame(
        {
            "A": [1, 1, 2, 3, 4],
            "B": [1.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    # Should not raise - duplicates trigger a warning but don't fail
    check_data_quality(df)
