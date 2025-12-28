"""
Tests for leakage detection.

Tests that potential data leakage columns are correctly identified.
"""

import pytest
import pandas as pd

from churn_compass.validation.leakage import detect_leakage_columns


@pytest.fixture
def clean_training_data():
    """DataFrame with no leakage columns."""
    return pd.DataFrame({
        "CreditScore": [650, 700, 750],
        "Age": [35, 45, 55],
        "Balance": [50000.0, 75000.0, 100000.0],
        "Exited": [0, 1, 0],
    })


@pytest.fixture
def data_with_leakage_columns():
    """DataFrame with known leakage columns."""
    return pd.DataFrame({
        "RowNumber": [1, 2, 3],
        "CustomerId": [12345, 67890, 11111],
        "Surname": ["Smith", "Jones", "Brown"],
        "Complain": [0, 1, 0],
        "Satisfaction Score": [3, 1, 5],
        "Point Earned": [200, 100, 300],
        "CreditScore": [650, 700, 750],
        "Exited": [0, 1, 0],
    })


def test_returns_empty_for_clean_data(clean_training_data):
    """Clean data without leakage columns should return empty list."""
    leakage = detect_leakage_columns(clean_training_data)
    assert leakage == []


def test_detects_configured_leakage_columns(data_with_leakage_columns):
    """Should detect Complain and Satisfaction Score as leakage."""
    leakage = detect_leakage_columns(data_with_leakage_columns)
    
    assert "Complain" in leakage
    assert "Satisfaction Score" in leakage


def test_detects_id_columns(data_with_leakage_columns):
    """Should detect RowNumber, CustomerId, Surname as leakage."""
    leakage = detect_leakage_columns(data_with_leakage_columns)
    
    assert "RowNumber" in leakage
    assert "CustomerId" in leakage
    assert "Surname" in leakage


def test_detects_misleading_columns(data_with_leakage_columns):
    """Should detect Point Earned as misleading column."""
    leakage = detect_leakage_columns(data_with_leakage_columns)
    
    assert "Point Earned" in leakage


def test_detects_pred_prefixed_columns():
    """Should detect columns starting with 'pred' as leakage."""
    df = pd.DataFrame({
        "CreditScore": [650],
        "pred_churn": [0.8],
        "prediction_score": [0.7],
    })
    
    leakage = detect_leakage_columns(df)
    
    assert "pred_churn" in leakage
    assert "prediction_score" in leakage


def test_detects_shap_prefixed_columns():
    """Should detect columns starting with 'shap' as leakage."""
    df = pd.DataFrame({
        "CreditScore": [650],
        "shap_age": [0.05],
        "shap_balance": [-0.02],
    })
    
    leakage = detect_leakage_columns(df)
    
    assert "shap_age" in leakage
    assert "shap_balance" in leakage


def test_returns_sorted_list():
    """Leakage columns should be returned in sorted order."""
    df = pd.DataFrame({
        "Surname": ["Smith"],
        "CustomerId": [12345],
        "RowNumber": [1],
    })
    
    leakage = detect_leakage_columns(df)
    
    assert leakage == sorted(leakage)
