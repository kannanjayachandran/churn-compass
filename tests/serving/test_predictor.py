import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from churn_compass.serving.predictor import ChurnPredictor


# Test fixtures
class DummyModel(BaseEstimator, ClassifierMixin):
    """Stateless sklearn-compatible dummy classifier"""

    def fit(self, X, y=None):
        # No training needed
        return self

    def predict_proba(self, X):
        n = len(X)
        probs = np.linspace(0.1, 0.9, n)
        return np.vstack([1 - probs, probs]).T

    def __sklearn_is_fitted__(self):
        # Explicitly tell sklearn this model is always "fitted"
        return True


@pytest.fixture
def dummy_pipeline():
    return Pipeline([
        ("model", DummyModel())
    ])



@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "CreditScore": [600, 700, 800],
            "Age": [30, 45, 60],
            "Balance": [10000, 50000, 90000],
        }
    )


# Tests
def test_predict_single(dummy_pipeline):
    predictor = ChurnPredictor(model=dummy_pipeline)

    result = predictor.predict_single(
        {
            "CreditScore": 650,
            "Age": 40,
            "Balance": 30000,
        }
    )

    assert "probability" in result
    assert "prediction" in result
    assert "risk_level" in result
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_batch_shape(dummy_pipeline, sample_df):
    predictor = ChurnPredictor(model=dummy_pipeline)

    results = predictor.predict_batch(sample_df)

    assert len(results) == len(sample_df)
    assert set(results.columns) == {
        "probability",
        "prediction",
        "risk_level",
    }


def test_predict_batch_empty_raises(dummy_pipeline):
    predictor = ChurnPredictor(model=dummy_pipeline)

    with pytest.raises(ValueError):
        predictor.predict_batch(pd.DataFrame())


def test_risk_level_boundaries():
    f = ChurnPredictor._get_risk_level

    assert f(0.1) == "low"
    assert f(0.4) == "medium"
    assert f(0.69) == "medium"
    assert f(0.7) == "high"
    assert f(0.95) == "high"


def test_top_k_customers(dummy_pipeline, sample_df):
    predictor = ChurnPredictor(model=dummy_pipeline)

    top_k = predictor.get_top_k_customers(sample_df, k=2)

    assert len(top_k) == 2
    assert top_k["probability"].is_monotonic_decreasing
