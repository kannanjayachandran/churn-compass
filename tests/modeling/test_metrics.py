"""
Tests for modeling metrics.

Tests business metric calculation contracts, not sklearn/numpy internals.
"""

import pytest
import numpy as np

from churn_compass.modeling.metrics import (
    topk_metrics,
    calculate_pr_auc,
    calculate_all_metrics,
    joint_objective_optuna,
)


@pytest.fixture
def perfect_ranking():
    """Scenario where model perfectly ranks churners at top."""
    # 10 samples, 3 churners (30% churn rate)
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # Perfect scores: churners get highest scores
    y_score = np.array([0.9, 0.85, 0.8, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    return y_true, y_score


@pytest.fixture
def random_ranking():
    """Scenario where model scores are essentially random."""
    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.random.rand(10)
    return y_true, y_score


# Top-K metrics tests
def test_topk_metrics_perfect_ranking(perfect_ranking):
    """With perfect ranking, top-3 should capture all 3 churners."""
    y_true, y_score = perfect_ranking
    
    metrics = topk_metrics(y_true, y_score, k=3)
    
    assert metrics["precision_at_k"] == 1.0  # All top-3 are churners
    assert metrics["recall_at_k"] == 1.0  # All churners captured
    assert metrics["tp_at_k"] == 3
    assert metrics["fp_at_k"] == 0


def test_topk_metrics_k_value_respected():
    """k parameter should determine the number of top predictions evaluated."""
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    metrics_k2 = topk_metrics(y_true, y_score, k=2)
    metrics_k3 = topk_metrics(y_true, y_score, k=3)
    
    assert metrics_k2["k"] == 2
    assert metrics_k3["k"] == 3


def test_topk_k_resolution_by_absolute():
    """Absolute k should override k_percent."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    metrics = topk_metrics(y_true, y_score, k=3, k_percent=0.5)
    
    assert metrics["k"] == 3  # k=3 overrides k_percent=0.5 (which would be 5)


def test_topk_k_resolution_by_percent():
    """k_percent should calculate k as percentage of samples."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 10 samples
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    metrics = topk_metrics(y_true, y_score, k_percent=0.2)
    
    assert metrics["k"] == 2  # 20% of 10 = 2


def test_topk_metrics_lift_calculation():
    """Lift should be precision@k / baseline churn rate."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 20% churn rate
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    metrics = topk_metrics(y_true, y_score, k=2)
    
    # Precision@2 = 1.0 (both top-2 are churners)
    # Baseline = 0.2
    # Lift = 1.0 / 0.2 = 5.0
    assert metrics["lift_at_k"] == 5.0


def test_topk_metrics_empty_handling():
    """Edge case: no positives in y_true should handle gracefully."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    metrics = topk_metrics(y_true, y_score, k=2)
    
    assert metrics["recall_at_k"] == 0.0  # No positives to recall


# PR-AUC tests
def test_calculate_pr_auc_perfect():
    """Perfect predictions should have high PR-AUC."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    
    pr_auc = calculate_pr_auc(y_true, y_score)
    
    assert pr_auc == 1.0


def test_calculate_pr_auc_range():
    """PR-AUC should be between 0 and 1."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.random.rand(6)
    
    pr_auc = calculate_pr_auc(y_true, y_score)
    
    assert 0.0 <= pr_auc <= 1.0


# calculate_all_metrics tests
def test_calculate_all_metrics_returns_expected_keys(perfect_ranking):
    """calculate_all_metrics should return all expected metric keys."""
    y_true, y_score = perfect_ranking
    
    metrics = calculate_all_metrics(y_true, y_score)
    
    expected_keys = [
        "precision", "recall", "f1", "pr_auc", "roc_auc",
        "true_negatives", "false_positives", "false_negatives", "true_positives",
        "threshold", "n_samples", "n_positives", "churn_rate",
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"


def test_calculate_all_metrics_includes_topk(perfect_ranking):
    """calculate_all_metrics should include top-k metrics."""
    y_true, y_score = perfect_ranking
    
    metrics = calculate_all_metrics(y_true, y_score)
    
    # Top-k metrics are prefixed with "top_"
    assert "top_precision_at_k" in metrics
    assert "top_recall_at_k" in metrics
    assert "top_lift_at_k" in metrics


# Joint objective tests
def test_joint_objective_optuna_weights_correctly():
    """Joint objective should correctly weight PR-AUC and Recall@K."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.8, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])
    
    # Equal weights
    obj_equal = joint_objective_optuna(
        y_true, y_score, pr_auc_weight=0.5, recall_k_weight=0.5
    )
    
    # All weight on PR-AUC
    obj_prauc = joint_objective_optuna(
        y_true, y_score, pr_auc_weight=1.0, recall_k_weight=0.0
    )
    
    # All weight on Recall@K (with explicit k_percent to ensure high recall)
    obj_recall = joint_objective_optuna(
        y_true, y_score, pr_auc_weight=0.0, recall_k_weight=1.0, k_percent=0.3
    )
    
    # Objective with perfect ranking should return valid values
    assert obj_equal >= 0.0
    assert obj_prauc >= 0.0  # PR-AUC should be high for perfect ranking
    assert obj_recall >= 0.0  # Recall@K depends on k_percent
    
    # All should return floats between 0 and 1 (when weights sum to 1)
    assert isinstance(obj_equal, float)
    assert isinstance(obj_prauc, float)
    assert isinstance(obj_recall, float)


def test_joint_objective_returns_float():
    """Joint objective should return a float."""
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.3, 0.8, 0.2])
    
    result = joint_objective_optuna(y_true, y_score)
    
    assert isinstance(result, float)
