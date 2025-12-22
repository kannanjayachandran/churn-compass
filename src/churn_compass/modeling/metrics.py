"""
Churn Compass - Evaluation Metrics

Business-focused metrics for churn prediction, emphasizing top-K targeting.

Key Metrics:
- Precision@K: Accuracy of top-K predictions
- Recall@K: Coverage of actual churners in top-K
- Lift@K: Model improvement over random selection
- PR-AUC: Overall ranking quality (area under precision-recall curve)

Business Objective:
Target top 10% of customers (K=10%) with retention campaigns.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from churn_compass import settings, setup_logger


logger = setup_logger(__name__)


# Core utilities
def _resolve_k(n_samples: int, k: Optional[int], k_percent: Optional[float]) -> int:
    if k is not None:
        return max(1, min(k, n_samples))
    pct = k_percent if k_percent is not None else settings.top_k_percent
    return max(1, int(n_samples * pct))


def topk_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: Optional[int] = None,
    k_percent: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate top-K business metrics for churn prediction.

    :param y_true: True binary labels (0=retained, 1=churned)
    :type y_true: np.ndarray
    :param y_score: Predicted churn probabilities (0.0 to 1.0)
    :type y_score: np.ndarray
    :param k: Absolute number of top predictions (overrides k_percent)
    :type k: Optional[int]
    :param k_percent: Percentage of top predictions (default: 10% from settings)
    :type k_percent: Optional[float]
    :return: Dictionary with metrics
    :rtype: Dict[str, float]
    """
    n = len(y_true)
    k_val = _resolve_k(n, k, k_percent)

    # using stable sort to avoid randomness on ties
    topk_idx = np.argsort(-y_score, kind="mergesort")[:k_val]

    tp = int(np.sum(y_true[topk_idx]))
    fp = k_val - tp
    total_pos = int(np.sum(y_true))

    precision_k = tp / k_val if k_val else 0.0
    recall_k = tp / total_pos if total_pos else 0.0

    baseline = total_pos / n if n else 0.0
    lift_k = precision_k / baseline if baseline else 0.0

    f1_k = (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if (precision_k + recall_k) > 0
        else 0.0
    )

    metrics = {
        "precision_at_k": precision_k,
        "recall_at_k": recall_k,
        "lift_at_k": lift_k,
        "f1_at_k": f1_k,
        "tp_at_k": tp,
        "fp_at_k": fp,
        "k": k_val,
        "k_percent": k_val / n,
    }

    logger.debug("Top-K metrics computed", extra=metrics)

    return metrics


# Ranking metrics
def calculate_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return float(auc(recall, precision))


# Aggregate evaluation
def calculate_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    k_percent: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    :param y_true: True binary labels
    :type y_true: np.ndarray
    :param y_scores: Predicted probabilities
    :type y_scores: np.ndarray
    :param threshold: Classification threshold (default: 0.5)
    :type threshold: float
    :param k_percent: Top-K percentage (default: from settings)
    :type k_percent: Optional[float]
    :return: Dictionary with all metrics
    :rtype: Dict[str, float]
    """
    y_pred = (y_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = 0.0

    pr_auc = calculate_pr_auc(y_true, y_scores)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    topk = topk_metrics(y_true, y_scores, k_percent=k_percent)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positives": int(np.sum(y_true)),
        "churn_rate": float(np.mean(y_true)),
        **{f"top_{k}": v for k, v in topk.items()},
    }

    logger.info(
        "Evaluation metrics calculated",
        extra={
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_at_k": topk["precision_at_k"],
            "recall_at_k": topk["recall_at_k"],
            "lift_at_k": topk["lift_at_k"],
        },
    )

    return metrics


# Optimization Objective
def joint_objective_optuna(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pr_auc_weight: float = 0.5,
    recall_k_weight: float = 0.5,
    k_percent: Optional[float] = None,
) -> float:
    """
    Joint objective function for Optuna hyperparameter optimization. Combines PR-AUC with Recall@TopK

    Formula: objective = (pr_auc_weight × PR-AUC) + (recall_k_weight × Recall@K)

    :param y_true: True binary labels
    :type y_true: np.ndarray
    :param y_scores: Predicted probabilities
    :type y_scores: np.ndarray
    :param pr_auc_weight: Weight for PR-AUC (default: 0.5)
    :type pr_auc_weight: float
    :param recall_k_weight: Weight for Recall@K
    :type recall_k_weight: float
    :param k_percent: Top-K percentage (default: from settings)
    :type k_percent: Optional[float]
    :return: Combined objective score (higher is better)
    :rtype: float

    Note:
        Weights should sum to 1.0 for interpretability
    """
    pr_auc = calculate_pr_auc(y_true, y_scores)
    recall_k = topk_metrics(y_true, y_scores, k_percent=k_percent)["recall_at_k"]
    objective = (pr_auc_weight * pr_auc) + (recall_k_weight * recall_k)

    return float(objective)


# segment level evaluation
def calculate_metrics_by_segment(
    df: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    segment_col: str,
    k_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate metrics for different customer segments.

    :param df: DataFrame with predictions and segments
    :type df: pd.DataFrame
    :param y_true_col: Column name for true labels
    :type y_true_col: str
    :param y_score_col: Column name for predicted scores
    :type y_score_col: str
    :param segment_col: Column name for segmentation
    :type segment_col: str
    :param k_percent: Top-K percentage
    :type k_percent: Optional[float]
    :return: DataFrame with metrics per segment
    :rtype: DataFrame
    """
    rows = []

    for value in df[segment_col].dropna().unique():
        segment = df[df[segment_col] == value]
        metrics = calculate_all_metrics(
            segment[y_true_col].values, segment[y_score_col].values, k_percent=k_percent
        )

        rows.append(
            {
                "segment": value,
                "n_samples": len(segment),
                "churn_rates": metrics["churn_rate"],
                "pr_auc": metrics["pr_auc"],
                "precision_at_k": metrics["topk_precision_at_k"],
                "recall_at_k": metrics["topk_recall_at_k"],
                "lift_at_k": metrics["topk_lift_at_k"],
            }
        )
    result = pd.DataFrame(rows)

    logger.info("Segment metrics computed", extra={"segments": len(result)})

    return result
