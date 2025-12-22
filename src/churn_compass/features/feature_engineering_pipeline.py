"""
Churn Compass - Feature Engineering Pipeline

Preprocessing pipeline using scikit-learn transformers. Includes scaling, encoding, and feature engineering.

Design Principles:
- Use ColumnTransformer for modular preprocessing
- OneHotEncoder with handle_unknown='ignore', drop='first' (avoid dummy trap)
- StandardScaler for numeric features
- All transformers fitted only on training data
- Pipeline is serializable with MLflow
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from churn_compass import setup_logger

logger = setup_logger(__name__)


# Feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for deterministic feature engineering.

    Creates additional features:
    - balance_per_product: Balance / NumOfProducts
    - tenure_age_ratio: Tenure / Age
    - is_zero_balance: Binary indicator for zero balance
    - high_value_customer: Binary indicator (Balance > 100k & EstimatedSalary > 100k)
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Required for scikit-learn API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features

        :param X: Input DF
        :type X: pd.DataFrame
        :return: DataFrame with additional features
        :rtype: DataFrame
        """
        logger.info("Starting feature engineering")
        X = X.copy()

        X["balance_per_product"] = np.where(
            X["NumOfProducts"] > 0, X["Balance"] / X["NumOfProducts"], 0.0
        )

        X["tenure_age_ratio"] = np.where(X["Age"] > 0, X["Tenure"] / X["Age"], 0.0)

        X["is_zero_balance"] = (X["Balance"] == 0).astype(int)
        X["high_value_customer"] = (
            (X["Balance"] > 100_000) & (X["EstimatedSalary"] > 100_000)
        ).astype(int)

        X["age_group"] = pd.cut(
            X["Age"],
            bins=[0, 35, 55, 100],
            labels=["young", "middle", "senior"],
        )

        logger.info(
            "Feature engineering completed",
            extra={"total_feature_count": len(X.columns)},
        )

        return X


# Feature configuration
BASE_NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "HasCrCard",
    "IsActiveMember",
]

BASE_CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "CardType"
]

ENGINEERED_NUMERIC_FEATURES = [
    "balance_per_product",
    "tenure_age_ratio",
    "is_zero_balance",
    "high_value_customer",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "age_group",
]


# Pipeline
def build_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Preprocessing Pipeline (Scikit-learn)

    :param include_feature_engineering: Whether to include custom features engineering
    :type include_feature_engineering: bool
    :return: Scikit-learn Pipeline ready for fitting
    :rtype: Pipeline
    """
    logger.info("Starting preprocessing pipeline")

    numeric_features = BASE_NUMERIC_FEATURES.copy()
    categorical_features = BASE_CATEGORICAL_FEATURES.copy()

    if include_feature_engineering:
        numeric_features += ENGINEERED_NUMERIC_FEATURES
        categorical_features += ENGINEERED_CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    steps = []
    if include_feature_engineering:
        steps.append(("feature_engineering", FeatureEngineer()))
    steps.append(("preprocessor", preprocessor))

    pipeline = Pipeline(steps)

    logger.info(
        "Preprocessing Pipeline built",
        extra={
            "numeric_features": len(numeric_features),
            "categorical_features": len(categorical_features),
            "feature_engineering": include_feature_engineering,
        },
    )

    return pipeline


# Utilities
def extract_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Extract feature names from a fitted pipeline.

    :param pipeline: Fitted preprocessing pipeline
    :type pipeline: Pipeline
    :return: List of feature names after transformation
    :rtype: List[str]
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names: List[str] = []

    for _, transformer, columns in preprocessor.transformers_:
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(columns).tolist())
        else:
            feature_names.extend(columns)

    return feature_names


def prepare_data_for_training(
    df: pd.DataFrame, target_col: str = "Exited", pipeline: Optional[Pipeline] = None
) -> Tuple[np.ndarray, np.ndarray, Pipeline, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing")

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    if pipeline is None:
        pipeline = build_preprocessing_pipeline()
        X_transformed = pipeline.fit_transform(X)
        fitted = True
    else:
        X_transformed = pipeline.transform(X)
        fitted = False

    try:
        feature_names = extract_feature_names(pipeline)
    except Exception:
        logger.warning("Falling back to positional feature names")
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    logger.info(
        "Data prepared for training",
        extra={
            "rows": X_transformed.shape[0],
            "feature_count": X_transformed.shape[1],
            "pipeline_fitted": fitted,
            "churn_rate": float(y.mean()),
            "feature_names": feature_names, 
        },
    )

    return X_transformed, y, pipeline, feature_names
