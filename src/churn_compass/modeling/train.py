"""
Churn Compass - Model Training

XGBoost model training with MLflow tracking and model registry.

Features:
- Train/validation/test split
- MLflow experiment tracking
- Model registration
- Comprehensive metrics logging
- Feature importance tracking
- Artifact saving (plots, feature names, metrics)
"""

import json
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn as ms

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline

from churn_compass import settings, setup_logger, log_execution_time
from churn_compass.io import FileIO
from churn_compass.features import prepare_data_for_training
from churn_compass.modeling import calculate_all_metrics


logger = setup_logger(__name__)


# utility
def extract_feature_names_from_pipeline(pipeline: SKPipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            feature_names.extend(ohe.get_feature_names_out(columns).tolist())

    return feature_names


# Data loading & splitting
def load_and_split_data(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Docstring for load_and_split_data

    :param data_path: Description
    :type data_path: str
    :return: Description
    :rtype: Tuple[DataFrame, DataFrame, DataFrame]
    """
    logger.info("Starting data splitting")

    df = FileIO().read_parquet(data_path)

    train_val, test = train_test_split(
        df,
        test_size=settings.test_size,
        stratify=df["Exited"],
        random_state=settings.random_seed,
    )

    train, val = train_test_split(
        train_val,
        test_size=settings.val_size,
        stratify=train_val["Exited"],
        random_state=settings.random_seed,
    )

    logger.info(
        "Data split completed",
        extra={
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    )

    return train, val, test


# Model configuration
def get_default_xgb_params() -> Dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "random_state": settings.random_seed,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 20
    }


# Training
@log_execution_time(logger)
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
) -> xgb.XGBClassifier:
    """
    Docstring for train_model

    :param X_train: Description
    :type X_train: np.ndarray
    :param y_train: Description
    :type y_train: np.ndarray
    :param X_val: Description
    :type X_val: np.ndarray
    :param y_val: Description
    :type y_val: np.ndarray
    :param params: Description
    :type params: Dict
    :return: Description
    :rtype: XGBClassifier
    """
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    logger.info(
        "Model trained",
        extra={"best_iteration": best_iteration, "best_score": best_score,"n_estimators_used": (
            best_iteration + 1 if best_iteration is not None else model.n_estimators
        ),},
    )

    return model


# Feature Importance
def log_feature_importance(
    model: xgb.XGBClassifier, feature_names: list, top_n: int = 22
) -> pd.DataFrame:
    """
    Docstring for log_feature_importance

    :param model: Description
    :type model: xgb.XGBClassifier
    :param feature_names: Description
    :type feature_names: list
    :param top_n: Description
    :type top_n: int
    :return: Description
    :rtype: DataFrame
    """
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifact_path = "feature_importance.csv"
    importance_df.to_csv(artifact_path, index=False)
    mlflow.log_artifact(artifact_path)

    for i in range(min(top_n, len(importance_df))):
        mlflow.log_param(f"top_features_{i + 1}", importance_df.loc[i, "feature"])

    logger.info(
        "Feature importance logged",
        extra={"top_feature": importance_df.loc[0, "feature"]},
    )
    return importance_df


# Orchestration
@log_execution_time(logger)
def train_and_evaluate(
    data_path: str,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    params: Optional[Dict] = None,
    register_model: bool = False,
) -> Dict:
    """
    Docstring for train_and_evaluate

    :param data_path: Description
    :type data_path: str
    :param experiment_name: Description
    :type experiment_name: Optional[str]
    :param run_name: Description
    :type run_name: Optional[str]
    :param params: Description
    :type params: Optional[Dict]
    :param register_model: Description
    :type register_model: bool
    :return: Description
    :rtype: Dict[Any, Any]
    """

    settings.setup()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("data_path", data_path)

        train_df, val_df, test_df = load_and_split_data(data_path)

        X_train, y_train, pipeline, features = prepare_data_for_training(train_df)
        X_val, y_val, _, _ = prepare_data_for_training(val_df, pipeline=pipeline)
        X_test, y_test, _, _ = prepare_data_for_training(test_df, pipeline=pipeline)

        params = params or get_default_xgb_params()
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = train_model(X_train, y_train, X_val, y_val, params)

        if hasattr(model, "best_iteration"):
            mlflow.log_metric("best_iteration", model.best_iteration)
            mlflow.log_metric("best_score", model.best_score)

        scores = {
            "train": model.predict_proba(X_train)[:, 1],
            "val": model.predict_proba(X_val)[:, 1],
            "test": model.predict_proba(X_test)[:, 1],
        }

        metrics = {}
        for split, y_score in scores.items():
            y_true = locals()[f"y_{split}"]
            m = calculate_all_metrics(y_true, y_score)

            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{split}_{k}", v)

            metrics[split] = m

        final_feature_names = extract_feature_names_from_pipeline(pipeline)
        log_feature_importance(model, final_feature_names)

        full_pipeline = SKPipeline([("preprocessing", pipeline), ("model", model)])

        ms.log_model(
            full_pipeline,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name
            if register_model
            else None,
        )

        logger.info(f"Inside: train and evaluate: metrics key is {metrics.keys()}")

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model": model,
            "pipeline": pipeline,
        }

def main() -> None:

    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed parquet file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )
    parser.add_argument(
    "--params",
    type=str,
    default=None,
    help="Path to JSON file with tuned hyperparameters"
    )


    args = parser.parse_args()

    params = None
    if args.params:
        with open(args.params) as f:
            params = json.load(f)

    train_and_evaluate(
        data_path=args.data,
        experiment_name=args.experiment,
        run_name=args.run_name,
        params=params
    )


if __name__ == "__main__":
    main()
