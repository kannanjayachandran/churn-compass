"""
Churn Compass - Hyperparameter Tuning with Optuna

Optimizes XGBoost hyperparameters using joint objective:

0.5 × PR-AUC + 0.5 × Recall@Top10%

Features:
    - Optuna TPE sampler for efficient search
    - MLflow integration with nested runs
    - Early stopping for trials
    - Parallel trial execution
    - Best parameters saved to JSON
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from optuna.study import Study
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import xgboost as xgb
import mlflow

from churn_compass import settings, setup_logger, log_execution_time
from churn_compass.modeling import load_and_split_data, joint_objective_optuna
from churn_compass.features import prepare_data_for_training


logger = setup_logger(__name__)


# search space
def suggest_params(trial: optuna.Trial) -> Dict:
    """
    Docstring for suggest_params

    :param trial: Description
    :type trial: optuna.Trial
    :return: Description
    :rtype: Dict[Any, Any]
    """
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 400),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": settings.random_seed,
        "n_jobs": -1,
        "early_stopping_rounds": 20,
        "verbosity": 0,
    }


# Objective
class OptunaObjective:
    """
    Docstring for OptunaObjective
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pr_auc_weight: float = 0.5,
        recall_k_weight: float = 0.5,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pr_auc_weight = pr_auc_weight
        self.recall_k_weight = recall_k_weight

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        try:
            model = xgb.XGBClassifier(**params)
            model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )

            y_val_scores = model.predict_proba(self.X_val)[:, 1]

            score = joint_objective_optuna(
                self.y_val,
                y_val_scores,
                pr_auc_weight=self.pr_auc_weight,
                recall_k_weight=self.recall_k_weight,
            )

            return score

        except Exception as e:
            logger.warning(
                "Optuna trial failed",
                extra={"trial": trial.number, "error": str(e)},
            )
            raise optuna.TrialPruned()


# Optimization runner
@log_execution_time(logger)
def optimize_hyperparameters(
    data_path: str,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    output_path: Optional[str] = None,
) -> Tuple[Dict[str, float], Study]:
    """
    Docstring for optimize_hyperparameters

    :param data_path: Description
    :type data_path: str
    :param n_trials: Description
    :type n_trials: int
    :param timeout: Description
    :type timeout: Optional[int]
    :param n_jobs: Description
    :type n_jobs: int
    :param output_path: Description
    :type output_path: Optional[str]
    :return: Description
    :rtype: Tuple[Dict[Any, Any], Any]
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment_name}_optuna")

    train_df, val_df, test_df = load_and_split_data(data_path)

    X_train, y_train, pipeline, _ = prepare_data_for_training(train_df)
    X_val, y_val, _, _ = prepare_data_for_training(val_df, pipeline=pipeline)

    with mlflow.start_run(run_name="optuna_optimization"):
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=settings.random_seed),
        )

        objective = OptunaObjective(X_train, y_train, X_val, y_val)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
        )

        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if not completed_trials:
            logger.error("Optuna finished with no successful trials", 
                         extra={
                             "n_trials": len(study.trials), 
                             "states": [t.state.name for t in study.trials], 
                         }, 
                         )
            raise RuntimeError("All Optuna trials failed or were pruned")

        best_params = {
            **study.best_params,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": settings.random_seed,
            "n_jobs": -1,
            "verbosity": 0,
        }

        mlflow.log_metric("best_objective", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        resolved_output_path: Path
        if output_path is not None:
            resolved_output_path = Path(output_path)
        else:
            resolved_output_path = settings.data_processed_dir / "best_params.json"

        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        with resolved_output_path.open("w") as f:
            json.dump(best_params, f, indent=2)

        mlflow.log_artifact(str(resolved_output_path))

        return best_params, study

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna for Churn Compass"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed training parquet file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel jobs for Optuna",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save best params JSON",
    )

    args = parser.parse_args()

    best_params, study = optimize_hyperparameters(
        data_path=args.data,
        n_trials=args.trials,
        n_jobs=args.jobs,
        output_path=args.output,
    )

    logger.info(
        "Optuna optimization completed",
        extra={
            "best_value": study.best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
        },
    )


if __name__ == "__main__":
    main()
