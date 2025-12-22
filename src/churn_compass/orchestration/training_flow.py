"""
Churn Compass - Training Orchestration Flow

Automated model training and registration workflow.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from churn_compass import settings, setup_logger, log_execution_time
from churn_compass.modeling import train_and_evaluate, optimize_hyperparameters
from churn_compass.io import FileIO
from churn_compass.validation import validate_training_data

logger = setup_logger(__name__)


@task(name="validate_data_quality", retries=1)
@log_execution_time(logger)
def validate_data_quality(data_path: str) -> None:
    """
    Docstring for validate_data_quality
    
    :param data_path: Description
    :type data_path: str
    :return: Description
    :rtype: bool
    """
    logger.info("Validating data quality", extra={"data_path": data_path})

    file_io = FileIO()
    suffix = Path(data_path).suffix.lower()

    if suffix == ".parquet":
        df = file_io.read_parquet(data_path)
    elif suffix == ".csv":
        df = file_io.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported training data format: {suffix}")
    validate_training_data(df)

    logger.info("Data quality validation passed")

@task(name="train_baseline_model", retries=1)
@log_execution_time(logger)
def train_baseline_model(data_path: str, experiment_name: Optional[str] = None) -> Dict:
    """
    Docstring for train_baseline_model
    
    :param data_path: Description
    :type data_path: str
    :param experiment_name: Description
    :type experiment_name: Optional[str]
    :return: Description
    :rtype: Dict[Any, Any]
    """
    logger.info("Training baseline model", extra={"experiment_name": experiment_name})
    results = train_and_evaluate(
        data_path=data_path, 
        experiment_name=experiment_name, 
        run_name="baseline_model", 
        params=None, 
        register_model=False
    )

    return results

@task(name="hyperparameter_tuning", retries=1)
@log_execution_time(logger)
def hyperparameter_tuning(data_path: str, n_trials: int = 50, experiment_name: Optional[str] = None) -> Dict:
    """
    Docstring for hyperparameter_tuning
    
    :param data_path: Description
    :type data_path: str
    :param n_trials: Description
    :type n_trials: int
    :param experiment_name: Description
    :type experiment_name: Optional[str]
    :return: Description
    :rtype: Dict[Any, Any]
    """
    logger.info("Starting hyperparameter tuning", extra={"n_trials": n_trials})

    best_params, study = optimize_hyperparameters(
        data_path=data_path, 
        n_trials=n_trials, 
    )

    return {
        "best_params": best_params, 
        "best_objective": study.best_value, 
        "best_trial": study.best_trial.number
    }

@task(name="train_optimized_model", retries=1)
@log_execution_time(logger)
def train_optimized_model(
    data_path: str, 
    best_params: Optional[Dict], 
    experiment_name: Optional[str], 
    register_model: bool = True
) -> Dict:
    logger.info("Training optimized model", 
                extra={
                    "using_defaults": best_params is None
                })
    
    return train_and_evaluate(
        data_path=data_path, 
        experiment_name=experiment_name, 
        run_name="optimized_model", 
        params=best_params if best_params else {}, 
        register_model=register_model
    )

@task(name="create_training_summary")
def create_training_summary(
    baseline_results: Dict,
    optimized_results: Dict
) -> str:
    baseline_test = baseline_results["metrics"]["test"]
    optimized_test = optimized_results["metrics"]["test"]

    pr_auc_improvement = optimized_test["pr_auc"] - baseline_test["pr_auc"]

    markdown = f"""
# Churn Compass Training Summary

## Performance
- **Baseline PR-AUC**: {baseline_test['pr_auc']:.4f}
- **Optimized PR-AUC**: {optimized_test['pr_auc']:.4f} ({pr_auc_improvement:+.4f})
"""

    importance_df = optimized_results.get("feature_importance")
    if importance_df is not None and not importance_df.empty:
        markdown += "\n## Top Features\n"
        for i, row in importance_df.head(5).iterrows():
            markdown += f"\n{i+1}. **{row['feature']}**: {row['importance']:.4f}"

    return markdown

@flow(
    name="model_training", 
    description="Complete model training workflow with hyperparameter optimization", 
    log_prints=True 
)
def training_flow(
    data_path: str, 
    skip_tuning: bool = False, 
    n_trials: int = 50, 
    register_model: bool = True, 
    experiment_name: Optional[str] = None
) -> Dict:
    """
    Docstring for training_flow

    Steps:
    1. Validate data quality
    2. Train baseline model
    3. Hyperparameter tuning (optional)
    4. Train optimized model
    5. Generate summary report
    
    :param data_path: Description
    :type data_path: str
    :param skip_tuning: Description
    :type skip_tuning: bool
    :param n_trials: Description
    :type n_trials: int
    :param register_model: Description
    :type register_model: bool
    :param experiment_name: Description
    :type experiment_name: Optional[str]
    :return: Description
    :rtype: Dict[Any, Any]
    """
    logger.info(
        "Starting training flow", 
        extra={
            "data_path": data_path, 
            "skip_tuning": skip_tuning, 
            "n_trials": n_trials
        }
    )

    if experiment_name is None:
        experiment_name = settings.mlflow_experiment_name

    # 1. Validate data
    validate_data_quality(data_path)

    # 2. Train baseline
    baseline_results = train_baseline_model(data_path, experiment_name)

    # 3 & 4. Tune and train optimized model
    if skip_tuning:
        logger.info("Skipping hyperparameter tuning (using defaults)")
        optimized_results = train_optimized_model(
            data_path, 
            best_params=None, 
            experiment_name=experiment_name, 
            register_model=register_model
        )
        tuning_results = None
    else:
        tuning_results = hyperparameter_tuning(data_path, n_trials, experiment_name)
        optimized_results = train_optimized_model(
            data_path, 
            best_params=tuning_results['best_params'], 
            experiment_name=experiment_name, 
            register_model=register_model
        )
    
    # 5. Create summary
    summary_markdown = create_training_summary(baseline_results, optimized_results)

    # Create Prefect artifact
    create_markdown_artifact(
        key="training-summary", 
        markdown=summary_markdown, 
        description="Model training performance summary"
    )

    # Log summary
    print("\n" + "="*80)
    print(summary_markdown)
    print("="*80 + "\n")
    
    results = {
        'baseline_results': baseline_results,
        'tuning_results': tuning_results,
        'optimized_results': optimized_results,
        'final_run_id': optimized_results['run_id'],
        'model_registered': register_model
    }
    
    logger.info("Training flow completed successfully")

    return results


def main():
    """CLI"""
    parser = argparse.ArgumentParser(description="Model training flow")
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Path to training data"
    )
    parser.add_argument(
        "--skip-tuning", 
        action="store_true", 
        help="Skip hyperparameter tuning"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=50, 
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--no-register", 
        action="store_true", 
        help="Don't register model in MLflow"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default=None, 
        help="MLflow experiment name"
    )

    args = parser.parse_args()

    try:
        results = training_flow(
            data_path=args.data, 
            skip_tuning=args.skip_tuning, 
            n_trials=args.trials, 
            register_model=not args.no_register, 
            experiment_name=args.experiment
        )

        print(f"✅ Training completed. Final run ID: {results['final_run_id']}")

    except Exception as e:
        logger.error("Training flow failed", exc_info=True)
        print(f"❌ Training failed: {str(e)}") 

if __name__ == "__main__":
    main()
    