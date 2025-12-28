"""
Churn Compass - Batch Scoring Pipeline

Score entire customer base and generate top-K targeting lists.
Writes results to database for CRM integration.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Literal

import pandas as pd
from prefect import flow, task

from churn_compass import settings, setup_logger, log_execution_time
from churn_compass.io import FileIO, DatabaseIO
from churn_compass.serving import get_model_registry, ChurnPredictor

logger = setup_logger(__name__)


@task(name="load_customer_data", retries=2)
@log_execution_time(logger)
def load_customer_data(
    source: str, 
    source_type: str = "parquet"
) -> pd.DataFrame:
    """
    Load customer data from file or database
    
    :param source: Path to file or SQL query
    :type source: str
    :param source_type: Data source type (parquet, csv, sql)
    :type source_type: str
    :return: customer DataFrame
    :rtype: DataFrame
    """

    logger.info("Loading customer data", extra={"source_type": source_type, "source": source})

    file_io = FileIO()

    if source_type == "parquet":
        df = file_io.read_parquet(source)
    elif source_type == "csv":
        df = file_io.read_csv(source)
    elif source_type == "sql":
        db = DatabaseIO()
        df = db.read_query(source)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    
    logger.info(
        "Customer data loaded", 
        extra={
            "rows": len(df), 
            "columns": len(df.columns), 
        }
    )

    return df

@task(name="score_customers", retries=1)
@log_execution_time(logger)
def score_customers(
    df: pd.DataFrame, 
    model_stage: str = "Production"
) -> pd.DataFrame:
    """
    Score all customers using production model.
    
    :param df: Customer DataFrame
    :type df: pd.DataFrame
    :param model_stages: Model stage to use
    :type model_stages: str
    :return: DataFrame with predictions
    :rtype: DataFrame
    """
    logger.info(f"Scoring {len(df)} customers with {model_stage} model")

    loader = get_model_registry()
    model = loader.load_by_stage(stage=model_stage)
    predictor = ChurnPredictor(model=model)
    results_df = predictor.predict_batch(df, include_features=True)
    results_df['scored_at'] = datetime.now(timezone.utc).isoformat()
    results_df['model_stage'] = model_stage
    cache_key = f"{settings.mlflow_model_name}_{model_stage}"
    metadata = loader.get_metadata(cache_key)
    if metadata:
        results_df['model_version'] = metadata.get('version', 'unknown')
    
    logger.info(
        "Scoring completed", 
        extra={
            "scored_count": len(results_df), 
            "mean_probability": float(results_df["probability"].mean()), 
            "high_risk_count": int(
                (results_df["probability"] >= settings.high_risk_threshold).sum()
            ),
        },
    )

    return results_df

@task(name="generate_top_k_list", retries=1)
@log_execution_time(logger)
def generate_top_k_list(
    results_df: pd.DataFrame, 
    k_percent: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate top-K targeting list.
    
    :param results_df: Scored customers
    :type results_df: pd.DataFrame
    :param k_percent: Percentage to target
    :type k_percent: Optional[float]
    :return: Top-K customers DataFrame
    :rtype: DataFrame
    """
    if k_percent is None:
        k_percent = settings.top_k_percent
    
    k = max(1, int(len(results_df) * k_percent))

    logger.info(
        "Generating top-K list", 
        extra={"k": k, "k_percent": k_percent}
    )

    top_k_df = results_df.nlargest(k, "probability").copy()

    top_k_df["risk_rank"] = range(1, len(top_k_df) + 1)

    return top_k_df

@task(name="save_results_to_database", retries=2)
@log_execution_time(logger)
def save_results_to_database(
    results_df: pd.DataFrame, 
    table_name: str, 
    if_exists: Literal["replace", "append", "fail"] = "replace"
):
    """
    Save scoring results to database
    
    :param results_df: Results DataFrame
    :type results_df: pd.DataFrame
    :param table_name: Target table name
    :type table_name: str
    :param if_exists: How to handle existing table
    :type if_exists: str
    :return: Number of rows written
    :rtype: int
    """
    logger.info("Saving results to database table", extra={"table_name": table_name, "if_exists": if_exists,})

    db = DatabaseIO()

    db.write_table(
        results_df, 
        table_name=table_name, 
        if_exists=if_exists, 
    )

    logger.info("Results saved to database", extra={
        "table": table_name, 
        "rows": len(results_df), 
    })
    return len(results_df)


@task(name="save_results_to_file", retries=2)
@log_execution_time(logger)
def save_results_to_file(
    results_df: pd.DataFrame, 
    output_path: str, 
    format: str = "parquet"
) -> str:
    """
    Save scoring results to file
    
    :param results_df: Results DataFrame
    :type results_df: pd.DataFrame
    :param output_path: Output file path
    :type output_path: str
    :param format: File format(parquet, sql)
    :type format: str
    :return: Output file path
    :rtype: str
    """
    logger.info("Saving results to file", extra={"Output_path": output_path})

    file_io = FileIO()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        file_io.write_parquet(results_df, output_path)
    elif format == "csv":
        file_io.write_csv(results_df, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info("Results saved", extra={"output_path": output_path})

    return output_path


@flow(
    name="batch_scoring", 
    description="Batch scoring pipeline for customer churn prediction", 
    log_prints=True
)
def scoring_flow(
    input_source: str, 
    source_type: str = "parquet", 
    output_table: Optional[str] = None, 
    output_file: Optional[str] = None, 
    top_k_table: Optional[str] = None, 
    top_k_file: Optional[str] = None, 
    k_percent: Optional[float] = None, 
    model_stage: str = "Production", 
) -> Dict:
    """
    Main batch scoring flow
    
    :param input_source: Description
    :type input_source: str
    :param source_type: Description
    :type source_type: str
    :param output_table: Description
    :type output_table: Optional[str]
    :param output_file: Description
    :type output_file: Optional[str]
    :param top_k_table: Description
    :type top_k_table: Optional[str]
    :param top_k_file: Description
    :type top_k_file: Optional[str]
    :param k_percent: Description
    :type k_percent: Optional[float]
    :param model_stage: Description
    :type model_stage: str
    :return: Description
    :rtype: Dict[Any, Any]
    """
    settings.setup()
    logger.info(
        "Starting batch scoring flow", 
        extra={
            "input_source": input_source, 
            "source_type": source_type, 
            "model_stage": model_stage, 
        }, 
    )

    # Load data
    df = load_customer_data(input_source, source_type)

    # Score customers
    results_df = score_customers(df, model_stage)

    # Generate top-K list
    top_k_df = generate_top_k_list(results_df, k_percent)

    # save results
    rows_written: int = 0
    files_written: list = []

    # save all scores to db
    if output_table:
        rows_written += save_results_to_database(
            results_df, 
            table_name=output_table, 
            if_exists="replace"
        )

    # Save all scores to file
    if output_file:
        path = save_results_to_file(results_df, output_file, format="parquet")
        files_written.append(path)

    # Save top-K to 
    if top_k_table:
        save_results_to_database(
            top_k_df, 
            table_name=top_k_table, 
            if_exists="replace", 
        )

    # Save top-K to file
    if top_k_file:
        path = save_results_to_file(top_k_df, top_k_file, format="csv")
        files_written.append(path)
    
    # Summary
    summary = {
        "total_scored": len(results_df), 
        "top_k": len(top_k_df), 
        "mean_probability": float(results_df['probability'].mean()), 
        "high_risk_count": int((results_df['probability'] >= 0.7).sum()), 
        "rows_written_db": rows_written, 
        "files_written": files_written, 
        "model_stage": model_stage, 
    }

    logger.info("Batch scoring completed", extra=summary)

    print("\n" + "="*80)
    print("Batch Scoring Completed!")
    print("="*80)
    print(f"Total Customers Scored: {summary['total_scored']:,}")
    print(f"Top-K Customers: {summary['top_k']:,}")
    print(f"Mean Churn Probability: {summary['mean_probability']:.2%}")
    print(f"High Risk Count: {summary['high_risk_count']:,}")
    if files_written:
        print(f"\nFiles Written:")
        for f in files_written:
            print(f"  - {f}")
    print("="*80 + "\n")

    return summary


def main():
    """CLI entry point for batch scoring"""
    parser = argparse.ArgumentParser(description="Batch scoring pipeline")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Input data source (file path or SQL query)"
    )
    parser.add_argument(
        "--source-type", 
        type=str, 
        default="parquet", 
        choices=["parquet", "csv", "sql"], 
        help="Input source type",
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        help="Database table for all scores"
    )
    parser.add_argument(
        "--top-k-table", 
        type=str, 
        help="Database table for top-K list"
    )
    parser.add_argument(
        "--top-k-file", 
        type=str, 
        default=None, 
    )
    parser.add_argument(
        "--k-percent",
        type=float,
        default=None,
        help="Top-K percentage (default from settings)"
    )
    parser.add_argument(
        "--model-stage", 
        type=str, 
        default="Production", 
    )
    parser.add_argument(
    "--output-table",
    type=str,
    help="Database table for all scores",
    )

    args = parser.parse_args()

    try:
        summary = scoring_flow(
            input_source=args.input,
            source_type=args.source_type,
            output_table=args.output_table,
            output_file=args.output_file,
            top_k_table=args.top_k_table,
            top_k_file=args.top_k_file,
            k_percent=args.k_percent,
            model_stage=args.model_stage
        )

        print(f"Scoring completed. Total scored: {summary['total_scored']:,}")

    except Exception as e:
        logger.error("Batch Scoring failed", exc_info=True)
        print(f"Scoring failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
