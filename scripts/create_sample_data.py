#!/usr/bin/env python3
"""
Churn Compass - Custom Synthetic Data Generator 

We initially use SDV for synthetic data generation, it was a massive 
overkill for what we’re actually using it hence we replaced SDV with a 
lightweight Gaussian-copula-like sampler +
conditional churn model (Exited depends on Age + Balance).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from churn_compass.config.settings import settings
from churn_compass.io.file_io import FileIO
from churn_compass.logging.logger import log_execution_time, setup_logger
from churn_compass.pipelines.ingest_pipeline import data_ingestion_flow


logger = setup_logger("Synthetic Data Generator")


@log_execution_time(logger)
def load_and_clean(input_path: str) -> pd.DataFrame:
    logger.info(f"Started loading and cleaning data from: {input_path}")

    try:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = FileIO().read_csv(input_path)

        logger.info(
            "Real data loaded for cleaning",
            extra={"shape": df.shape, "columns": df.columns.tolist()},
        )

        # remove duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Removing {n_duplicates} duplicate rows")
            df = df.drop_duplicates()

        logger.info(f"Cleaned dataset successfully. Final shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(
            "Failed to load and clean real data",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise


# ----------------------- metadata configuration -----------------------
@log_execution_time(logger)
def configure_metadata(df: pd.DataFrame) -> Dict:
    """
    Create a lightweight metadata dict describing column types and basic stats.
    (Replaces SDV SingleTableMetadata)
    """
    try:
        numerical = []
        categorical = []
        binary = []
        id_cols = []

        for col in df.columns:
            if col.lower() in ("rownumber", "row_number", "customerid", "customer_id"):
                id_cols.append(col)
                continue

            if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                # Heuristic: small unique values -> categorical/binary
                nunique = df[col].nunique(dropna=True)
                if nunique <= 5:
                    unique_vals = set(df[col].dropna().unique())
                    # Only classify as binary if values are exactly {0, 1}
                    if unique_vals == {0, 1} or unique_vals == {0} or unique_vals == {1}:
                        binary.append(col)
                    else:
                        categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)

        # basic stats for numerical columns
        stats_summary = {}
        for col in numerical:
            col_series = df[col].dropna()
            stats_summary[col] = {
                "min": float(col_series.min()),
                "max": float(col_series.max()),
                "mean": float(col_series.mean()),
                "std": float(col_series.std()),
            }

        metadata = {
            "numerical": numerical,
            "categorical": categorical,
            "binary": binary,
            "id_cols": id_cols,
            "stats": stats_summary,
        }

        logger.info("Metadata configured successfully", extra={"metadata": metadata})
        return metadata

    except Exception as e:
        logger.error(
            "Failed to configure metadata",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise


# ----------------------- drift application -----------------------
@log_execution_time(logger)
def apply_drift(df: pd.DataFrame, Original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply drift transformation to synthetic data.
    """
    logger.info("Applying drift transformations")

    try:
        df_drift = df.copy()

        # Age drift +5 years
        if "Age" in df_drift.columns and "Age" in Original_df.columns:
            original_age_mean = Original_df["Age"].mean()
            df_drift["Age"] = df_drift["Age"] + 5
            df_drift["Age"] = df_drift["Age"].clip(
                lower=Original_df["Age"].min(), upper=Original_df["Age"].max() + 10
            )
            new_age_mean = df_drift["Age"].mean()

            logger.info(
                f"Age drift applied: {original_age_mean:.2f} -> {new_age_mean:.2f}"
                f"(+{new_age_mean - original_age_mean:.2f} Years)"
            )

        # churn rate drift: -> 28%
        if "Exited" in df_drift.columns:
            target_churn = 0.28
            current_churn = df_drift["Exited"].mean()
            additional_churns = (
                int(len(df_drift) * target_churn) - int(df_drift["Exited"].sum())
            )

            if additional_churns > 0:
                non_churned_idx = df_drift[df_drift["Exited"] == 0].index
                if len(non_churned_idx) >= additional_churns:
                    flip_idx = np.random.choice(
                        non_churned_idx, size=additional_churns, replace=False
                    )
                    df_drift.loc[flip_idx, "Exited"] = 1

            new_churn = df_drift["Exited"].mean()
            logger.info(
                f"Churn rate drift applied: {current_churn:.2%} -> {new_churn:.2%}"
                f"(target: {target_churn:.2%})"
            )

        # noise to other numerical columns
        drift_cols = [
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "Tenure",
            "NumOfProducts",
        ]
        for col in drift_cols:
            if col in df_drift.columns and pd.api.types.is_numeric_dtype(df_drift[col]):
                noise = np.random.normal(0, 0.35, len(df_drift))
                df_drift[col] = df_drift[col] * (1 + noise)
                if col in Original_df.columns:
                    df_drift[col] = df_drift[col].clip(
                        lower=Original_df[col].min(), upper=Original_df[col].max()
                    )

        logger.info(f"Drift applied to {len(drift_cols)} additional columns")
        return df_drift

    except Exception as e:
        logger.error(
            "Failed to apply data drift",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise


# ----------------------- quality evaluation (SDV-free) -----------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for a single numerical vector."""
    eps = 1e-8
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # compute quantile bins based on expected
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected, quantiles)

    # ensure unique bin edges for histogram (constant data edge case)
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        # constant data: PSI is 0 if both constant and equal, else undefined
        return 0.0

    e_counts, _ = np.histogram(expected, bins=unique_edges)
    a_counts, _ = np.histogram(actual, bins=unique_edges)

    e_perc = e_counts / (len(expected) + eps)
    a_perc = a_counts / (len(actual) + eps)

    psi_value = np.sum((e_perc - a_perc) * np.log((e_perc + eps) / (a_perc + eps)))
    return float(psi_value)


def ks_report(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    result = {}
    for col in cols:
        if col in real.columns and col in synth.columns and pd.api.types.is_numeric_dtype(real[col]):
            try:
                stat = stats.ks_2samp(real[col].dropna(), synth[col].dropna()).statistic
                result[col] = float(stat)
            except Exception:
                result[col] = None
        else:
            result[col] = None
    return result


def corr_matrix_diff(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    cols_present = [c for c in cols if c in real.columns and c in synth.columns and pd.api.types.is_numeric_dtype(real[c])]
    if len(cols_present) < 2:
        return 0.0
    r_corr = real[cols_present].corr().fillna(0).values
    s_corr = synth[cols_present].corr().fillna(0).values
    diff = np.abs(r_corr - s_corr)
    # summary metric: mean absolute difference
    return float(np.nanmean(diff))


@log_execution_time(logger)
def evaluate_synthetic_data_quality(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict) -> Dict:
    """
    Run lightweight quality checks: KS for numerical columns, PSI, and correlation matrix diff.
    """
    logger.info("Running lightweight quality evaluation (KS / PSI / CorrDiff)")
    try:
        numerical = metadata.get("numerical", [])

        ks = ks_report(real_data, synthetic_data, numerical)

        psi_vals = {}
        for col in numerical:
            if col in real_data.columns and col in synthetic_data.columns:
                psi_vals[col] = psi(real_data[col].dropna().values, synthetic_data[col].dropna().values)
            else:
                psi_vals[col] = None

        corr_diff = corr_matrix_diff(real_data, synthetic_data, numerical)

        # simple aggregated score (lower is better). We return raw components so consumers can decide.
        overall = {
            "ks": ks,
            "psi": psi_vals,
            "corr_matrix_mean_abs_diff": corr_diff,
        }

        logger.info("Quality evaluation completed", extra={"summary": {"corr_diff": corr_diff}})
        return overall

    except Exception as e:
        logger.warning(f"Quality evaluation failed: {e}")
        return {"error": str(e)}


# ----------------------- conditional churn model -----------------------
class ConditionalChurnModel:
    """Logistic model for P(Exited=1 | Age, Balance) learned from real data."""

    def __init__(self, age_col: str = "Age", balance_col: str = "Balance", target_col: str = "Exited"):
        self.age_col = age_col
        self.balance_col = balance_col
        self.target_col = target_col
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None
        self.base_rate: float = 0.0

    def fit(self, df: pd.DataFrame):
        if self.age_col not in df.columns or self.balance_col not in df.columns or self.target_col not in df.columns:
            raise ValueError("Dataframe must contain Age, Balance, and Exited columns to fit churn model")

        X = df[[self.age_col, self.balance_col]].fillna(0.0)
        y = df[self.target_col].fillna(0).astype(int).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(X_scaled, y)

        self.base_rate = float(np.mean(y))
        logger.info(f"Fitted conditional churn model. base_rate={self.base_rate:.4f}")
        return self

    def sample(self, synthetic_df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")

        Xs = synthetic_df[[self.age_col, self.balance_col]].copy().fillna(0.0)
        Xs_scaled = self.scaler.transform(Xs)
        probs = self.model.predict_proba(Xs_scaled)[:, 1]
        return np.random.binomial(1, probs)


# ----------------------- lightweight tabular synthesizer -----------------------
class LightweightTabularSynthesizer:
    """Lightweight Gaussian-copula-like sampler for tabular data.

    - Numerical columns: transforms empirical -> normal by ranks; samples multivariate normal using empirical covariance of transformed ranks; inverse-quantiles back.
    - Categorical: empirical sampling
    - Binary (non-churn): independent Bernoulli using empirical rates
    - Exited: handled by ConditionalChurnModel when provided
    """

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str],
        id_cols: Optional[List[str]] = None,
        churn_model: Optional[ConditionalChurnModel] = None,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.id_cols = id_cols or []
        self.churn_model = churn_model

        # learned attributes
        self.num_empirical_sorted = {}
        self.cov = None
        self.cat_distributions = {}
        self.binary_rates = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        # numerical: build empirical CDFs and gaussian transforms
        gaussian_data = []
        for col in self.numerical_cols:
            if col not in df.columns:
                continue
            vals = df[col].astype(float).dropna().values
            if len(vals) == 0:
                continue
            sorted_vals = np.sort(vals)
            self.num_empirical_sorted[col] = sorted_vals

            # ranks -> uniform -> normal
            ranks = stats.rankdata(vals) / (len(vals) + 1.0)
            gaussian = stats.norm.ppf(ranks)
            gaussian_data.append(gaussian)

        if len(gaussian_data) >= 1:
            gaussian_stack = np.vstack(gaussian_data).T
            if gaussian_stack.ndim == 1:
                gaussian_stack = gaussian_stack.reshape(-1, 1)
            raw_cov = np.cov(gaussian_stack, rowvar=False)
            # ensure cov is always 2D (np.cov returns scalar for single column)
            self.cov = np.atleast_2d(raw_cov)
        else:
            self.cov = None

        # categorical
        for col in self.categorical_cols:
            if col in df.columns:
                counts = df[col].value_counts(dropna=True)
                if counts.sum() > 0:
                    self.cat_distributions[col] = (counts.index.tolist(), (counts / counts.sum()).values.tolist())

        # binary (excluding Exited)
        for col in self.binary_cols:
            if col in df.columns and col != "Exited":
                vals = df[col].dropna().astype(int).values
                if len(vals) > 0:
                    rate = float(np.mean(vals))
                    # clip rate to valid probability range
                    self.binary_rates[col] = np.clip(rate, 0.0, 1.0)

        # churn model
        if self.churn_model is not None:
            try:
                self.churn_model.fit(df)
            except Exception as e:
                logger.warning(f"Churn model fitting failed: {e}")
                self.churn_model = None

        self._fitted = True
        logger.info("Lightweight synthesizer fitted")
        return self

    def sample(self, n_rows: int, start_row_number: int = 1, start_customer_id: int = 100000) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Synthesizer not fitted. Call fit() with real data first.")

        out = pd.DataFrame(index=range(n_rows))

        # Generate ID columns first
        for col in self.id_cols:
            col_lower = col.lower()
            if col_lower in ("rownumber", "row_number"):
                out[col] = np.arange(start_row_number, start_row_number + n_rows)
            elif col_lower in ("customerid", "customer_id"):
                out[col] = np.arange(start_customer_id, start_customer_id + n_rows)
            else:
                # Generic ID: generate sequential integers
                out[col] = np.arange(1, n_rows + 1)

        # numerical sampling
        if self.cov is not None and len(self.num_empirical_sorted) > 0:
            cols = [c for c in self.numerical_cols if c in self.num_empirical_sorted]
            dim = len(cols)
            # if cov is scalar (single column), make it 1x1
            cov = np.array(self.cov, copy=True)
            mean = np.zeros(dim)
            try:
                z = np.random.multivariate_normal(mean=mean, cov=cov, size=n_rows)
            except Exception:
                # fallback: independent normals
                z = np.random.normal(size=(n_rows, dim))

            for i, col in enumerate(cols):
                u = stats.norm.cdf(z[:, i])  # uniform in (0,1)
                # map to empirical quantiles
                sorted_vals = self.num_empirical_sorted[col]
                out[col] = np.quantile(sorted_vals, u)

        else:
            # as fallback, sample from each empirical distribution independently
            for col, sorted_vals in self.num_empirical_sorted.items():
                u = np.random.rand(n_rows)
                out[col] = np.quantile(sorted_vals, u)

        # categorical
        for col, (choices, probs) in self.cat_distributions.items():
            out[col] = np.random.choice(choices, size=n_rows, p=probs)

        # binary (non-churn)
        for col, rate in self.binary_rates.items():
            # ensure rate is valid for binomial distribution
            valid_rate = np.clip(rate, 0.0, 1.0) if not np.isnan(rate) else 0.5
            out[col] = np.random.binomial(1, valid_rate, size=n_rows)

        # churn
        if self.churn_model is not None:
            try:
                out["Exited"] = self.churn_model.sample(out)
            except Exception as e:
                logger.warning(f"Churn sampling failed: {e} -- falling back to global rate")
                base_rate = getattr(self.churn_model, "base_rate", None)
                if base_rate is None:
                    base_rate = 0.1
                out["Exited"] = np.random.binomial(1, base_rate, size=n_rows)
        else:
            # fallback: uniform based on mean of provided binary rates or 0.1
            if "Exited" in self.binary_rates:
                rate = self.binary_rates["Exited"]
            else:
                rate = 0.1
            out["Exited"] = np.random.binomial(1, rate, size=n_rows)

        return out


# ----------------------- data generation workflow -----------------------
@log_execution_time(logger)
def generate_synthetic_data(
    input_path: str,
    base_rows: int = 2500,
    drift_rows: int = 1000,
    skip_pipeline: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:

    logger.info("Synthetic data generator Started")

    try:
        if output_dir:
            raw_dir = Path(output_dir) / "raw"
            processed_dir = Path(output_dir) / "processed"
        else:
            raw_dir = Path(settings.data_raw_dir)
            processed_dir = Path(settings.data_processed_dir)

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        raw_csv_path = raw_dir / "synthetic_raw_sample.csv"
        base_parquet_path = processed_dir / "synthetic_sample.parquet"
        drift_parquet_path = processed_dir / "synthetic_sample_drifted.parquet"
        quality_report_path = processed_dir / "synthetic_quality_report.json"

        # Step: 1 - Load data
        real_df = load_and_clean(input_path)

        # Step: 2 - Configure metadata (lightweight)
        metadata = configure_metadata(real_df)

        # Step: 3 - Train synthesizer (lightweight)
        numerical_cols = metadata.get("numerical", [])
        categorical_cols = metadata.get("categorical", [])
        binary_cols = metadata.get("binary", [])
        id_cols = metadata.get("id_cols", [])
        
        # Ensure required ID columns are always generated for pipeline compatibility
        required_id_cols = ["RowNumber", "CustomerId"]
        for col in required_id_cols:
            if col not in id_cols:
                id_cols.append(col)

        churn_model = ConditionalChurnModel(age_col="Age", balance_col="Balance", target_col="Exited")

        synthesizer = LightweightTabularSynthesizer(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            id_cols=id_cols,
            churn_model=churn_model,
        )
        synthesizer.fit(real_df)

        # Step: 4 - Generate synthetic data
        synthetic_base = synthesizer.sample(n_rows=base_rows)

        # Step: 5 - Generate drifted synthetic data
        synthetic_drift_raw = synthesizer.sample(n_rows=drift_rows)
        synthetic_drift = apply_drift(synthetic_drift_raw, real_df)

        # Step: 6 - Save raw synthetic data
        logger.info(f"Saving raw synthetic data to {raw_csv_path}")
        FileIO().write_csv(synthetic_base, raw_csv_path)

        # Step: 7 - Process through pipeline (Optional)
        if skip_pipeline:
            logger.warning("Skipping pipeline processing (--skip-pipeline flag set)")
            logger.info(
                "Writing synthetic data directly to processed directory. This might cause data inconsistency..."
            )

            FileIO().write_parquet(synthetic_base, base_parquet_path)
            FileIO().write_parquet(synthetic_drift, drift_parquet_path)

            base_processed_path = str(base_parquet_path)
            drift_processed_path = str(drift_parquet_path)

        else:
            logger.info("Processing synthetic data through ingestion pipeline")
            base_processed_path = data_ingestion_flow(
                input_path=str(raw_csv_path), output_path=str(base_parquet_path)
            )

            # for drifted data, save temp CSV and process through pipeline
            temp_drift_csv = raw_dir / "sample_drift_temp.csv"
            FileIO().write_csv(synthetic_drift, temp_drift_csv)

            logger.info("-> Processing Drifted Synthetic data...")
            drift_processed_path = data_ingestion_flow(
                input_path=str(temp_drift_csv), output_path=str(drift_parquet_path)
            )

            # clean up temp csv file
            if temp_drift_csv.exists():
                temp_drift_csv.unlink()

            logger.info("Pipeline processing completed successfully")

        # Step: 8 - Quality evaluation
        quality_metrics = evaluate_synthetic_data_quality(real_df, synthetic_base, metadata)

        # save quality report
        with open(quality_report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "base_rows": base_rows,
                    "drift_rows": drift_rows,
                    "real_data_shape": real_df.shape,
                    "quality_metrics": quality_metrics,
                },
                f,
                indent=2,
            )

        logger.info(f"Quality report saved to {quality_report_path}")

        return {
            "raw_csv": str(raw_csv_path),
            "base_parquet": str(base_processed_path),
            "drift_parquet": str(drift_processed_path),
            "quality_report": str(quality_report_path),
        }

    except Exception as e:
        logger.error(
            "Synthetic data generation failed",
            extra={
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


# ----------------------- CLI arguments -----------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic churn dataset (lightweight) + Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to real dataset CSV for training the synthetic model",
    )

    parser.add_argument(
        "--base-rows",
        type=int,
        default=2000,
        help="Number of rows for base synthetic dataset (default: 2000)",
    )

    parser.add_argument(
        "--drift-rows",
        type=int,
        default=1000,
        help="Number of rows for drifted synthetic dataset (default: 1000)",
    )

    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip running synthetic data through ingestion pipeline (Faster but no validation)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: uses settings.data_raw_dir and settings.data_processed_dir)",
    )

    return parser.parse_args()


# ----------------------- CLI main -----------------------

def main():
    args = parse_arguments()

    logger.info(
        "Starting synthetic data generation with arguments:\n"
        f"  Input: {args.input}\n"
        f"  Base rows: {args.base_rows}"
        f"  Drift rows: {args.drift_rows}"
        f"  Skip pipeline: {args.skip_pipeline}"
        f"  Output dir: {args.output_dir or 'default (from settings)'}"
    )

    output_paths = generate_synthetic_data(
        input_path=args.input,
        base_rows=args.base_rows,
        drift_rows=args.drift_rows,
        skip_pipeline=args.skip_pipeline,
        output_dir=args.output_dir,
    )

    print("\n", "=" * 80)
    print("Synthetic Data Generation Successful")
    print("\nGenerated Files:")
    for file_type, path in output_paths.items():
        print(f"    . {file_type}: {path}")
    print("\n", "=" * 80)


if __name__ == "__main__":
    main()
