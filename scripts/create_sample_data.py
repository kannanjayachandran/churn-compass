#!/usr/bin/env python3
"""
Churn Compass - Custom Synthetic Data Generator

Generates synthetic customer churn datasets using a lightweight statistical
approach. This tool utilizes a Gaussian copula-based sampler to maintain
feature correlations and a conditional probability model for the churn status
(Exited), which is derived from key variables such as Age and Balance.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from churn_compass import (
    get_settings,
    log_execution_time,
    setup_logger,
)
from churn_compass.config import generate_customer_uuid
from churn_compass.io import FileIO

logger = setup_logger("Synthetic Data Generator")


@log_execution_time(logger)
def load_and_clean(input_path: str) -> pd.DataFrame:
    """
    Load reference CSV and remove duplicates/nulls.

    :param input_path: Reference CSV file path
    :type input_path: str
    :return: Cleaned DataFrame
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If input file doesn't exist
    """
    logger.info(f"Loading and cleaning data from: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = FileIO().read_csv(input_path, validate_uuid=False)

    logger.info(
        "Real data loaded",
        extra={"shape": df.shape, "columns": df.columns.tolist()},
    )

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Removing rows with missing values (total: {n_missing})")
        df = df.dropna()

    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Removing {n_duplicates} duplicate rows")
        df = df.drop_duplicates()

    logger.info(f"Cleaned dataset. Final shape: {df.shape}")

    return df


@log_execution_time(logger)
def configure_metadata(df: pd.DataFrame) -> Dict:
    """
    Create metadata describing column types and statistics.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Metadata dictionary
    :rtype: Dict
    """
    numerical = []
    categorical = []
    binary = []
    id_cols = []

    for col in df.columns:
        if col.lower() in (
            "rownumber",
            "row_number",
            "customerid",
            "customer_id",
            "customer_uuid",
        ):
            id_cols.append(col)
            continue

        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(
            df[col]
        ):
            nunique = df[col].nunique(dropna=True)
            if nunique <= 5:
                unique_vals = set(df[col].dropna().unique())
                if unique_vals <= {0, 1}:
                    binary.append(col)
                else:
                    categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

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

    logger.info("Metadata configured", extra={"metadata": metadata})
    return metadata


@log_execution_time(logger)
def apply_drift(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply drift transformations to synthetic data.

    Drift includes:
    1. Age drift (+5 years)
    2. Feature-driven churn drift (target ~28%)
    3. Noise injection to numerical features
    4. Clipping to original data ranges

    :param df: Synthetic DataFrame
    :type df: pd.DataFrame
    :param original_df: Original DataFrame for reference ranges
    :type original_df: pd.DataFrame
    :return: Drifted DataFrame
    :rtype: pd.DataFrame
    """
    logger.info("Applying drift transformations")

    df_drift = df.copy()

    # Age drift (+5 years)
    if "Age" in df_drift.columns and "Age" in original_df.columns:
        original_age_mean = original_df["Age"].mean()
        df_drift["Age"] = df_drift["Age"] + 5
        df_drift["Age"] = df_drift["Age"].clip(
            lower=original_df["Age"].min(),
            upper=original_df["Age"].max() + 10,
        )
        new_age_mean = df_drift["Age"].mean()
        logger.info(
            f"Age drift: {original_age_mean:.2f} → {new_age_mean:.2f} "
            f"(+{new_age_mean - original_age_mean:.2f} years)"
        )

    # Feature-driven churn drift (target ~28%)
    if "Exited" in df_drift.columns:
        target_churn = 0.28
        current_churn = df_drift["Exited"].mean()

        risk = np.zeros(len(df_drift))
        if "Age" in df_drift.columns:
            risk += 0.03 * df_drift["Age"]
        if "Balance" in df_drift.columns:
            risk += 0.000002 * df_drift["Balance"]
        if "IsActiveMember" in df_drift.columns:
            risk += -1.2 * df_drift["IsActiveMember"].astype(int)
        if "NumOfProducts" in df_drift.columns:
            risk += 0.2 * (df_drift["NumOfProducts"] == 1)

        # logistic transform → probability (P(churn | features))
        prob = 1 / (1 + np.exp(-risk))

        # Stochastic label assignment based on computed probabilities (bernoulli sampling)
        df_drift["Exited"] = (np.random.rand(len(df_drift)) < prob).astype(int)

        new_churn = df_drift["Exited"].mean()
        if abs(new_churn - target_churn) > 0.01:
            scale = target_churn / max(new_churn, 1e-6)
            prob_calibrated = np.clip(prob * scale, 0, 1)
            df_drift["Exited"] = (
                np.random.rand(len(df_drift)) < prob_calibrated
            ).astype(int)

        final_churn = df_drift["Exited"].mean()
        logger.info(
            f"Churn drift: {current_churn:.2%} → {final_churn:.2%} (target: {target_churn:.2%})"
        )

    # Noise injection to numerical features
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
            if col in original_df.columns:
                df_drift[col] = df_drift[col].clip(
                    lower=original_df[col].min(),
                    upper=original_df[col].max(),
                )

    logger.info(f"Drift applied to {len(drift_cols)} additional columns")
    return df_drift


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index."""
    eps = 1e-8
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected, quantiles)
    unique_edges = np.unique(bin_edges)

    if len(unique_edges) < 2:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=unique_edges)
    a_counts, _ = np.histogram(actual, bins=unique_edges)

    e_perc = e_counts / (len(expected) + eps)
    a_perc = a_counts / (len(actual) + eps)

    psi_value = np.sum((e_perc - a_perc) * np.log((e_perc + eps) / (a_perc + eps)))
    return float(psi_value)


def ks_report(
    real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]
) -> Dict[str, float]:
    """Calculate Kolmogorov-Smirnov statistics for numerical columns."""
    result = {}
    for col in cols:
        if (
            col in real.columns
            and col in synth.columns
            and pd.api.types.is_numeric_dtype(real[col])
        ):
            try:
                stat = stats.ks_2samp(real[col].dropna(), synth[col].dropna()).statistic
                result[col] = float(stat)
            except Exception:
                result[col] = None
        else:
            result[col] = None
    return result


def corr_matrix_diff(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    """Calculate mean absolute difference in correlation matrices."""
    cols_present = [
        c
        for c in cols
        if c in real.columns
        and c in synth.columns
        and pd.api.types.is_numeric_dtype(real[c])
    ]
    if len(cols_present) < 2:
        return 0.0

    r_corr = real[cols_present].corr().fillna(0).values
    s_corr = synth[cols_present].corr().fillna(0).values
    diff = np.abs(r_corr - s_corr)
    return float(np.nanmean(diff))


@log_execution_time(logger)
def evaluate_synthetic_data_quality(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict
) -> Dict:
    """
    Evaluate synthetic data quality using KS, PSI, and correlation metrics.

    :param real_data: Original DataFrame
    :type real_data: pd.DataFrame
    :param synthetic_data: Synthetic DataFrame
    :type synthetic_data: pd.DataFrame
    :param metadata: Metadata dictionary
    :type metadata: Dict
    :return: Quality metrics
    :rtype: Dict
    """
    logger.info("Running quality evaluation (KS / PSI / CorrDiff)")

    numerical = metadata.get("numerical", [])

    ks_vals = ks_report(real_data, synthetic_data, numerical)

    psi_vals = {}
    for col in numerical:
        if col in real_data.columns and col in synthetic_data.columns:
            psi_vals[col] = psi(
                real_data[col].dropna().values,
                synthetic_data[col].dropna().values,
            )
        else:
            psi_vals[col] = None

    corr_diff = corr_matrix_diff(real_data, synthetic_data, numerical)

    overall = {
        "ks": ks_vals,
        "psi": psi_vals,
        "corr_matrix_mean_abs_diff": corr_diff,
    }

    logger.info("Quality evaluation completed", extra={"corr_diff": corr_diff})
    return overall


class ConditionalChurnModel:
    """Logistic regression model for P(Exited=1 | Age, Balance)."""

    def __init__(
        self,
        age_col: str = "Age",
        balance_col: str = "Balance",
        target_col: str = "Exited",
    ):
        self.age_col = age_col
        self.balance_col = balance_col
        self.target_col = target_col
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None
        self.base_rate: float = 0.0

    def fit(self, df: pd.DataFrame):
        """Fit conditional churn model on real data."""
        if (
            self.age_col not in df.columns
            or self.balance_col not in df.columns
            or self.target_col not in df.columns
        ):
            raise ValueError("DataFrame must contain Age, Balance, and Exited columns")

        X = df[[self.age_col, self.balance_col]].fillna(0.0)
        y = df[self.target_col].fillna(0).astype(int).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(X_scaled, y)

        self.base_rate = float(np.mean(y))
        logger.info(f"Fitted conditional churn model (base_rate={self.base_rate:.4f})")
        return self

    def sample(self, synthetic_df: pd.DataFrame) -> np.ndarray:
        """Sample churn labels based on fitted model."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted")

        Xs = synthetic_df[[self.age_col, self.balance_col]].copy().fillna(0.0)
        Xs_scaled = self.scaler.transform(Xs)
        probs = self.model.predict_proba(Xs_scaled)[:, 1]
        return np.random.binomial(1, probs)


class TabularSynthesizer:
    """
    Gaussian-copula-based synthesizer for tabular data.

    Numerical: Empirical CDF → Gaussian → multivariate normal → inverse CDF
    Categorical: Empirical sampling
    Binary: Independent Bernoulli
    Exited: Conditional model based on Age and Balance
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

        self.num_empirical_sorted = {}
        self.cov = None
        self.cat_distributions = {}
        self.binary_rates = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit synthesizer on real data."""
        gaussian_data = []
        for col in self.numerical_cols:
            if col not in df.columns:
                continue
            vals = df[col].astype(float).dropna().values
            if len(vals) == 0:
                continue
            sorted_vals = np.sort(vals)
            self.num_empirical_sorted[col] = sorted_vals

            ranks = stats.rankdata(vals) / (len(vals) + 1.0)
            gaussian = stats.norm.ppf(ranks)
            gaussian_data.append(gaussian)

        if len(gaussian_data) >= 1:
            gaussian_stack = np.vstack(gaussian_data).T
            if gaussian_stack.ndim == 1:
                gaussian_stack = gaussian_stack.reshape(-1, 1)
            raw_cov = np.cov(gaussian_stack, rowvar=False)
            self.cov = np.atleast_2d(raw_cov)
        else:
            self.cov = None

        for col in self.categorical_cols:
            if col in df.columns:
                counts = df[col].value_counts(dropna=True)
                if counts.sum() > 0:
                    self.cat_distributions[col] = (
                        counts.index.tolist(),
                        (counts / counts.sum()).values.tolist(),
                    )

        for col in self.binary_cols:
            if col in df.columns and col != "Exited":
                vals = df[col].dropna().astype(int).values
                if len(vals) > 0:
                    rate = float(np.mean(vals))
                    self.binary_rates[col] = np.clip(rate, 0.0, 1.0)

        if self.churn_model is not None:
            try:
                self.churn_model.fit(df)
            except Exception as e:
                logger.warning(f"Churn model fitting failed: {e}")
                self.churn_model = None

        self._fitted = True
        logger.info("Synthesizer fitted successfully")
        return self

    def sample(
        self, n_rows: int, start_row_number: int = 1, start_customer_id: int = 100000
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        :param n_rows: Number of rows to generate
        :type n_rows: int
        :param start_row_number: Starting RowNumber value
        :type start_row_number: int
        :param start_customer_id: Starting CustomerId value
        :type start_customer_id: int
        :return: Synthetic DataFrame
        :rtype: pd.DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Synthesizer not fitted. Call fit() first.")

        out = pd.DataFrame(index=range(n_rows))

        for col in self.id_cols:
            col_lower = col.lower()
            if col_lower in ("rownumber", "row_number"):
                out[col] = np.arange(start_row_number, start_row_number + n_rows)
            elif col_lower in ("customerid", "customer_id"):
                out[col] = np.arange(start_customer_id, start_customer_id + n_rows)

        if self.cov is not None and len(self.num_empirical_sorted) > 0:
            cols = [c for c in self.numerical_cols if c in self.num_empirical_sorted]
            dim = len(cols)
            cov = np.array(self.cov, copy=True)
            mean = np.zeros(dim)
            try:
                z = np.random.multivariate_normal(mean=mean, cov=cov, size=n_rows)
            except Exception:
                z = np.random.normal(size=(n_rows, dim))

            for i, col in enumerate(cols):
                u = stats.norm.cdf(z[:, i])
                sorted_vals = self.num_empirical_sorted[col]
                out[col] = np.quantile(sorted_vals, u)
        else:
            for col, sorted_vals in self.num_empirical_sorted.items():
                u = np.random.rand(n_rows)
                out[col] = np.quantile(sorted_vals, u)

        for col, (choices, probs) in self.cat_distributions.items():
            out[col] = np.random.choice(choices, size=n_rows, p=probs)

        for col, rate in self.binary_rates.items():
            valid_rate = np.clip(rate, 0.0, 1.0) if not np.isnan(rate) else 0.5
            out[col] = np.random.binomial(1, valid_rate, size=n_rows)

        if self.churn_model is not None:
            try:
                out["Exited"] = self.churn_model.sample(out)
            except Exception as e:
                logger.warning(f"Churn sampling failed: {e}, using fallback")
                base_rate = getattr(self.churn_model, "base_rate", 0.1)
                out["Exited"] = np.random.binomial(1, base_rate, size=n_rows)
        else:
            rate = self.binary_rates.get("Exited", 0.1)
            out["Exited"] = np.random.binomial(1, rate, size=n_rows)

        return out


def add_customer_uuids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic customer_uuid column based on CustomerId.

    :param df: DataFrame with CustomerId column
    :type df: pd.DataFrame
    :return: DataFrame with customer_uuid column
    :rtype: pd.DataFrame
    :raises ValueError: If CustomerId column missing
    """
    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId column required for UUID generation")

    df = df.copy()
    df["customer_uuid"] = df["CustomerId"].apply(
        lambda x: str(generate_customer_uuid(x))
    )

    logger.info(
        "Customer UUIDs generated",
        extra={"unique_customers": df["customer_uuid"].nunique()},
    )
    return df


@log_execution_time(logger)
def generate_synthetic_data(
    input_path: str,
    base_rows: int = 2500,
    drift_rows: int = 1000,
    output_dir: Optional[str] = None,
    settings=None,
) -> Dict[str, str]:
    """
    Generate synthetic datasets with optional drift.

    :param input_path: Path to real dataset CSV
    :type input_path: str
    :param base_rows: Number of rows for base dataset
    :type base_rows: int
    :param drift_rows: Number of rows for drifted dataset
    :type drift_rows: int
    :param output_dir: Custom output directory
    :type output_dir: Optional[str]
    :param settings: Settings instance
    :type settings: Optional
    :return: Paths to generated files
    :rtype: Dict[str, str]
    """
    logger.info("Synthetic data generation started")

    if output_dir:
        raw_dir = Path(output_dir) / "raw"
        processed_dir = Path(output_dir) / "processed"
    else:
        if settings is None:
            raise RuntimeError("settings must be provided when output_dir is not set")
        raw_dir = Path(settings.data_raw_dir)
        processed_dir = Path(settings.data_processed_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = raw_dir / "synthetic_raw_sample.csv"
    base_parquet_path = processed_dir / "synthetic_sample.parquet"
    drift_parquet_path = processed_dir / "synthetic_sample_drifted.parquet"
    quality_report_path = processed_dir / "synthetic_quality_report.json"

    real_df = load_and_clean(input_path)
    metadata = configure_metadata(real_df)

    numerical_cols = metadata.get("numerical", [])
    categorical_cols = metadata.get("categorical", [])
    binary_cols = metadata.get("binary", [])
    id_cols = metadata.get("id_cols", [])

    required_id_cols = ["RowNumber", "CustomerId"]
    for col in required_id_cols:
        if col not in id_cols:
            id_cols.append(col)

    churn_model = ConditionalChurnModel()
    synthesizer = TabularSynthesizer(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        id_cols=id_cols,
        churn_model=churn_model,
    )
    synthesizer.fit(real_df)

    synthetic_base = synthesizer.sample(n_rows=base_rows)
    synthetic_base = add_customer_uuids(synthetic_base)

    synthetic_drift_raw = synthesizer.sample(
        n_rows=drift_rows, start_customer_id=100000 + base_rows
    )
    synthetic_drift = apply_drift(synthetic_drift_raw, real_df)
    synthetic_drift = add_customer_uuids(synthetic_drift)

    logger.info(f"Saving raw synthetic CSV to {raw_csv_path}")
    FileIO().write_csv(synthetic_base, raw_csv_path)

    logger.info("Writing processed synthetic data (with customer_uuid)")
    FileIO().write_parquet(synthetic_base, base_parquet_path)
    FileIO().write_parquet(synthetic_drift, drift_parquet_path)

    quality_metrics = evaluate_synthetic_data_quality(real_df, synthetic_base, metadata)

    with open(quality_report_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "base_rows": base_rows,
                "drift_rows": drift_rows,
                "real_data_shape": list(real_df.shape),
                "quality_metrics": quality_metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Quality report saved to {quality_report_path}")

    return {
        "raw_csv": str(raw_csv_path),
        "base_parquet": str(base_parquet_path),
        "drift_parquet": str(drift_parquet_path),
        "quality_report": str(quality_report_path),
    }


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic churn dataset with statistical quality evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to real dataset CSV for training synthesizer",
    )

    parser.add_argument(
        "--base-rows",
        type=int,
        default=2500,
        help="Number of rows for base synthetic dataset (default: 2500)",
    )

    parser.add_argument(
        "--drift-rows",
        type=int,
        default=1000,
        help="Number of rows for drifted synthetic dataset (default: 1000)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: uses settings paths)",
    )

    return parser.parse_args()


def main():
    """CLI entry point."""
    settings = get_settings()
    args = parse_arguments()

    logger.info(
        "Starting synthetic data generation:\n"
        f"  Input: {args.input}\n"
        f"  Base rows: {args.base_rows}\n"
        f"  Drift rows: {args.drift_rows}\n"
        f"  Output dir: {args.output_dir or 'default (from settings)'}"
    )

    output_paths = generate_synthetic_data(
        input_path=args.input,
        base_rows=args.base_rows,
        drift_rows=args.drift_rows,
        output_dir=args.output_dir,
        settings=settings,
    )

    print("\n" + "=" * 80)
    print("✓ Synthetic Data Generation Successful")
    print("\nGenerated Files:")
    for file_type, path in output_paths.items():
        print(f"  • {file_type}: {path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
