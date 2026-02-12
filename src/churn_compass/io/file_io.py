"""
Churn Compass - File I/O Layer

Provides consistent interface for reading/writing CSV and Parquet files.

Features:
- PyArrow backend for performance and memory efficiency
- Local file operations (CSV, Parquet)
- DuckDB integration for OLAP
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

import duckdb
import pandas as pd

from churn_compass import get_settings, setup_logger

logger = setup_logger(__name__)


class FileIO:
    """
    Unified interface for reading/writing CSV/Parquet with PyArrow optimizations.

    Supports:
    - CSV and Parquet formats with PyArrow backend
    - DuckDB for SQL-based analytics
    """

    def __init__(self):
        self._settings = get_settings()

    # CSV
    def read_csv(
        self, filepath: Union[str, Path], chunk_size: Optional[int] = None, **kwargs
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Read CSV file with PyArrow backend.

        :param filepath: Path to CSV file
        :type filepath: Union[str, Path]
        :param chunk_size: Number of rows per chunk (None for full load)
        :type chunk_size: Optional[int]
        :param **kwargs: Additional arguments passed to pd.read_csv
        :return: Dataframe or TextFileReader for chunked reading
        :rtype: Union[DataFrame, TextFileReader]
        :raises FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        try:
            logger.info(
                "Reading CSV file",
                extra={"filepath": str(filepath), "chunk_size": chunk_size},
            )

            if chunk_size:
                return pd.read_csv(
                    filepath,
                    chunksize=chunk_size,
                    engine="pyarrow",
                    dtype_backend="pyarrow",
                    **kwargs,
                )

            df = pd.read_csv(
                filepath, engine="pyarrow", dtype_backend="pyarrow", **kwargs
            )

            logger.info(
                "CSV loaded successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                },
            )
            return df

        except Exception as e:
            logger.error(
                f"Failed to read csv for file: {str(filepath)}",
                extra={
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def write_csv(self, df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
        """
        Write DataFrame to CSV file.

        :param df: DataFrame to write
        :type df: pd.DataFrame
        :param filepath: Destination path
        :type filepath: Union[str, Path]
        :param kwargs: Additional arguments passed to df.to_csv
        """
        filepath = Path(filepath)

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Writing CSV file", extra={"filepath": str(filepath)})
            df.to_csv(filepath, index=False, **kwargs)

            logger.info(
                "CSV written successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                },
            )
        except Exception as e:
            logger.error(
                "Failed to write CSV",
                extra={
                    "filepath": str(filepath),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    # Parquet
    def read_parquet(
        self, filepath: Union[str, Path], columns: Optional[List[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Read Parquet file with PyArrow backend.

        :param filepath: Path to Parquet file
        :type filepath: Union[str, Path]
        :param columns: List of columns to read (None = all)
        :type columns: Optional[List[str]]
        :param kwargs: Additional arguments passed to pd.read_parquet
        :return: DataFrame with loaded data
        :rtype: DataFrame
        :raises FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Parquet file not found: {filepath}")

        try:
            logger.info(
                "Reading Parquet file",
                extra={
                    "filepath": str(filepath),
                },
            )
            df = pd.read_parquet(
                filepath,
                columns=columns,
                engine="pyarrow",
                dtype_backend="pyarrow",
                **kwargs,
            )

            logger.info(
                "Parquet loaded successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                },
            )
            return df

        except Exception as e:
            logger.error(
                "Failed to read Parquet",
                extra={
                    "filepath": str(filepath),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def write_parquet(
        self,
        df: pd.DataFrame,
        filepath: Union[str, Path],
        compression: Literal["snappy", "gzip", "brotli", "zstd"] = "snappy",
        **kwargs,
    ) -> None:
        """
        Write DataFrame to Parquet file

        :param df: DataFrame to write
        :type df: pd.DataFrame
        :param filepath: Destination path
        :type filepath: Union[str, Path]
        :param compression: Compression algorithm (snappy, gzip, brotli)
        :type compression: Literal["snappy", "gzip", "brotli", "zstd"]
        :param kwargs: Additional arguments passed to df.to_parquet
        """
        filepath = Path(filepath)

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Writing Parquet file",
                extra={
                    "filepath": str(filepath),
                },
            )

            df.to_parquet(
                filepath,
                compression=compression,
                index=False,
                engine="pyarrow",
                **kwargs,
            )

            logger.info(
                "Parquet written successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "compression": compression,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to write Parquet",
                extra={
                    "filepath": str(filepath),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def query_with_duckdb(
        self, sql: str, files: Optional[dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query on local files using DuckDB.

        :param sql: SQL query string
        :type sql: str
        :param files: Dict mapping table names to file paths (optional)
        :type files: Optional[dict[str, str]]
        :return: Query results as DataFrame with PyArrow backend
        :rtype: DataFrame
        :raises ValueError: If invalid table name or unsupported file type

        Example:
        >>> io = FileIO()
        >>> df = io.query_with_duckdb(
        ...      "SELECT * FROM customers WHERE churn_probability > 0.7",
        ...      files = {"customers": "data/processed/predictions.parquet"}
        ... )
        """
        conn = None
        try:
            conn = duckdb.connect(":memory:")

            if files:
                for table_name, filepath in files.items():
                    fp = str(filepath)

                    # Hard safety checks
                    if not table_name.isidentifier():
                        raise ValueError(f"Invalid table name: {table_name}")

                    if fp.endswith(".parquet"):
                        rel = conn.read_parquet(fp)

                    elif fp.endswith(".csv"):
                        rel = conn.read_csv(fp, auto_detect=True)

                    else:
                        raise ValueError(f"Unsupported file type: {fp}")

                    conn.register(table_name, rel)

            logger.info("Executing DuckDB analytics query")
            result_df = conn.execute(sql).fetch_df()
            df = pd.DataFrame(result_df).convert_dtypes(dtype_backend="pyarrow")

            logger.info(
                "DuckDB query complete",
                extra={"rows_returned": len(df), "columns": len(df.columns)},
            )
            return df

        except Exception as e:
            logger.error(
                "DuckDB query failed",
                extra={"error_type": type(e).__name__},
                exc_info=True,
            )
            raise

        finally:
            if conn is not None:
                conn.close()
