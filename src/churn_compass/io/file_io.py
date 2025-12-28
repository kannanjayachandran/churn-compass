"""
Churn Compass - File I/O Layer

Abstracted file operations supporting local filesystem and S3.
Provides consistent interface for reading/writing CSV and Parquet files.

Features:
- Local file operations (CSV, Parquet)
- S3 support (placeholder for production)
- DuckDB integration for fast SQL queries on local files
"""

from pathlib import Path
from typing import Optional, Union, List, Literal
import pandas as pd
import duckdb
import psutil

from churn_compass import settings, setup_logger


logger = setup_logger(__name__)


class FileIO:
    """
    Unified interface for reading/writing CSV/Parquet and running DuckDB queries.

    Supports:
    - CSV and Parquet formats
    - Local filesystem
    - S3 (When enabled)
    - DuckDB for SQL queries
    """

    def __init__(self):
        self.s3_enabled = settings.s3_enabled
        self.s3_bucket = settings.s3_bucket
        self.s3_prefix = settings.s3_prefix

    # CSV
    def read_csv(self, filepath: Union[str, Path], chunk_size: Optional[int] = None, **kwargs) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Read CSV file from local filesystem or S3.

        :param filepath: Path to CSV file
        :type filepath: Union[str, Path]
        :param chunk_size: Number of rows per chunk (None for all at once)
        :type chunk_size: Optional[int]
        :param **kwargs: Additional arguments passed to pd.read_csv
        :return: Dataframe or TextFileReader for chunked reading
        :rtype: Union[DataFrame, TextFileReader]
        """
        filepath = Path(filepath)

        try:
            if self._is_s3_path(filepath):
                # Placeholder for S3 chunked reading if needed
                return self._read_csv_from_s3(str(filepath), chunksize=chunk_size, **kwargs)
            
            logger.info("Reading CSV from local", extra={"filepath": str(filepath), "chunk_size": chunk_size})
            
            if chunk_size:
                return pd.read_csv(filepath, chunksize=chunk_size, **kwargs)
            
            df = pd.read_csv(filepath, **kwargs)

            logger.info(
                "Loaded CSV successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                    "system_memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
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
        Write DataFrame to CSV file (local or S3)

        :param df: DataFrame to write
        :type df: pd.DataFrame
        :param filepath: Destination path
        :type filepath: Union[str, Path]
        :param kwargs: Additional arguments passed to df.to_csv
        """
        filepath = Path(filepath)
        try:
            if self._is_s3_path(filepath):
                self._write_csv_to_s3(df, str(filepath), **kwargs)
            else:
                filepath.parent.mkdir(parents=True, exist_ok=True)

                logger.info("Writing CSV to local", extra={"filepath": str(filepath)})
                df.to_csv(filepath, index=False, **kwargs)

            logger.info(
                "Wrote CSV successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to write CSV for file: {str(filepath)}",
                extra={
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    # Parquet
    def read_parquet(
        self, filepath: Union[str, Path], columns: Optional[List[str]] = None, engine: str = "pyarrow", **kwargs
    ) -> pd.DataFrame:
        """
        Read Parquet file from local filesystem or S3

        :param filepath: Path to Parquet file
        :type filepath: Union[str, Path]
        :param columns: List of columns to read (None = all)
        :type columns: Optional[List[str]]
        :param engine: Parquet engine (pyarrow or fastparquet)
        :type engine: str
        :param kwargs: Additional arguments passed to pd.read_parquet
        :return: DataFrame with loaded data
        :rtype: DataFrame
        """
        filepath = Path(filepath)

        try:
            if self._is_s3_path(filepath):
                df = self._read_parquet_from_s3(
                    str(filepath), columns=columns, engine=engine, **kwargs
                )
            else:
                logger.info(
                    "Reading Parquet from local path", extra={"filepath": str(filepath), "engine": engine}
                )
                df = pd.read_parquet(filepath, columns=columns, engine=engine, **kwargs)

            logger.info(
                "Loaded Parquet successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                    "system_memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
                },
            )
            return df

        except Exception as e:
            logger.error(
                f"Failed to read Parquet for file: {str(filepath)}",
                extra={
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
        engine: str = "pyarrow",
        **kwargs,
    ) -> None:
        """
        Write DataFrame to Parquet file (local or S3)

        :param df: DataFrame to write
        :type df: pd.DataFrame
        :param filepath: Destination path
        :type filepath: Union[str, Path]
        :param compression: Compression algorithm (snappy, gzip, brotli)
        :type compression: str
        :param engine: Parquet engine (pyarrow or fastparquet)
        :type engine: str
        :param kwargs: Additional arguments passed to df.to_parquet
        """
        filepath = Path(filepath)

        try:
            if self._is_s3_path(filepath):
                self._write_parquet_to_s3(
                    df, str(filepath), compression=compression, engine=engine, **kwargs
                )
            else:
                filepath.parent.mkdir(parents=True, exist_ok=True)

                logger.info(
                    "Writing Parquet to local path", extra={"filepath": str(filepath), "engine": engine}
                )
                df.to_parquet(filepath, compression=compression, index=False, engine=engine, **kwargs)
            logger.info(
                "Wrote Parquet successfully",
                extra={
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "compression": compression,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to write parquet for file: {str(filepath)}",
                extra={
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def query_with_duckdb(
        self, sql: str, files: Optional[dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query on local files using DuckDB

        :param sql: SQL query string
        :type sql: str
        :param files: Dict mapping table names to file paths (optional)
        :type files: Optional[dict]
        :return: Query results as DataFrame
        :rtype: DataFrame

        Example:
            >>> io = FileIO()
            >>> df = io.query_with_duckdb(
            ...      "SELECT * FROM customers WHERE churn = 1",
            ...      files = {"customers": "data/processed/customers.parquet"}
            ... )
        """
        conn = None
        try:
            conn = duckdb.connect(":memory:")

            # Register views
            if files:
                for table_name, filepath in files.items():
                    fp = str(filepath)
                    if fp.endswith(".parquet"):
                        conn.execute(
                            "CREATE VIEW ? AS SELECT * FROM read_parquet(?)",
                            [table_name, fp],
                        )
                    elif fp.endswith(".csv"):
                        conn.execute(
                            "CREATE VIEW ? AS SELECT * FROM read_csv_auto(?)",
                            [table_name, fp],
                        )

            logger.info(f"Executing DuckDB query: {sql[:100]}...")
            df = conn.execute(sql).fetch_df()

            logger.info(
                "DuckDB query complete",
                extra={"rows_returned": len(df), "columns": len(df.columns)},
            )
            return df

        except Exception as e:
            logger.error(
                "DuckDB query failed",
                extra={
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

        finally:
            if conn is not None:
                conn.close()

    # Helpers
    def _is_s3_path(self, filepath: Union[str, Path]) -> bool:
        """
        Check if path is an S3 URI.
        """
        return str(filepath).startswith(("s3://", "s3a://", "s3n://"))

    # S3 placeholders
    def _read_csv_from_s3(self, s3_path: str, **kwargs):
        raise NotImplementedError("S3 support not implemented. Install boto3 or s3fs.")

    def _write_csv_to_s3(self, df, s3_path: str, **kwargs):
        raise NotImplementedError("S3 support not implemented. Install boto3 or s3fs.")

    def _read_parquet_from_s3(self, s3_path: str, **kwargs):
        raise NotImplementedError("S3 support not implemented. Install boto3 or s3fs.")

    def _write_parquet_to_s3(self, df, s3_path: str, **kwargs):
        raise NotImplementedError("S3 support not implemented. Install boto3 or s3fs.")
