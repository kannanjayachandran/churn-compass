"""
Churn Compass - File I/O Layer

Provides consistent interface for reading/writing CSV and Parquet files.

Features:
- Local file operations (CSV, Parquet)
- DuckDB integration for fast SQL queries during development
"""

from pathlib import Path
from typing import Optional, Union, List, Literal
import pandas as pd
import duckdb

from churn_compass import setup_logger


logger = setup_logger(__name__)


class FileIO:
    """
    Unified interface for reading/writing CSV/Parquet and running DuckDB queries.

    Supports:
    - CSV and Parquet formats
    - DuckDB for SQL queries
    """

    # CSV
    def read_csv(
        self, filepath: Union[str, Path], chunk_size: Optional[int] = None, **kwargs
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Read CSV file from local filesystem

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
                "Loaded CSV successfully",
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
        Write DataFrame to CSV file

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
        self, filepath: Union[str, Path], columns: Optional[List[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Read Parquet file

        :param filepath: Path to Parquet file
        :type filepath: Union[str, Path]
        :param columns: List of columns to read (None = all)
        :type columns: Optional[List[str]]
        :param kwargs: Additional arguments passed to pd.read_parquet
        :return: DataFrame with loaded data
        :rtype: DataFrame
        """
        filepath = Path(filepath)

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
                "Loaded Parquet successfully",
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
        compression: Literal["snappy", "gzip", "brotli"] = "snappy",
        **kwargs,
    ) -> None:
        """
        Write DataFrame to Parquet file

        :param df: DataFrame to write
        :type df: pd.DataFrame
        :param filepath: Destination path
        :type filepath: Union[str, Path]
        :param compression: Compression algorithm (snappy, gzip, brotli)
        :type compression: str
        :param kwargs: Additional arguments passed to df.to_parquet
        """
        filepath = Path(filepath)

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Writing Parquet to file",
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

    def run_internal_analytics_with_duckdb(
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

        This method is used for system automation and internal analytics (Internal Use Only).

        Example:
        >>> io = FileIO()
        >>> df = io.run_internal_analytics_with_duckdb(
        ...      "SELECT * FROM customers WHERE churn = 1",
        ...      files = {"customers": "data/processed/customers.parquet"}
        ... )

        How it works:
        - Creates in-memory DuckDB instance
        - Register local files (csv / parquet) as SQL tables or views
        - Run an arbitrary SQL query against them
        - Return a pandas DataFrame with the results
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

            logger.info(f"Executing DuckDB query: {sql[:10]}...")
            df = conn.execute(sql).fetch_df()

            logger.info(
                "DuckDB query complete",
                extra={"rows_returned": len(df), "columns": len(df.columns)},
            )
            return df

        except Exception:
            logger.error("DuckDB query failed", exc_info=True)
            raise

        finally:
            if conn is not None:
                conn.close()
