"""
Churn Compass - Database I/O Layer

PostgreSQL for OLTP, DuckDB for OLAP analytics.

Features:
    - Handles PyArrow to SQL-compatible dtype conversion.
"""

from contextlib import contextmanager
from typing import Literal, Optional, cast

import duckdb
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql import Executable

from churn_compass import get_settings, setup_logger

logger = setup_logger(__name__)


class DatabaseIO:
    """
    Database operations with PyArrow optimization.

    db_type:
    - 'postgres': OLTP
    - 'duckdb': OLAP
    """

    def __init__(self, db_type: Optional[str] = None):
        self._settings = get_settings()
        self.db_type = db_type or self._settings.db_type
        self._engine: Optional[Engine] = None

        logger.info("Initializing DatabaseIO", extra={"db_type": self.db_type})

    # PostgreSQL
    @property
    def engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine for PostgreSQL

        :return: SQLAlchemy Engine
        :rtype: Engine
        :raises ValueError: If db_type is not postgres
        """
        if self.db_type != "postgres":
            raise ValueError("SQLAlchemy engine is only available for postgres")

        if self._engine is None:
            try:
                uri = self._settings.get_postgres_uri()
                self._engine = create_engine(
                    uri,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    echo=False,  # Set True for SQL logging
                )

                # sanity check
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                logger.info("PostgreSQL connection established")

            except Exception as e:
                logger.exception(
                    "Failed to initialize PostgreSQL engine",
                    extra={
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return self._engine

    # Connections
    @contextmanager
    def get_connection(self):
        """
        Context-managed database connection.

        - Postgres: transactional connection with auto-commit
        - DuckDB: Ephemeral (open-per-use) connection (prevents file locks)
        """
        if self.db_type == "postgres":
            with self.engine.begin() as conn:
                yield conn
        else:
            conn = duckdb.connect(str(self._settings.duckdb_path))
            try:
                yield conn
            finally:
                conn.close()

    def read_query(
        self, query: str, params: Optional[dict] = None, use_pyarrow: bool = True
    ) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        :param query: SQL query string
        :type query: str
        :param params: Query parameters
        :type params: Optional[dict]
        :param use_pyarrow: Use PyArrow backend for result DataFrame
        :type use_pyarrow: bool
        :return: Query results as DataFrame
        :rtype: pd.DataFrame
        :raises ValueError: If DuckDB called with params
        """
        try:
            logger.info(
                "Executing query",
                extra={"db_type": self.db_type, "query_preview": query[:100]},
            )

            if self.db_type == "postgres":
                with self.get_connection() as conn:
                    sa_conn = cast(Connection, conn)
                    df = pd.read_sql(
                        text(query),
                        sa_conn,
                        params=params,
                        dtype_backend="pyarrow" if use_pyarrow else "numpy_nullable",
                    )

            else:  # duckdb
                if params:
                    raise ValueError("DuckDB does not support parameterized queries")

                with self.get_connection() as conn:
                    duck_conn = cast(duckdb.DuckDBPyConnection, conn)
                    result = duck_conn.execute(query).fetchdf()
                    df = (
                        pd.DataFrame(result).convert_dtypes(dtype_backend="pyarrow")
                        if use_pyarrow
                        else result
                    )

            logger.info(
                "Query completed",
                extra={"rows_returned": len(df), "columns": len(df.columns)},
            )
            return df

        except Exception as e:
            logger.error(
                "Query execution failed",
                extra={
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def execute(self, query: str, params: Optional[dict] = None) -> None:
        """
        Execute SQL statements without returning results (DDL, DML).

        :param query: SQL statement
        :type query: str
        :param params: Query parameters
        :type params: Optional[dict]
        :raises ValueError: If DuckDB called with params
        """
        try:
            logger.info("Executing statement", extra={"db_type": self.db_type})

            if self.db_type == "postgres":
                with self.get_connection() as conn:
                    sa_conn = cast(Connection, conn)
                    stmt = cast(Executable, text(query))
                    sa_conn.execute(stmt, params or {})

            else:  # duckdb
                if params:
                    raise ValueError("DuckDB does not support named parameters")

                with self.get_connection() as conn:
                    duck_conn = cast(duckdb.DuckDBPyConnection, conn)
                    duck_conn.execute(query)

            logger.info("Statement executed successfully")

        except Exception as e:
            logger.exception(
                "Statement execution failed",
                extra={
                    "status": "error",
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: Literal["replace", "append", "fail"] = "replace",
        index: bool = False,
    ) -> None:
        """
        Write DataFrame to database table.

        Automatically convert PyArrow dtypes to SQL-compatible types.

        :param df: DataFrame to persist
        :type df: pd.DataFrame
        :param table_name: Target table name
        :type table_name: str
        :param if_exists: Behavior if table exists
        :type if_exists: Literal['replace' | 'append' | 'fail']
        :param index: Whether to write DataFrame index
        :type index: bool
        """
        try:
            logger.info(
                "Writing DataFrame to table",
                extra={
                    "db_type": self.db_type,
                    "table": table_name,
                    "rows": len(df),
                    "if_exists": if_exists,
                },
            )

            df_to_write = df.copy()
            if hasattr(df_to_write, "convert_dtypes"):
                if str(df_to_write.attrs.get("dtype_backend")) == "pyarrow" or any(
                    "pyarrow" in str(dtype) for dtype in df_to_write.dtypes
                ):
                    logger.debug("Converting PyArrow dtypes for SQL compatibility")
                    df_to_write = df_to_write.convert_dtypes(
                        dtype_backend="numpy_nullable"
                    )

            if self.db_type == "postgres":
                with self.get_connection() as conn:
                    sa_conn = cast(Connection, conn)
                    df.to_sql(
                        table_name,
                        sa_conn,
                        if_exists=if_exists,
                        index=index,
                        method="multi",
                    )
            else:  # duckdb
                with self.get_connection() as conn:
                    duck_conn = cast(duckdb.DuckDBPyConnection, conn)

                    if if_exists == "replace":
                        duck_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    elif if_exists == "fail":
                        tables = duck_conn.execute(
                            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                            [table_name],
                        ).fetchall()
                        if tables:
                            raise ValueError(f"Table {table_name} already exists")

                    duck_conn.register("tmp_write_df", df_to_write)

                    if if_exists == "append":
                        duck_conn.execute(
                            f"INSERT INTO {table_name} SELECT * FROM tmp_write_df"
                        )
                    else:
                        duck_conn.execute(
                            f"CREATE TABLE {table_name} AS SELECT * FROM tmp_write_df"
                        )
                        duck_conn.unregister("tmp_write_df")

            logger.info(
                "Table write completed",
                extra={"table": table_name, "rows_written": len(df)},
            )

        except Exception as e:
            logger.exception(
                "Failed to write DataFrame to table",
                extra={
                    "table": table_name,
                    "db_type": self.db_type,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise

    # Utilities
    def table_exists(self, table_name: str) -> bool:
        """
        Check whether a table exists in database.
        :param table_name: Name of table to check
        :type table_name: str
        :return: True if table exists
        :rtype: bool
        """
        try:
            if self.db_type == "postgres":
                inspector = inspect(self.engine)
                return table_name in inspector.get_table_names()
            else:
                with duckdb.connect(str(self._settings.duckdb_path)) as conn:
                    result = conn.execute(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                        [table_name],
                    ).fetchone()
                    return bool(result and result[0] > 0)

        except Exception as e:
            logger.error(
                "Table existence check failed",
                extra={"table": table_name, "error_type": type(e).__name__},
                exc_info=True,
            )
            return False

    # Cleanup
    def close(self):
        """Dispose of database connection and cleanup resources."""
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL engine disposed")
