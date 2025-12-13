"""
Churn Compass - Database I/O Layer

PostgreSQL for production, DuckDB for local development.

Features:
- PostgreSQL connection management with SQLAlchemy
- Read queries to DataFrame
- Execute DDL/DML statements
- DuckDB fallback for local development
- Connection pooling and error handling
"""

from typing import Optional, cast
from contextlib import contextmanager

import pandas as pd
import duckdb
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.sql import Executable

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger


logger = setup_logger(__name__)


class DatabaseIO:
    """
    Database operations for PostgreSQL and DuckDB.

    db_type:
    - 'postgres': Production PostgreSQL
    - 'duckdb': Local DuckDB
    """

    def __init__(self, db_type: Optional[str] = None):
        self.db_type = db_type or settings.db_type
        self._engine: Optional[Engine] = None
        logger.info("Initializing DatabaseIO", extra={"db_type": self.db_type})

    
    # PostgreSQL
    @property
    def engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine for PostgreSQL
        
        :return: SQLAlchemy Engine
        :rtype: Engine
        """
        if self.db_type != "postgres":
            raise ValueError("SQLAlchemy engine is only available for postgres")

        if self._engine is None:
            try:
                uri = settings.get_postgres_uri()
                self._engine = create_engine(
                    uri,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    echo=False, # Set True for SQL logging
                )

                # sanity check
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                logger.info("PostgreSQL connection established")

            except Exception as e:
                logger.exception("Failed to initialize PostgreSQL engine", extra={
                    "status": "error", 
                    "error_type": type(e).__name__, 
                }, )
                raise

        return self._engine

    # Connections
    @contextmanager
    def get_connection(self):
        """
        Context-managed database connection.

        - Postgres: transactional connection
        - DuckDB: open-per-use to avoid file locking
        """
        if self.db_type == "postgres":
            with self.engine.begin() as conn:
                yield conn
        else:
            conn = duckdb.connect(str(settings.duckdb_path))
            try:
                yield conn
            finally:
                conn.close()

    # Read Queries
    def read_query(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        :param query: SQL query string
        :type query: str
        :param params: Query parameters
        :type params: Optional[dict]
        :return: Query results as DF
        :rtype: DataFrame
        """
        try:
            logger.info(
                "Executing query",
                extra={"db_type": self.db_type, "query_preview": query[:100]},
            )

            if self.db_type == "postgres":
                with self.get_connection() as conn:
                    sa_conn = cast(Connection, conn)
                    df = pd.read_sql(text(query), sa_conn, params=params)

            else:  # duckdb
                if params:
                    raise ValueError("DuckDB does not support named parameters")

                with self.get_connection() as conn:
                    duck_conn = cast(duckdb.DuckDBPyConnection, conn)
                    df = duck_conn.execute(query).fetchdf()

            logger.info(
                "Query completed",
                extra={"rows_returned": len(df), "columns": len(df.columns)},
            )
            return df

        except Exception as e:
            logger.exception("Query execution failed", 
                             extra=
                             {"status": "error", "error_type": type(e).__name__,}, 
                             exc_info=True
                             )
            raise


    # Execute Statements
    def execute(self, query: str, params: Optional[dict] = None) -> None:
        """
        Execute SQL statements without returning results (DDL, DML)
        
        :param query: SQL statement
        :type query: str
        :param params: Query parameters
        :type params: Optional[dict]
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
            logger.exception("Statement execution failed", 
                             extra={"status": "error", "error_type": type(e).__name__, }, 
                             exc_info=True)
            raise


    # Utilities
    def table_exists(self, table_name: str) -> bool:
        """
        Check whether a table exists.
        """
        try:
            if self.db_type == "postgres":
                inspector = inspect(self.engine)
                return table_name in inspector.get_table_names()
            else:
                with duckdb.connect(str(settings.duckdb_path)) as conn:
                    result = conn.execute(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                        [table_name],
                    ).fetchone()
                    return bool(result and result[0] > 0)

        except Exception:
            logger.exception("Table existence check failed")
            return False
   
    # Cleanup
    def close(self):
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL engine disposed")
