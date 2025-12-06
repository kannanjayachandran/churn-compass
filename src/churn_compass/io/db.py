"""
Churn Compass - Database I/O Layer

Database operations for PostgreSQL with fallback to DuckDB for local development.

Features:
- PostgreSQL connection management with SQLAlchemy
- Read queries to DataFrame
- Write DataFrame to tables
- DuckDB fallback for local development
- Connection pooling and error handling
"""

from typing import Optional, List, Any
from contextlib import contextmanager
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool
import duckdb

from churn_compass.config.settings import settings
from churn_compass.logging.logger import setup_logger


logger = setup_logger(__name__)


class DatabaseIO:
    """Database operations supporting PostgreSQL and DuckDB

    Uses settings.db_type to determine which database to use:
    - 'postgres': Production PostgreSQL database
    - 'duckdb': Local DuckDB file for development
    """

    def __init__(self, db_type: Optional[str] = None):
        """
        Initialize database connection
        
        :param db_type: Override settings.db_type ('postgres' or 'duckdb')
        :type db_type: Optional[str]
        """
        self.db_type = db_type or settings.db_type
        self._engine: Optional[Engine] = None
        self._duckdb_conn = None

        logger.info(
            "Initializing DatabaseIO",
            extra={"db_type": self.db_type}
        )

    @property
    def engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine for PostgreSQL

        :return: SQLAlchemy engine
        :rtype: Engine
        """
        if self.db_type != "postgres":
            raise ValueError(f"Engine only available for postgres, current type: {self.db_type}")
        
        if self._engine is None:
            try:
                connection_uri = settings.get_postgres_uri()

                # Create engine with connection pooling
                self._engine = create_engine(
                    connection_uri,
                    pool_pre_ping=True, # verify connections before using
                    pool_size=5,
                    max_overflow=10,
                    echo=False # Set to True for SQL logging
                )

                # Test connection
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                logger.info("PostgreSQL connection established")
            except Exception as e:
                logger.error("Failed to connect to PostgreSQL", exc_info=True)
                raise

        return self._engine
    
    @property
    def duckdb_conn(self):
        """
        Get or create DuckDB connection
        
        :return: DuckDB connection
        """
        if self.db_type != "duckdb":
            raise ValueError(f"DuckDB connection only available for duckdb type, current: {self.db_type}")
        if self._duckdb_conn is None:
            try:
                db_path = str(settings.duckdb_path)
                self._duckdb_conn = duckdb.connect(db_path)
                logger.info("DuckDB connection established", 
                            extra={"filepath": str(db_path)})
            except Exception as e:
                logger.error("Failed to connect to DuckDB", exc_info=True)
                raise
        return self._duckdb_conn
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connection

        Usage:
            with db.get_connection() as conn:
                df = pd.read_sql("SELECT * FROM table", conn)
        """
        if self.db_type == "postgres":
            conn = self.engine.connect()
            try:
                yield conn
            finally:
                conn.close()
        else:
            yield self.duckdb_conn
    
    def read_query(
            self,
            query: str,
            params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        :param query: SQL query string
        :type query: str
        :param params: Query parameters (dict for named params)
        :type params: Optional[dict]
        :return: Query results as DataFrame
        :rtype: DataFrame

        Example:
            >>> db = DatabaseIO()
            >>> df = db.read_query(
            ...     "SELECT * FROM customers WHERE tenure > :min_tenure",
            ...     params={"min_tenure": 5} 
            
            )
        """
        try:
            logger.info(
                "Executing query",
                extra={"query_preview": query[:100], 
                       "db_type": self.db_type}
            )

            if self.db_type == "postgres":
                with self.get_connection() as conn:
                    if params:
                        df = pd.read_sql(text(query), conn, params=params) # type: ignore
                    else:
                        df = pd.read_sql(text(query), conn) # type: ignore
            else:
                # DuckDB
                if params:
                    # DuckDB used $1, $2 style parameters
                    df = self.duckdb_conn.execute(query, list(params.values())).fetchdf()
                else:
                    df = self.duckdb_conn.execute(query).fetchdf()

            logger.info(
                "Query completed successfully",
                extra={"rows_returned": len(df), 
                       "columns": len(df.columns)}
            )

            return df
        
        except Exception as e:
            logger.error("Query execution failed", exc_info=True)
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        :param table_name: Table name to check
        :type table_name: str
        :return: True if table exists, False otherwise
        :rtype: bool
        """
        try:
            if self.db_type == "postgres":
                inspector =  inspect(self.engine)
                exists = table_name in inspector.get_table_names()
            else:
                result = self.duckdb_conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                ).fetchone()
                exists = bool(result and result[0] > 0)

            logger.debug(f"Table '{table_name}' exists: {exists}")
            return exists
        
        except Exception as e:
            logger.error(f"Error checking if table exists: {table_name}", exc_info=True)
            return False
        
    def _should_commit(self, query: str) -> bool:
        """Check whether to commit or not"""
        q = query.strip().lower()
        return q.startswith(("insert", "update", "delete", "create", "drop", "alter", "truncate", "with"))
    
    def execute(self, query: str, params: Optional[dict] = None) -> None:
        """
        Execute SQL statement without returning results (DDL, DML).
        :param query: SQL statement
        :type table_name: str
        :param params: Query parameters

        Example:
            >>> db = DatabaseIO()
            >>> db.execute("CREATE INDEX idx_customer_id ON customers(customer_id)")
        """
        try:
            logger.info(f"Executing statement: {query[:100]}...")
            
            if self.db_type == "postgres":
                with self.get_connection() as conn:
                    stmt = text(query)
                    if params:
                        conn.execute(stmt, params)   # type: ignore[arg-type]
                    else:
                        conn.execute(stmt)           # type: ignore[arg-type]
                    if self._should_commit(query):
                        conn.commit()
            else:
                if params:
                    self.duckdb_conn.execute(query, list(params.values()))
                else:
                    self.duckdb_conn.execute(query)
            
            logger.info("Statement executed successfully")
        except Exception as e:
            logger.error("Statement execution failed", exc_info=True)
            raise
    
    def close(self):
        """Close database connection"""
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL engine disposed")
        
        if self._duckdb_conn:
            self._duckdb_conn.close()
            logger.info("DuckDB connection closed")
