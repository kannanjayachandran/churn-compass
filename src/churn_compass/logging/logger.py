"""
Churn Compass - Structured Logging System

Production-grade logging with JSON formatting for easy parsing by
log aggregation systems (ELK, Splunk, CloudWatch, Datadog).

Features:
- Structured JSON logs with context fields
- PII masking capabilities
- Rotating file handlers
- Console and file output
- Performance tracking decorators
"""

import logging
import logging.handlers
import sys
import json
import time
import functools
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from churn_compass.config.settings import settings

# Track configured loggers
_configured_loggers = set()


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    Output logs in JSON format with timestamp, level, message, and context.
    """

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add Exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields (context) if enabled
        if self.include_extra:
            extra_fields = {
                k: v
                for k, v in record.__dict__.items()
                if k
                not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "taskName",
                ]
            }
            if extra_fields:
                log_data["context"] = extra_fields
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Standard text formatter for human-readable console output"""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def mask_pii(
    data: Dict[str, Any], fields_to_mask: Optional[list] = None
) -> Dict[str, Any]:
    """
    Mask PII fields in dictionaries before logging.

    Args:
        data: Dictionary potentially containing PII
        fields_to_mask: List of field names to mask (default: common PII fields)

    Returns:
        Dictionary with masked fields
    """
    if fields_to_mask is None:
        fields_to_mask = [
            "CustomerId",
            "customerid",
            "customer_id",
            "Surname",
            "surname",
            "last_name",
            "email",
            "phone",
            "ssn",
            "password",
        ]
    masked_data = data.copy()
    for field in fields_to_mask:
        if field in masked_data:
            value = str(masked_data[field])
            if len(value) > 4:
                masked_data[field] = f"***{value[-4:]}"
            else:
                masked_data[field] = "***"
    return masked_data


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with JSON and/or text formatting

    :param name: Logger name (usually __name__ of calling module)
    :type name: str
    :param level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :type level: Optional[str]
    :param log_to_file: Whether to log to file
    :type log_to_file: bool
    :param log_to_console: Whether to log to console
    :type log_to_console: bool
    :return: Returns a logging.Logger object
    """
    logger = logging.getLogger(name)

    # if handler already added, return logger immediately
    if name in _configured_loggers:
        return logger

    # set level from parameter or settings
    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper()))

    # console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        console_handler.setFormatter(
            JSONFormatter() if settings.log_format == "json" else TextFormatter()
        )
        logger.addHandler(console_handler)

    # File handler
    if log_to_file and settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    # Prevent propagation to root logger
    logger.propagate = False

    # Mark as configured
    _configured_loggers.add(name)

    return logger


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time

    Usage:
        @log_execution_time(logger)
        def my_function():
            pass
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"Function {func.__name__} completed",
                    extra={
                        "function": func.__name__,
                        "execution_time_seconds": round(execution_time, 3),
                        "status": "success",
                    },
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        "function": func.__name__,
                        "execution_time_seconds": round(execution_time, 3),
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Create default logger for the package
default_logger = setup_logger("churn_compass")
