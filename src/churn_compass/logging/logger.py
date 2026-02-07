"""
Churn Compass - Structured Logging System

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
from typing import Any, Dict, Optional, Set
from contextvars import ContextVar

from churn_compass import settings

# Context (pipeline-level metadata)
_run_id_ctx: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_stage_ctx: ContextVar[Optional[str]] = ContextVar("stage", default=None)


def set_run_context(run_id: Optional[str] = None, stage: Optional[str] = None) -> None:
    """Set the current execution context."""
    if run_id is not None:
        _run_id_ctx.set(run_id)
    if stage is not None:
        _stage_ctx.set(stage)


def clear_run_context() -> None:
    """Clear the current execution context."""
    _run_id_ctx.set(None)
    _stage_ctx.set(None)


# Internal constants
_STANDARD_LOGRECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}

_DEFAULT_PII_FIELDS = {
    "customerid",
    "customer_id",
    "surname",
    "last_name",
    "email",
    "phone",
    "ssn",
    "password",
    "api_key",
    "token",
}

_configured_loggers: set[str] = set()


class ContextFilter(logging.Filter):
    """Cleaner way to inject context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id_ctx.get()
        record.stage = _stage_ctx.get()
        return True


# PII masking
def mask_pii(
    data: Dict[str, Any], fields_to_mask: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Efficiently mask PII fields in a dictionary."""
    if not data:
        return data

    mask_set = fields_to_mask or _DEFAULT_PII_FIELDS
    masked = data.copy()

    for key, value in masked.items():
        if key.lower() in mask_set:
            val_str = str(value)
            masked[key] = f"***{val_str[-4:]}" if len(val_str) > 4 else "***"
        elif isinstance(value, dict):
            masked[key] = mask_pii(value, mask_set)

    return masked


# Formatters
class JSONFormatter(logging.Formatter):
    """
    Optimized JSON formatter for structured logging.
    Produces a flat structure compatible with log aggregation systems.
    """

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def _prepare_log_dict(self, record: logging.LogRecord) -> Dict[str, Any]:
        # Standard fields with @timestamp for ELK/Datadog compatibility
        log_data = {
            "@timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Add context from filter
        if getattr(record, "run_id", None):
            log_data["run_id"] = record.run_id
        if getattr(record, "stage", None):
            log_data["stage"] = record.stage

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            extra = {
                k: v
                for k, v in record.__dict__.items()
                if k not in _STANDARD_LOGRECORD_ATTRS
                and not k.startswith("_")
                and k not in ("run_id", "stage")
            }
            if extra:
                # Mask PII and merge into top-level for better ingestibility
                masked_extra = mask_pii(extra)
                for k, v in masked_extra.items():
                    log_data[k] = v

        return log_data

    def format(self, record: logging.LogRecord) -> str:
        log_dict = self._prepare_log_dict(record)
        try:
            return json.dumps(log_dict, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps(
                {"error": "Log record serialization failed", "msg": str(record.msg)}
            )


class TextFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)

    # Prevent re-configuring the same logger
    if name in _configured_loggers:
        return logger

    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Add context filter for all records
    logger.addFilter(ContextFilter())

    if log_to_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            JSONFormatter() if settings.log_format == "json" else TextFormatter()
        )
        logger.addHandler(handler)

    if log_to_file and settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        # File always gets JSON for ingestibility
        file_handler.setFormatter(JSONFormatter(include_extra=True))
        logger.addHandler(file_handler)

    logger.propagate = False
    _configured_loggers.add(name)
    return logger


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time and status."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.info(
                    f"Task {func.__name__} completed",
                    extra={
                        "duration_sec": round(duration, 3),
                        "status": "success",
                    },
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"Task {func.__name__} failed: {str(e)}",
                    extra={
                        "duration_sec": round(duration, 4),
                        "status": "error",
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Default package-wide logger
default_logger = setup_logger("churn_compass")
