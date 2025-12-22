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
from contextvars import ContextVar

from churn_compass.config import settings

# Context (pipeline-level metadata)
_run_id_ctx: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_stage_ctx: ContextVar[Optional[str]] = ContextVar("stage", default=None)


def set_run_context(run_id: Optional[str] = None, stage: Optional[str] = None) -> None:
    _run_id_ctx.set(run_id)
    if stage is not None:
        _stage_ctx.set(stage)


def clear_run_context() -> None:
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

_configured_loggers: set[str] = set()


# Formatters
class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def _safe_serialize(self, obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Attach implicit context
        run_id = _run_id_ctx.get()
        stage = _stage_ctx.get()

        if run_id:
            log_data["run_id"] = run_id
        if stage:
            log_data["stage"] = stage

        if self.include_extra:
            extra_keys = {
                k: v
                for k, v in record.__dict__.items()
                if k not in _STANDARD_LOGRECORD_ATTRS and not k.startswith("_")
            }

            if extra_keys:
                safe_extra = {k: self._safe_serialize(v) for k, v in extra_keys.items()}
                log_data["context"] = mask_pii(safe_extra)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# PII masking
def mask_pii(
    data: Dict[str, Any], fields_to_mask: Optional[list[str]] = None
) -> Dict[str, Any]:
    if fields_to_mask is None:
        fields_to_mask = [
            "customerid",
            "customer_id",
            "surname",
            "last_name",
            "email",
            "phone",
            "ssn",
            "password",
        ]

    masked = data.copy()
    lower_keys = {k.lower(): k for k in masked.keys()}

    for field in fields_to_mask:
        if field.lower() in lower_keys:
            key = lower_keys[field.lower()]
            value = str(masked[key])
            masked[key] = f"***{value[-4:]}" if len(value) > 4 else "***"

    return masked


# Logger setup
def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)

    log_level_str = level or settings.log_level
    log_level_int = getattr(logging, log_level_str.upper())
    logger.setLevel(log_level_int)

    if name not in _configured_loggers:
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level_int)
            console_handler.setFormatter(
                JSONFormatter() if settings.log_format == "json" else TextFormatter()
            )
            logger.addHandler(console_handler)

        if log_to_file and settings.log_file:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONFormatter(include_extra=True))
            logger.addHandler(file_handler)

        logger.propagate = False
        _configured_loggers.add(name)

    return logger


# Execution time decorator
def log_execution_time(logger: logging.Logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    "Function completed",
                    extra={
                        "execution_time_seconds": round(duration, 3),
                        "status": "success",
                    },
                )
                return result
            except Exception:
                duration = time.time() - start_time
                logger.error(
                    "Function failed",
                    extra={
                        "execution_time_seconds": round(duration, 3),
                        "status": "error",
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Default package logger
default_logger = setup_logger("churn_compass")
