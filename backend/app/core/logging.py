"""Structured logging configuration with secret redaction and request context."""

import logging
import re
import sys
from contextvars import ContextVar
from typing import Optional

# Request ID context variable shared across coroutines
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# Regex to match Google API keys and common token formats
API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z-_]{35}")
BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)


class SecretMaskingFilter(logging.Filter):
    """Logging filter that scrubs API keys and sensitive bearer tokens from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = API_KEY_PATTERN.sub("[REDACTED_API_KEY]", record.msg)
            record.msg = BEARER_PATTERN.sub("Bearer [REDACTED_TOKEN]", record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    API_KEY_PATTERN.sub("[REDACTED_API_KEY]", str(arg))
                    if isinstance(arg, str)
                    else arg
                    for arg in record.args
                )
        return True


class RequestContextFilter(logging.Filter):
    """Logging filter that injects the current request ID into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        return True


def setup_logging(log_level: str = "INFO") -> None:
    """Initialize root logger with formatting, filters, and standard stream handler."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on re-initialization
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] [req:%(request_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestContextFilter())
    console_handler.addFilter(SecretMaskingFilter())

    root_logger.addHandler(console_handler)

    # Mute noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve named logger."""
    return logging.getLogger(name or "crispe_backend")
