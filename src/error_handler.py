# src/error_handler.py
import logging
import os
import re
import socket
from datetime import datetime
import requests

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logging() -> logging.Logger:
    """
    Configure file and console logging for the application and src modules.
    Safe to call repeatedly across application requests without duplicating handlers.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for logger_name in ("knowledge_rag", "src"):
        log = logging.getLogger(logger_name)
        log.setLevel(logging.INFO)
        if not log.handlers:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.WARNING)
            stream_handler.setFormatter(formatter)

            log.addHandler(file_handler)
            log.addHandler(stream_handler)
            log.propagate = False

    return logging.getLogger("knowledge_rag")


def is_timeout_error(exc: Exception) -> bool:
    """Return True if the exception or any exception in its chain is a timeout error."""
    visited = set()
    current = exc

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        if isinstance(
            current,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                TimeoutError,
                socket.timeout,
            ),
        ):
            return True

        if "timeout" in type(current).__name__.lower():
            return True

        msg = str(current).lower()
        if any(
            marker in msg
            for marker in ("read timed out", "connect timed out", "request timed out", "timed out")
        ):
            return True

        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return False


def is_overloaded_error(exc: Exception) -> bool:
    """Return True if the exception or any in its chain is a 503 Overloaded or 429 Rate Limit error."""
    visited = set()
    current = exc

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current).lower()
        if any(
            marker in msg
            for marker in (
                "503",
                "service is overloaded",
                "server is overloaded",
                "overloaded",
                "429",
                "too many requests",
                "rate limit",
                "504 gateway",
                "502 bad gateway",
            )
        ):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return False


def _extract_timeout_meta(exc: Exception) -> tuple[str | None, str | None]:
    """Extract host and timeout duration (seconds) across the exception chain if present."""
    visited = set()
    current = exc
    host = None
    timeout_sec = None

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        text = str(current)
        if host is None:
            host_match = re.search(r"host=['\"]([^'\"]+)['\"]", text)
            if host_match:
                host = host_match.group(1)
        if timeout_sec is None:
            timeout_match = re.search(r"timeout\s*=\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if timeout_match:
                timeout_sec = f"{timeout_match.group(1)}s"
        if host and timeout_sec:
            break
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return host, timeout_sec


def log_api_error(stage: str, exc: Exception, query: str | None = None) -> dict:
    """
    Log an API/pipeline exception (including full traceback) to logs/app.log
    and return a structured dictionary for rendering a clean UI card.
    """
    logger = setup_logging()
    timeout = is_timeout_error(exc)
    overloaded = is_overloaded_error(exc)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host, timeout_sec = _extract_timeout_meta(exc)

    query_preview = f" | query={query[:120]!r}" if query else ""
    if timeout:
        logger.error(
            "REQUEST TIMEOUT during [%s]%s | host=%s | timeout=%s | error=%s: %s",
            stage,
            query_preview,
            host or "unknown",
            timeout_sec or "unknown",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    elif overloaded:
        logger.error(
            "SERVICE OVERLOADED (503/429) during [%s]%s | error=%s: %s",
            stage,
            query_preview,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    else:
        logger.error(
            "API ERROR during [%s]%s | error=%s: %s",
            stage,
            query_preview,
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    return {
        "is_timeout": timeout,
        "is_overloaded": overloaded,
        "stage": stage,
        "timestamp": timestamp,
        "host": host,
        "timeout_sec": timeout_sec,
        "error_type": type(exc).__name__,
        "details": f"{type(exc).__name__}: {exc}",
        "log_file": LOG_FILE,
    }
