"""
Logging Utility for SecureScribe Backend using Loguru

This module provides simple, powerful logging using loguru with Loki integration.
Features include:

- Beautiful colorful console output by default
- Automatic log rotation and retention
- Minimal configuration needed
- FastAPI middleware integration
- Exception tracking and formatting with full traceback capture
- Loki HTTP streaming with complete exception information
- Global exception handler for uncaught exceptions

Usage:
    from app.utils.logging import logger, setup_logging

    # Setup logging (call once in main.py)
    setup_logging()

    # Use logger directly
    logger.info("Application started")
    logger.warning("Something might be wrong")
    logger.error("An error occurred")
    logger.debug("Detailed debug information")
    logger.success("Operation completed successfully")

    # For exceptions, use logger.exception() to include traceback
    try:
        risky_operation()
    except Exception:
        logger.exception("Something went wrong")
"""

import os
import sys
import threading
import time
import traceback

from loguru import logger as loguru_logger
from loki_logger_handler.formatters.loguru_formatter import LoguruFormatter
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler

# Export logger for easy import
logger = loguru_logger


def _global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to catch uncaught exceptions and log them with full traceback to Loki.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Format the full traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = "".join(tb_lines)

    # Log with full traceback - this will be sent to both console and Loki
    logger.critical(f"Uncaught exception in thread {threading.current_thread().name}: {exc_value}", extra={"traceback": tb_text})

    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration using loguru with console and Loki HTTP streaming.

    Features:
    - Console logging with colors and formatting
    - Loki HTTP streaming with full traceback capture
    - Global exception handler for uncaught exceptions
    - Async logging to prevent blocking
    - Backtrace and diagnostic information included

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    Environment Variables:
        LOKI_URL: Loki HTTP API endpoint (e.g., https://loki.wc504.io.vn/loki/api/v1/push)
                 If not set, only console logging is enabled.
        PYTHON_ENVIRONMENT: Environment name for Loki labels (default: "development")

    Example:
        setup_logging("DEBUG")  # Enable debug logging with full tracebacks
    """
    # Remove default handler
    loguru_logger.remove()

    # Add console handler with colors
    loguru_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add Loki HTTP handler if LOKI_URL is configured
    loki_url = "https://loki.wc504.io.vn/loki/api/v1/push"
    if loki_url:
        try:
            loki_handler = LokiLoggerHandler(
                url=loki_url,
                labels={"job": f"chang-ai-{os.getenv('PYTHON_ENVIRONMENT', 'development')}", "environment": os.getenv("PYTHON_ENVIRONMENT", "development")},
                timeout=10,
                label_keys={},
                default_formatter=LoguruFormatter(),
            )
            # Configure Loki handler to capture full exception information
            loguru_logger.add(
                loki_handler,
                serialize=True,
                level=level,
                backtrace=True,  # Include backtrace in serialized logs
                diagnose=True,  # Include diagnostic information
                enqueue=True,  # Async logging to avoid blocking
            )
            loguru_logger.info(f"Loki HTTP logging configured: {loki_url}")
        except Exception as e:
            loguru_logger.warning(f"Failed to configure Loki logging: {e}")
    else:
        loguru_logger.debug("LOKI_URL not set, Loki HTTP logging disabled")

    # Install global exception handler to catch uncaught exceptions
    sys.excepthook = _global_exception_handler

    # Log successful setup
    loguru_logger.info("Logging setup completed with traceback capture enabled")


# FastAPI middleware for request/response logging
class FastAPILoggingMiddleware:
    """
    FastAPI middleware that logs HTTP requests and responses with timing.

    Logs request method, path, response status, and duration.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request info
        method = scope["method"]
        path = scope["path"]
        query = scope.get("query_string", b"").decode("utf-8")
        if query:
            path = f"{path}?{query}"

        logger.info(f"→ {method} {path}")

        # Track timing
        start_time = time.time()
        original_send = send
        response_status = None
        response_length = 0

        async def logging_send(message):
            nonlocal response_status, response_length

            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body":
                response_length += len(message.get("body", b""))

            await original_send(message)

        try:
            await self.app(scope, receive, logging_send)
            duration = time.time() - start_time

            if response_status and response_status < 400:
                logger.success(f"← {method} {path} | {response_status} | {duration:.3f}s | {response_length} bytes")
            else:
                logger.warning(f"← {method} {path} | {response_status} | {duration:.3f}s | {response_length} bytes")
        except Exception:
            duration = time.time() - start_time
            logger.exception(f"Error processing request {method} {path} | ERROR | {duration:.3f}s")
            raise
