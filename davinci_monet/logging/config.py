"""Logging configuration for DAVINCI.

This module provides structured logging configuration to replace
print() statements throughout the package. It supports:
- Console output with color formatting
- File output with rotation
- Structured extra fields for context
- Multiple log levels

The logging follows Python's standard logging module patterns while
providing convenience functions for easy setup.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

# Package-wide logger name prefix
LOGGER_PREFIX = "davinci_monet"


class LogLevel(Enum):
    """Logging levels supported by DAVINCI."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Convert string to LogLevel.

        Parameters
        ----------
        level
            Level name (case-insensitive).

        Returns
        -------
        LogLevel
            Corresponding log level.

        Raises
        ------
        ValueError
            If level name is not recognized.
        """
        try:
            return cls[level.upper()]
        except KeyError:
            valid = ", ".join(m.name for m in cls)
            raise ValueError(f"Invalid log level '{level}'. Valid levels: {valid}") from None


class ColorFormatter(logging.Formatter):
    """Formatter that adds color codes for console output.

    Colors are applied based on log level:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red
    """

    # ANSI color codes
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color codes."""
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        if color:
            return f"{color}{message}{self.RESET}"
        return message


class StructuredFormatter(logging.Formatter):
    """Formatter that includes structured extra fields.

    Extra fields passed via the 'extra' parameter are appended
    to the log message in key=value format.
    """

    # Fields that are part of the standard LogRecord
    STANDARD_FIELDS = {
        "name",
        "msg",
        "args",
        "asctime",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "exc_info",
        "exc_text",
        "thread",
        "threadName",
        "taskName",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with extra fields."""
        message = super().format(record)

        # Extract extra fields (non-standard attributes)
        extra_fields = {k: v for k, v in record.__dict__.items() if k not in self.STANDARD_FIELDS}

        if extra_fields:
            extras = " ".join(f"{k}={v!r}" for k, v in sorted(extra_fields.items()))
            message = f"{message} [{extras}]"

        return message


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the given module name.

    The logger name is prefixed with the package name to ensure
    all DAVINCI loggers are grouped together.

    Parameters
    ----------
    name
        Module name (typically __name__). If None, returns the
        root package logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    >>> logger.warning("Missing data", extra={"variable": "O3"})
    """
    if name is None:
        return logging.getLogger(LOGGER_PREFIX)

    # If name already starts with prefix, use as-is
    if name.startswith(LOGGER_PREFIX):
        return logging.getLogger(name)

    # Otherwise, prefix it
    return logging.getLogger(f"{LOGGER_PREFIX}.{name}")


def set_log_level(level: str | LogLevel | int) -> None:
    """Set the log level for all DAVINCI loggers.

    Parameters
    ----------
    level
        Log level as string ('DEBUG', 'INFO', etc.), LogLevel enum,
        or integer value.

    Examples
    --------
    >>> set_log_level("DEBUG")
    >>> set_log_level(LogLevel.WARNING)
    >>> set_log_level(logging.ERROR)
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level).value
    elif isinstance(level, LogLevel):
        level = level.value

    root_logger = logging.getLogger(LOGGER_PREFIX)
    root_logger.setLevel(level)


def configure_logging(
    level: str | LogLevel | int = LogLevel.INFO,
    log_file: str | Path | None = None,
    log_format: str | None = None,
    date_format: str | None = None,
    use_color: bool = True,
    use_structured: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Configure logging for DAVINCI.

    This function sets up logging handlers for console and optionally
    file output. It should be called once at application startup.

    Parameters
    ----------
    level
        Minimum log level to capture. Can be string ('DEBUG', 'INFO', etc.),
        LogLevel enum, or integer.
    log_file
        Optional path to log file. If provided, logs are written to this
        file in addition to console.
    log_format
        Custom log format string. If None, uses a sensible default.
    date_format
        Custom date format string. If None, uses ISO format.
    use_color
        Whether to colorize console output. Only applies to terminals.
    use_structured
        Whether to include extra fields in log output.
    propagate
        Whether to propagate logs to parent loggers.

    Returns
    -------
    logging.Logger
        The configured root package logger.

    Examples
    --------
    Basic configuration:

    >>> configure_logging()

    With file output and debug level:

    >>> configure_logging(level="DEBUG", log_file="davinci.log")

    Custom format:

    >>> configure_logging(
    ...     log_format="%(levelname)s - %(message)s",
    ...     use_color=False
    ... )
    """
    # Convert level to integer
    if isinstance(level, str):
        level_int = LogLevel.from_string(level).value
    elif isinstance(level, LogLevel):
        level_int = level.value
    else:
        level_int = level

    # Default formats
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Get the root package logger
    root_logger = logging.getLogger(LOGGER_PREFIX)
    root_logger.setLevel(level_int)
    root_logger.propagate = propagate

    # Remove existing handlers to avoid duplicates and close file descriptors.
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level_int)

    # Choose formatter based on options
    if use_color and _supports_color():
        if use_structured:

            class ColorStructuredFormatter(ColorFormatter, StructuredFormatter):
                """Combined color and structured formatter."""

                def format(self, record: logging.LogRecord) -> str:
                    # First apply structured formatting
                    message = StructuredFormatter.format(self, record)
                    # Then apply color
                    color = self.COLORS.get(record.levelno, "")
                    if color:
                        return f"{color}{message}{self.RESET}"
                    return message

            console_formatter: logging.Formatter = ColorStructuredFormatter(log_format, date_format)
        else:
            console_formatter = ColorFormatter(log_format, date_format)
    elif use_structured:
        console_formatter = StructuredFormatter(log_format, date_format)
    else:
        console_formatter = logging.Formatter(log_format, date_format)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level_int)

        # File output never uses colors
        if use_structured:
            file_formatter: logging.Formatter = StructuredFormatter(log_format, date_format)
        else:
            file_formatter = logging.Formatter(log_format, date_format)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def _supports_color() -> bool:
    """Check if the terminal supports color output.

    Returns
    -------
    bool
        True if color is supported, False otherwise.
    """
    # Check if we're writing to a real terminal
    if not hasattr(sys.stderr, "isatty"):
        return False
    if not sys.stderr.isatty():
        return False

    # Check for known non-color environments
    import os

    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False

    return True
