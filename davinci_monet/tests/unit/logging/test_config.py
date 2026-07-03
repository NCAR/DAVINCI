"""Tests for logging configuration."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from davinci_monet.logging import LogLevel
from davinci_monet.logging import config as logging_config
from davinci_monet.logging import configure_logging, get_logger, set_log_level
from davinci_monet.logging.config import (
    LOGGER_PREFIX,
    ColorFormatter,
    StructuredFormatter,
    _supports_color,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Verify log level values match Python logging."""
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR
        assert LogLevel.CRITICAL.value == logging.CRITICAL

    def test_from_string_valid(self) -> None:
        """Test converting valid strings to LogLevel."""
        assert LogLevel.from_string("debug") == LogLevel.DEBUG
        assert LogLevel.from_string("INFO") == LogLevel.INFO
        assert LogLevel.from_string("Warning") == LogLevel.WARNING

    def test_from_string_invalid(self) -> None:
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            LogLevel.from_string("invalid")


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Test getting a logger with a module name."""
        logger = get_logger("mymodule")
        assert logger.name == f"{LOGGER_PREFIX}.mymodule"

    def test_get_logger_without_name(self) -> None:
        """Test getting the root package logger."""
        logger = get_logger()
        assert logger.name == LOGGER_PREFIX

    def test_get_logger_with_prefixed_name(self) -> None:
        """Test that already-prefixed names are not double-prefixed."""
        logger = get_logger(f"{LOGGER_PREFIX}.submodule")
        assert logger.name == f"{LOGGER_PREFIX}.submodule"

    def test_get_logger_returns_same_instance(self) -> None:
        """Test that getting the same logger twice returns same instance."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        assert logger1 is logger2


class TestSetLogLevel:
    """Tests for set_log_level function."""

    def test_set_level_with_string(self) -> None:
        """Test setting level with string."""
        set_log_level("DEBUG")
        root = logging.getLogger(LOGGER_PREFIX)
        assert root.level == logging.DEBUG

    def test_set_level_with_enum(self) -> None:
        """Test setting level with LogLevel enum."""
        set_log_level(LogLevel.WARNING)
        root = logging.getLogger(LOGGER_PREFIX)
        assert root.level == logging.WARNING

    def test_set_level_with_int(self) -> None:
        """Test setting level with integer."""
        set_log_level(logging.ERROR)
        root = logging.getLogger(LOGGER_PREFIX)
        assert root.level == logging.ERROR


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def teardown_method(self) -> None:
        """Clean up loggers after each test."""
        root = logging.getLogger(LOGGER_PREFIX)
        for handler in root.handlers:
            handler.close()
        root.handlers.clear()
        root.setLevel(logging.WARNING)  # Reset to default

    def test_basic_configuration(self) -> None:
        """Test basic logging configuration."""
        logger = configure_logging()
        assert logger.name == LOGGER_PREFIX
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_configure_with_level_string(self) -> None:
        """Test configuring with string level."""
        logger = configure_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_configure_with_level_enum(self) -> None:
        """Test configuring with enum level."""
        logger = configure_logging(level=LogLevel.ERROR)
        assert logger.level == logging.ERROR

    def test_configure_with_file(self) -> None:
        """Test configuring with file output."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = Path(f.name)

        try:
            logger = configure_logging(log_file=log_path)
            assert len(logger.handlers) == 2

            # Log a message
            logger.info("Test message")

            # Verify message was written to file
            content = log_path.read_text()
            assert "Test message" in content
        finally:
            log_path.unlink()

    def test_color_console_does_not_mutate_record_seen_by_file_handler(
        self, tmp_path, monkeypatch
    ) -> None:
        """Console coloring must not duplicate formatted text/extras in file logs."""
        monkeypatch.setattr(logging_config, "_supports_color", lambda: True)
        log_path = tmp_path / "davinci.log"

        logger = configure_logging(log_file=log_path, use_color=True, use_structured=True)
        logger.info("Processing", extra={"dataset": "CMAQ"})

        content = log_path.read_text()
        assert content.count("Processing") == 1
        assert content.count("dataset='CMAQ'") == 1
        assert "asctime=" not in content

    def test_configure_without_color(self) -> None:
        """Test configuring without color."""
        logger = configure_logging(use_color=False)
        handler = logger.handlers[0]
        # Formatter should not be ColorFormatter
        assert not isinstance(handler.formatter, ColorFormatter)

    def test_configure_without_structured(self) -> None:
        """Test configuring without structured formatting."""
        logger = configure_logging(use_structured=False, use_color=False)
        handler = logger.handlers[0]
        # Should be a plain Formatter
        assert type(handler.formatter) is logging.Formatter

    def test_configure_propagate(self) -> None:
        """Test propagate setting."""
        logger = configure_logging(propagate=True)
        assert logger.propagate is True

        logger = configure_logging(propagate=False)
        assert logger.propagate is False

    def test_configure_clears_existing_handlers(self) -> None:
        """Test that reconfiguring clears existing handlers."""
        configure_logging()
        logger = logging.getLogger(LOGGER_PREFIX)
        initial_count = len(logger.handlers)

        configure_logging()
        assert len(logger.handlers) == initial_count  # Not doubled


class TestColorFormatter:
    """Tests for ColorFormatter."""

    def test_format_adds_color_codes(self) -> None:
        """Test that format adds ANSI color codes."""
        formatter = ColorFormatter("%(levelname)s: %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        # Should contain green color code for INFO
        assert "\033[32m" in result
        assert "\033[0m" in result  # Reset code
        assert "Test message" in result

    def test_format_different_levels(self) -> None:
        """Test different color codes for different levels."""
        formatter = ColorFormatter("%(message)s")

        levels = [
            (logging.DEBUG, "\033[36m"),  # Cyan
            (logging.INFO, "\033[32m"),  # Green
            (logging.WARNING, "\033[33m"),  # Yellow
            (logging.ERROR, "\033[31m"),  # Red
            (logging.CRITICAL, "\033[1;31m"),  # Bold Red
        ]

        for level, expected_code in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=0,
                msg="msg",
                args=(),
                exc_info=None,
            )
            result = formatter.format(record)
            assert expected_code in result, f"Missing color code for level {level}"


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_format_includes_extra_fields(self) -> None:
        """Test that extra fields are included in output."""
        formatter = StructuredFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.dataset = "CMAQ"
        record.geometry = "AQS"

        result = formatter.format(record)
        assert "Test message" in result
        assert "dataset='CMAQ'" in result
        assert "geometry='AQS'" in result

    def test_format_without_extra_fields(self) -> None:
        """Test formatting without extra fields."""
        formatter = StructuredFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert result == "Test message"
        assert "[" not in result  # No extra fields bracket


class TestLoggingWorkflow:
    """Logging workflow tests (calls logging APIs directly)."""

    def teardown_method(self) -> None:
        """Clean up loggers after each test."""
        root = logging.getLogger(LOGGER_PREFIX)
        root.handlers.clear()

    def test_log_with_extra_fields(self) -> None:
        """Test logging with extra context fields."""
        configure_logging(level="DEBUG", use_color=False)
        logger = get_logger("integration_test")

        # Create a handler to capture output
        captured: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        capture_handler = CaptureHandler()
        capture_handler.setFormatter(StructuredFormatter("%(message)s"))
        logger.addHandler(capture_handler)

        logger.info("Processing", extra={"dataset": "CMAQ", "variable": "O3"})

        assert len(captured) == 1
        assert "Processing" in captured[0]
        assert "dataset='CMAQ'" in captured[0]
        assert "variable='O3'" in captured[0]

    def test_child_logger_inherits_config(self) -> None:
        """Test that child loggers inherit configuration."""
        root = configure_logging(level="DEBUG")
        child = get_logger("child.module")

        # Child should use parent's handlers
        assert child.parent is not None
        assert child.level == 0  # NOTSET - inherits from parent


class TestSupportsColor:
    """Tests for _supports_color function."""

    def test_supports_color_returns_bool(self) -> None:
        """Verify _supports_color returns a boolean."""
        result = _supports_color()
        assert isinstance(result, bool)
