"""
Unit tests for enhanced logging configuration.

This module tests the enhanced logging functionality including
structured logging, error context, and logging utilities.
"""

import pytest
import json
import logging
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from io import StringIO

from src.utils.logging_config import (
    StructuredFormatter,
    ErrorContextFilter,
    setup_logging,
    get_logger_with_context,
    log_error_with_context,
    setup_development_logging,
    setup_production_logging
)
from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    WebScrapingError,
    DataValidationError
)


class TestStructuredFormatter:
    """Test cases for StructuredFormatter class."""
    
    def test_structured_formatter_initialization(self):
        """Test structured formatter initialization."""
        formatter = StructuredFormatter()
        assert formatter.include_stack_trace is False
        
        formatter_with_stack = StructuredFormatter(include_stack_trace=True)
        assert formatter_with_stack.include_stack_trace is True
    
    def test_format_basic_log_record(self):
        """Test formatting a basic log record."""
        formatter = StructuredFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "path"
        assert log_data["function"] == "<module>"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
    
    def test_format_with_exception(self):
        """Test formatting log record with exception."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert "stack_trace" not in log_data["exception"]  # Not included by default
    
    def test_format_with_exception_and_stack_trace(self):
        """Test formatting log record with exception and stack trace."""
        formatter = StructuredFormatter(include_stack_trace=True)
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert "stack_trace" in log_data["exception"]
        assert isinstance(log_data["exception"]["stack_trace"], list)
    
    def test_format_with_custom_error(self):
        """Test formatting log record with custom SpotAnalyzerBaseError."""
        formatter = StructuredFormatter()
        
        custom_error = WebScrapingError(
            message="Scraping failed",
            url="https://example.com",
            status_code=404
        )
        
        try:
            raise custom_error
        except WebScrapingError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Custom error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["exception"]["type"] == "WebScrapingError"
        assert log_data["exception"]["error_code"] == "WEB_SCRAPING_ERROR"
        assert log_data["exception"]["details"]["url"] == "https://example.com"
        assert log_data["exception"]["details"]["status_code"] == 404
    
    def test_format_with_extra_fields(self):
        """Test formatting log record with extra fields."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.request_id = "req-123"
        record.operation = "test_operation"
        record.custom_field = "custom_value"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["request_id"] == "req-123"
        assert log_data["operation"] == "test_operation"
        assert log_data["extra"]["custom_field"] == "custom_value"
    
    def test_format_serialization_fallback(self):
        """Test fallback when JSON serialization fails."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add a non-serializable object
        class NonSerializable:
            def __str__(self):
                raise Exception("Cannot serialize")
        
        record.bad_field = NonSerializable()
        
        # Should not raise exception, should return fallback format
        formatted = formatter.format(record)
        
        # Should be a simple string format, not JSON
        assert "test.logger" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted


class TestErrorContextFilter:
    """Test cases for ErrorContextFilter class."""
    
    def test_error_context_filter_initialization(self):
        """Test error context filter initialization."""
        filter_obj = ErrorContextFilter()
        assert filter_obj.service_name == "spot-price-analyzer"
        
        custom_filter = ErrorContextFilter("custom-service")
        assert custom_filter.service_name == "custom-service"
    
    def test_filter_adds_context(self):
        """Test that filter adds context to log records."""
        filter_obj = ErrorContextFilter("test-service")
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True  # Record should be allowed
        assert record.service == "test-service"
        assert hasattr(record, 'request_id')
        assert record.request_id is None  # Default value


class TestLoggingSetup:
    """Test cases for logging setup functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def tearDown(self):
        """Clean up test environment."""
        # Clear handlers after test
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        self.setUp()
        
        setup_logging(log_level="INFO")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) >= 1
        
        # Test that we can log
        logger = logging.getLogger("test")
        logger.info("Test message")  # Should not raise exception
        
        self.tearDown()
    
    def test_setup_logging_structured(self):
        """Test structured logging setup."""
        self.setUp()
        
        setup_logging(log_level="DEBUG", structured=True)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        # Check that handler uses StructuredFormatter
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)
        
        self.tearDown()
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        self.setUp()
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            setup_logging(log_level="INFO", log_file=log_file)
            
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) >= 2  # Console + file
            
            # Test logging to file
            logger = logging.getLogger("test")
            logger.info("Test file message")
            
            # Check that file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test file message" in content
        
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
            self.tearDown()
    
    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid log level."""
        with pytest.raises(ValueError):
            setup_logging(log_level="INVALID")
    
    def test_setup_development_logging(self):
        """Test development logging setup."""
        self.setUp()
        
        setup_development_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        self.tearDown()
    
    @patch('src.utils.logging_config.setup_logging')
    def test_setup_production_logging(self, mock_setup):
        """Test production logging setup."""
        setup_production_logging("/var/log/test.log")
        
        mock_setup.assert_called_once_with(
            log_level='INFO',
            structured=True,
            include_stack_trace=False,
            log_file="/var/log/test.log",
            service_name="spot-price-analyzer"
        )


class TestLoggerWithContext:
    """Test cases for context-aware logger functionality."""
    
    def test_get_logger_with_context_basic(self):
        """Test getting logger with basic context."""
        logger = get_logger_with_context("test.logger")
        
        assert logger.logger.name == "test.logger"
        assert hasattr(logger, 'extra')
    
    def test_get_logger_with_context_full(self):
        """Test getting logger with full context."""
        logger = get_logger_with_context(
            name="test.logger",
            request_id="req-123",
            operation="test_operation",
            custom_field="custom_value"
        )
        
        assert logger.logger.name == "test.logger"
        assert logger.extra["request_id"] == "req-123"
        assert logger.extra["operation"] == "test_operation"
        assert logger.extra["custom_field"] == "custom_value"
    
    def test_logger_context_in_log_calls(self):
        """Test that context is included in log calls."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        test_logger = logging.getLogger("test.context")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        
        context_logger = get_logger_with_context(
            name="test.context",
            request_id="req-456",
            operation="context_test"
        )
        
        context_logger.info("Test context message")
        
        # Get the logged output
        output = stream.getvalue()
        
        # Should be JSON with context
        log_data = json.loads(output.strip())
        assert log_data["request_id"] == "req-456"
        assert log_data["operation"] == "context_test"
        assert log_data["message"] == "Test context message"


class TestLogErrorWithContext:
    """Test cases for log_error_with_context function."""
    
    def test_log_error_with_context_basic(self):
        """Test logging error with basic context."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        test_logger = logging.getLogger("test.error")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.ERROR)
        
        error = ValueError("Test error")
        
        log_error_with_context(
            logger=test_logger,
            error=error,
            message="Error occurred",
            request_id="req-789"
        )
        
        # Get the logged output
        output = stream.getvalue()
        
        # Should be JSON with error context
        log_data = json.loads(output.strip())
        assert log_data["message"] == "Error occurred"
        assert log_data["request_id"] == "req-789"
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"
    
    def test_log_error_with_context_custom_error(self):
        """Test logging custom error with context."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        test_logger = logging.getLogger("test.custom.error")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.ERROR)
        
        error = DataValidationError(
            message="Invalid data",
            field_name="test_field",
            field_value="invalid_value"
        )
        
        log_error_with_context(
            logger=test_logger,
            error=error,
            message="Validation failed",
            operation="data_validation",
            custom_context="test_value"
        )
        
        # Get the logged output
        output = stream.getvalue()
        
        # Should be JSON with custom error context
        log_data = json.loads(output.strip())
        assert log_data["message"] == "Validation failed"
        assert log_data["operation"] == "data_validation"
        assert log_data["custom_context"] == "test_value"
        assert log_data["error_code"] == "DATA_VALIDATION_ERROR"
        assert log_data["error_details"]["field_name"] == "test_field"
        assert log_data["exception"]["error_code"] == "DATA_VALIDATION_ERROR"


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_end_to_end_structured_logging(self):
        """Test complete structured logging workflow."""
        # Set up structured logging
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(include_stack_trace=True))
        handler.addFilter(ErrorContextFilter("integration-test"))
        
        logger = logging.getLogger("integration.test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log various types of messages
        logger.info("Basic info message")
        
        context_logger = get_logger_with_context(
            name="integration.test",
            request_id="req-integration",
            operation="integration_test"
        )
        context_logger.warning("Warning with context")
        
        # Log error with exception
        try:
            raise WebScrapingError("Integration test error", url="https://test.com")
        except WebScrapingError as e:
            log_error_with_context(
                logger=logger,
                error=e,
                message="Integration test error occurred",
                request_id="req-integration",
                test_field="test_value"
            )
        
        # Parse all log entries
        output_lines = stream.getvalue().strip().split('\n')
        log_entries = [json.loads(line) for line in output_lines if line]
        
        # Verify basic info message
        info_entry = log_entries[0]
        assert info_entry["level"] == "INFO"
        assert info_entry["message"] == "Basic info message"
        assert info_entry["service"] == "integration-test"
        
        # Verify warning with context
        warning_entry = log_entries[1]
        assert warning_entry["level"] == "WARNING"
        assert warning_entry["message"] == "Warning with context"
        assert warning_entry["request_id"] == "req-integration"
        assert warning_entry["operation"] == "integration_test"
        
        # Verify error with exception
        error_entry = log_entries[2]
        assert error_entry["level"] == "ERROR"
        assert error_entry["message"] == "Integration test error occurred"
        assert error_entry["request_id"] == "req-integration"
        assert error_entry["test_field"] == "test_value"
        assert error_entry["exception"]["type"] == "WebScrapingError"
        assert error_entry["exception"]["error_code"] == "WEB_SCRAPING_ERROR"
        assert error_entry["exception"]["details"]["url"] == "https://test.com"
        assert "stack_trace" in error_entry["exception"]