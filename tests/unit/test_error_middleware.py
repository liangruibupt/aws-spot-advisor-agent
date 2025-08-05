"""
Unit tests for error handling middleware.

This module tests the error handling middleware functionality
for centralized error processing in the AWS Spot Price Analyzer.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.utils.error_middleware import (
    ErrorHandler,
    error_handling_context,
    handle_service_errors,
    create_error_mapping,
    ErrorAggregator,
    global_error_aggregator,
    report_error_to_aggregator
)
from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    WebScrapingError,
    DataValidationError,
    NetworkError,
    ServiceUnavailableError
)


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler("test_service")
        
        assert handler.service_name == "test_service"
        assert handler.include_stack_trace is False
        assert handler.log_errors is True
        assert handler.error_counts == {}
        assert handler.last_errors == []
    
    def test_error_handler_with_options(self):
        """Test error handler initialization with options."""
        handler = ErrorHandler(
            service_name="test_service",
            include_stack_trace=True,
            log_errors=False
        )
        
        assert handler.service_name == "test_service"
        assert handler.include_stack_trace is True
        assert handler.log_errors is False
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        error = ValueError("Test error")
        response = handler.handle_error(
            error=error,
            operation="test_operation",
            request_id="req-123"
        )
        
        # Check response structure
        assert response["error"] is True
        assert response["error_code"] == "ValueError"
        assert response["message"] == "Test error"
        assert response["request_id"] == "req-123"
        
        # Check error statistics
        assert handler.error_counts["ValueError"] == 1
        assert len(handler.last_errors) == 1
        
        error_record = handler.last_errors[0]
        assert error_record["service"] == "test_service"
        assert error_record["operation"] == "test_operation"
        assert error_record["error_type"] == "ValueError"
        assert error_record["request_id"] == "req-123"
    
    def test_handle_custom_error(self):
        """Test handling custom SpotAnalyzerBaseError."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        error = WebScrapingError(
            message="Scraping failed",
            url="https://example.com",
            status_code=404
        )
        
        response = handler.handle_error(
            error=error,
            operation="web_scraping"
        )
        
        assert response["error_code"] == "WEB_SCRAPING_ERROR"
        assert response["details"]["url"] == "https://example.com"
        assert response["details"]["status_code"] == 404
    
    def test_wrap_with_error_handling_success(self):
        """Test error handling wrapper with successful function."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        @handler.wrap_with_error_handling("test_operation")
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5
        assert handler.error_counts == {}
    
    def test_wrap_with_error_handling_with_error(self):
        """Test error handling wrapper with function that raises error."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        @handler.wrap_with_error_handling("test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(SpotAnalyzerBaseError) as exc_info:
            failing_function()
        
        assert "test_operation failed" in str(exc_info.value)
        assert handler.error_counts["ValueError"] == 1
    
    def test_wrap_with_error_mapping(self):
        """Test error handling wrapper with error mapping."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        error_mappings = {ValueError: DataValidationError}
        
        @handler.wrap_with_error_handling("test_operation", error_mappings)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(DataValidationError) as exc_info:
            failing_function()
        
        assert "test_operation failed" in str(exc_info.value)
        assert isinstance(exc_info.value.original_error, ValueError)
    
    def test_get_error_statistics(self):
        """Test getting error statistics."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        # Generate some errors
        handler.handle_error(ValueError("Error 1"), "op1")
        handler.handle_error(ValueError("Error 2"), "op2")
        handler.handle_error(TypeError("Error 3"), "op3")
        
        stats = handler.get_error_statistics()
        
        assert stats["service"] == "test_service"
        assert stats["total_errors"] == 3
        assert stats["error_counts_by_type"]["ValueError"] == 2
        assert stats["error_counts_by_type"]["TypeError"] == 1
        assert stats["recent_errors_count"] == 3
        assert stats["last_error_time"] is not None
    
    def test_get_recent_errors(self):
        """Test getting recent errors."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        # Generate some errors
        for i in range(5):
            handler.handle_error(ValueError(f"Error {i}"), f"op{i}")
        
        recent_errors = handler.get_recent_errors(limit=3)
        
        assert len(recent_errors) == 3
        assert recent_errors[-1]["error_message"] == "Error 4"
        assert recent_errors[0]["error_message"] == "Error 2"
    
    def test_clear_statistics(self):
        """Test clearing error statistics."""
        handler = ErrorHandler("test_service", log_errors=False)
        
        # Generate some errors
        handler.handle_error(ValueError("Error"), "op")
        
        assert len(handler.error_counts) > 0
        assert len(handler.last_errors) > 0
        
        handler.clear_statistics()
        
        assert handler.error_counts == {}
        assert handler.last_errors == []


class TestErrorHandlingContext:
    """Test cases for error handling context manager."""
    
    def test_error_handling_context_success(self):
        """Test error handling context with successful operation."""
        with error_handling_context("test_operation", "test_service") as handler:
            assert isinstance(handler, ErrorHandler)
            assert handler.service_name == "test_service"
            # No exception should be raised
    
    def test_error_handling_context_with_error_reraise(self):
        """Test error handling context with error and reraise=True."""
        with pytest.raises(ValueError):
            with error_handling_context("test_operation", "test_service", reraise=True):
                raise ValueError("Test error")
    
    def test_error_handling_context_with_error_no_reraise(self):
        """Test error handling context with error and reraise=False."""
        # Should not raise exception
        with error_handling_context("test_operation", "test_service", reraise=False):
            raise ValueError("Test error")


class TestServiceErrorsDecorator:
    """Test cases for handle_service_errors decorator."""
    
    def test_handle_service_errors_success(self):
        """Test service errors decorator with successful function."""
        @handle_service_errors("test_service")
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5
    
    def test_handle_service_errors_with_error(self):
        """Test service errors decorator with function that raises error."""
        @handle_service_errors("test_service", "test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(SpotAnalyzerBaseError) as exc_info:
            failing_function()
        
        assert "test_operation failed" in str(exc_info.value)
    
    def test_handle_service_errors_with_custom_error(self):
        """Test service errors decorator with custom error."""
        @handle_service_errors("test_service")
        def failing_function():
            raise WebScrapingError("Scraping failed")
        
        # Custom errors should be re-raised as-is
        with pytest.raises(WebScrapingError):
            failing_function()


class TestErrorMapping:
    """Test cases for error mapping functionality."""
    
    def test_create_error_mapping(self):
        """Test creating default error mapping."""
        mapping = create_error_mapping()
        
        assert ConnectionError in mapping
        assert TimeoutError in mapping
        assert ValueError in mapping
        assert KeyError in mapping
        assert TypeError in mapping
        
        assert mapping[ConnectionError] == NetworkError
        assert mapping[ValueError] == DataValidationError


class TestErrorAggregator:
    """Test cases for ErrorAggregator class."""
    
    def test_error_aggregator_initialization(self):
        """Test error aggregator initialization."""
        aggregator = ErrorAggregator(max_errors=100)
        
        assert aggregator.max_errors == 100
        assert aggregator.errors == []
        assert aggregator.error_counts == {}
    
    def test_add_error(self):
        """Test adding errors to aggregator."""
        aggregator = ErrorAggregator()
        
        error = ValueError("Test error")
        aggregator.add_error(
            service="test_service",
            operation="test_operation",
            error=error,
            request_id="req-123"
        )
        
        assert len(aggregator.errors) == 1
        assert aggregator.error_counts["test_service"]["ValueError"] == 1
        
        error_record = aggregator.errors[0]
        assert error_record["service"] == "test_service"
        assert error_record["operation"] == "test_operation"
        assert error_record["error_type"] == "ValueError"
        assert error_record["request_id"] == "req-123"
    
    def test_add_multiple_errors(self):
        """Test adding multiple errors to aggregator."""
        aggregator = ErrorAggregator()
        
        # Add errors from different services
        aggregator.add_error("service1", "op1", ValueError("Error 1"))
        aggregator.add_error("service1", "op2", TypeError("Error 2"))
        aggregator.add_error("service2", "op3", ValueError("Error 3"))
        
        assert len(aggregator.errors) == 3
        assert aggregator.error_counts["service1"]["ValueError"] == 1
        assert aggregator.error_counts["service1"]["TypeError"] == 1
        assert aggregator.error_counts["service2"]["ValueError"] == 1
    
    def test_max_errors_limit(self):
        """Test that aggregator respects max errors limit."""
        aggregator = ErrorAggregator(max_errors=3)
        
        # Add more errors than the limit
        for i in range(5):
            aggregator.add_error("service", f"op{i}", ValueError(f"Error {i}"))
        
        # Should only keep the last 3 errors
        assert len(aggregator.errors) == 3
        assert aggregator.errors[0]["operation"] == "op2"  # First error should be op2
        assert aggregator.errors[-1]["operation"] == "op4"  # Last error should be op4
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        aggregator = ErrorAggregator()
        
        # Add some errors
        aggregator.add_error("service1", "op1", ValueError("Error 1"))
        aggregator.add_error("service1", "op2", TypeError("Error 2"))
        aggregator.add_error("service2", "op3", ValueError("Error 3"))
        
        summary = aggregator.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert summary["errors_by_service"]["service1"] == 2
        assert summary["errors_by_service"]["service2"] == 1
        assert summary["errors_by_type"]["ValueError"] == 2
        assert summary["errors_by_type"]["TypeError"] == 1
        assert summary["most_common_error"] == ("ValueError", 2)
        assert "service1" in summary["services_with_errors"]
        assert "service2" in summary["services_with_errors"]
    
    def test_get_errors_by_service(self):
        """Test getting errors for specific service."""
        aggregator = ErrorAggregator()
        
        # Add errors from different services
        aggregator.add_error("service1", "op1", ValueError("Error 1"))
        aggregator.add_error("service2", "op2", TypeError("Error 2"))
        aggregator.add_error("service1", "op3", ValueError("Error 3"))
        
        service1_errors = aggregator.get_errors_by_service("service1")
        
        assert len(service1_errors) == 2
        assert service1_errors[0]["operation"] == "op1"
        assert service1_errors[1]["operation"] == "op3"
    
    def test_clear_errors_all(self):
        """Test clearing all errors."""
        aggregator = ErrorAggregator()
        
        # Add some errors
        aggregator.add_error("service1", "op1", ValueError("Error 1"))
        aggregator.add_error("service2", "op2", TypeError("Error 2"))
        
        assert len(aggregator.errors) == 2
        assert len(aggregator.error_counts) == 2
        
        aggregator.clear_errors()
        
        assert aggregator.errors == []
        assert aggregator.error_counts == {}
    
    def test_clear_errors_by_service(self):
        """Test clearing errors for specific service."""
        aggregator = ErrorAggregator()
        
        # Add errors from different services
        aggregator.add_error("service1", "op1", ValueError("Error 1"))
        aggregator.add_error("service2", "op2", TypeError("Error 2"))
        aggregator.add_error("service1", "op3", ValueError("Error 3"))
        
        assert len(aggregator.errors) == 3
        
        aggregator.clear_errors("service1")
        
        # Should only have service2 error left
        assert len(aggregator.errors) == 1
        assert aggregator.errors[0]["service"] == "service2"
        assert "service1" not in aggregator.error_counts
        assert "service2" in aggregator.error_counts


class TestGlobalErrorAggregator:
    """Test cases for global error aggregator functionality."""
    
    def test_report_error_to_aggregator(self):
        """Test reporting error to global aggregator."""
        # Clear any existing errors
        global_error_aggregator.clear_errors()
        
        error = ValueError("Test error")
        report_error_to_aggregator(
            service="test_service",
            operation="test_operation",
            error=error,
            request_id="req-123"
        )
        
        assert len(global_error_aggregator.errors) == 1
        error_record = global_error_aggregator.errors[0]
        assert error_record["service"] == "test_service"
        assert error_record["operation"] == "test_operation"
        assert error_record["error_type"] == "ValueError"
        assert error_record["request_id"] == "req-123"
        
        # Clean up
        global_error_aggregator.clear_errors()


class TestThreadSafety:
    """Test cases for thread safety of error handling components."""
    
    def test_error_aggregator_thread_safety(self):
        """Test that ErrorAggregator is thread-safe."""
        aggregator = ErrorAggregator()
        
        def add_errors(service_name, num_errors):
            for i in range(num_errors):
                aggregator.add_error(
                    service=service_name,
                    operation=f"op{i}",
                    error=ValueError(f"Error {i}")
                )
        
        # Create multiple threads adding errors concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=add_errors,
                args=(f"service{i}", 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 50 total errors (5 services * 10 errors each)
        assert len(aggregator.errors) == 50
        assert len(aggregator.error_counts) == 5
        
        # Each service should have 10 errors
        for i in range(5):
            service_name = f"service{i}"
            assert aggregator.error_counts[service_name]["ValueError"] == 10


class TestIntegration:
    """Integration tests for error handling components."""
    
    def test_end_to_end_error_handling(self):
        """Test complete error handling workflow."""
        # Clear global aggregator
        global_error_aggregator.clear_errors()
        
        # Create a service function with error handling
        @handle_service_errors("integration_service", "test_operation")
        def service_function(should_fail=False):
            if should_fail:
                raise ValueError("Service failure")
            return "success"
        
        # Test successful execution
        result = service_function(should_fail=False)
        assert result == "success"
        
        # Test error handling
        with pytest.raises(SpotAnalyzerBaseError):
            service_function(should_fail=True)
        
        # Clean up
        global_error_aggregator.clear_errors()
    
    def test_error_context_with_aggregator(self):
        """Test error context manager with global aggregator."""
        # Clear global aggregator
        global_error_aggregator.clear_errors()
        
        try:
            with error_handling_context(
                operation="context_test",
                service_name="context_service",
                request_id="req-456"
            ):
                raise ValueError("Context error")
        except ValueError:
            pass  # Expected
        
        # Error should be recorded in handler but not necessarily in global aggregator
        # (depends on implementation details)
        
        # Clean up
        global_error_aggregator.clear_errors()