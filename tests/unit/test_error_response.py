"""
Unit tests for error response formatting utilities.

This module tests the error response formatting functionality
for creating structured error responses in the AWS Spot Price Analyzer.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import patch

from src.utils.error_response import (
    ErrorResponseFormatter,
    format_error_response,
    format_error_json
)
from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    WebScrapingError,
    BedrockServiceError,
    DataValidationError,
    InsufficientDataError
)


class TestErrorResponseFormatter:
    """Test cases for the ErrorResponseFormatter class."""
    
    def test_formatter_initialization(self):
        """Test formatter initialization with default settings."""
        formatter = ErrorResponseFormatter()
        assert formatter.include_stack_trace is False
        
        formatter_with_stack_trace = ErrorResponseFormatter(include_stack_trace=True)
        assert formatter_with_stack_trace.include_stack_trace is True
    
    def test_format_string_error(self):
        """Test formatting a string error message."""
        formatter = ErrorResponseFormatter()
        
        response = formatter.format_error_response("Test error message")
        
        assert response["error"] is True
        assert response["error_code"] == "GENERIC_ERROR"
        assert response["message"] == "Test error message"
        assert "timestamp" in response
        assert response["details"] == {}
        assert response.get("request_id") is None
    
    def test_format_string_error_with_parameters(self):
        """Test formatting a string error with additional parameters."""
        formatter = ErrorResponseFormatter()
        details = {"key": "value"}
        
        response = formatter.format_error_response(
            error="Custom error",
            error_code="CUSTOM_ERROR",
            details=details,
            request_id="req-123"
        )
        
        assert response["error_code"] == "CUSTOM_ERROR"
        assert response["message"] == "Custom error"
        assert response["details"] == details
        assert response["request_id"] == "req-123"
    
    def test_format_generic_exception(self):
        """Test formatting a generic exception."""
        formatter = ErrorResponseFormatter()
        exception = ValueError("Invalid value")
        
        response = formatter.format_error_response(exception)
        
        assert response["error"] is True
        assert response["error_code"] == "ValueError"
        assert response["message"] == "Invalid value"
        assert response["details"]["exception_type"] == "ValueError"
        assert "stack_trace" not in response
    
    def test_format_generic_exception_with_stack_trace(self):
        """Test formatting a generic exception with stack trace."""
        formatter = ErrorResponseFormatter(include_stack_trace=True)
        exception = ValueError("Invalid value")
        
        response = formatter.format_error_response(exception)
        
        assert "stack_trace" in response
        assert isinstance(response["stack_trace"], list)
    
    def test_format_custom_error(self):
        """Test formatting a custom SpotAnalyzerBaseError."""
        formatter = ErrorResponseFormatter()
        
        custom_error = WebScrapingError(
            message="Scraping failed",
            url="https://example.com",
            status_code=404
        )
        
        response = formatter.format_error_response(custom_error)
        
        assert response["error"] is True
        assert response["error_code"] == "WEB_SCRAPING_ERROR"
        assert response["message"] == "Scraping failed"
        assert response["details"]["url"] == "https://example.com"
        assert response["details"]["status_code"] == 404
    
    def test_format_custom_error_with_original_error(self):
        """Test formatting a custom error with original error."""
        formatter = ErrorResponseFormatter()
        
        original_error = ConnectionError("Connection failed")
        custom_error = WebScrapingError(
            message="Scraping failed",
            original_error=original_error
        )
        
        response = formatter.format_error_response(custom_error)
        
        assert "original_error" in response
        assert response["original_error"]["type"] == "ConnectionError"
        assert response["original_error"]["message"] == "Connection failed"
    
    def test_format_validation_error(self):
        """Test formatting validation errors."""
        formatter = ErrorResponseFormatter()
        
        field_errors = {
            "spot_price": ["Must be positive", "Must be a number"],
            "region": ["Required field"]
        }
        
        response = formatter.format_validation_error(
            field_errors=field_errors,
            message="Validation failed",
            request_id="req-456"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "VALIDATION_ERROR"
        assert response["message"] == "Validation failed"
        assert response["details"]["field_errors"] == field_errors
        assert response["details"]["total_errors"] == 3
        assert response["request_id"] == "req-456"
    
    def test_format_insufficient_data_error(self):
        """Test formatting insufficient data errors."""
        formatter = ErrorResponseFormatter()
        
        criteria = {"max_interruption_rate": 0.05}
        
        response = formatter.format_insufficient_data_error(
            required_count=3,
            available_count=1,
            criteria=criteria,
            request_id="req-789"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "INSUFFICIENT_DATA_ERROR"
        assert "Insufficient data available" in response["message"]
        assert response["details"]["required_count"] == 3
        assert response["details"]["available_count"] == 1
        assert response["details"]["shortage"] == 2
        assert response["details"]["criteria"] == criteria
        assert "suggestions" in response
        assert len(response["suggestions"]) > 0
    
    def test_format_service_unavailable_error(self):
        """Test formatting service unavailable errors."""
        formatter = ErrorResponseFormatter()
        
        response = formatter.format_service_unavailable_error(
            service_name="AWS Bedrock",
            retry_after_seconds=30.0,
            request_id="req-101"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "SERVICE_UNAVAILABLE_ERROR"
        assert "AWS Bedrock" in response["message"]
        assert "30.0 seconds" in response["message"]
        assert response["details"]["service_name"] == "AWS Bedrock"
        assert response["details"]["retry_after_seconds"] == 30.0
        assert "suggestions" in response
    
    def test_format_rate_limit_error(self):
        """Test formatting rate limit errors."""
        formatter = ErrorResponseFormatter()
        
        response = formatter.format_rate_limit_error(
            service_name="AWS Bedrock",
            retry_after_seconds=60.0,
            current_rate=100.0,
            limit=50.0,
            request_id="req-202"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "RATE_LIMIT_ERROR"
        assert "Rate limit exceeded" in response["message"]
        assert response["details"]["service_name"] == "AWS Bedrock"
        assert response["details"]["retry_after_seconds"] == 60.0
        assert response["details"]["current_rate"] == 100.0
        assert response["details"]["rate_limit"] == 50.0
        assert "suggestions" in response
    
    def test_format_configuration_error(self):
        """Test formatting configuration errors."""
        formatter = ErrorResponseFormatter()
        
        response = formatter.format_configuration_error(
            config_key="max_interruption_rate",
            issue="Invalid value",
            expected_value="float between 0.0 and 1.0",
            current_value="invalid",
            request_id="req-303"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "CONFIGURATION_ERROR"
        assert "max_interruption_rate" in response["message"]
        assert response["details"]["config_key"] == "max_interruption_rate"
        assert response["details"]["issue"] == "Invalid value"
        assert response["details"]["expected_value"] == "float between 0.0 and 1.0"
        assert response["details"]["current_value"] == "invalid"
        assert "suggestions" in response
    
    def test_to_json_string(self):
        """Test converting error response to JSON string."""
        formatter = ErrorResponseFormatter()
        
        error_response = {
            "error": True,
            "error_code": "TEST_ERROR",
            "message": "Test message",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        json_string = formatter.to_json_string(error_response)
        
        # Verify it's valid JSON
        parsed = json.loads(json_string)
        assert parsed == error_response
    
    def test_to_json_string_with_indent(self):
        """Test converting error response to indented JSON string."""
        formatter = ErrorResponseFormatter()
        
        error_response = {
            "error": True,
            "error_code": "TEST_ERROR",
            "message": "Test message"
        }
        
        json_string = formatter.to_json_string(error_response, indent=2)
        
        # Verify it's valid JSON and contains indentation
        parsed = json.loads(json_string)
        assert parsed == error_response
        assert "\n" in json_string  # Should contain newlines from indentation
    
    def test_to_json_string_serialization_error(self):
        """Test handling of JSON serialization errors."""
        formatter = ErrorResponseFormatter()
        
        # Create an object that can't be serialized even with default=str
        # We'll mock json.dumps to raise an exception
        error_response = {
            "error": True,
            "test_field": "test_value"
        }
        
        with patch('json.dumps', side_effect=[TypeError("Mock serialization error"), TypeError("Mock serialization error"), '{"error": true, "error_code": "JSON_SERIALIZATION_ERROR", "message": "Critical serialization failure"}']):
            json_string = formatter.to_json_string(error_response)
        
        # Should return a minimal fallback error response
        parsed = json.loads(json_string)
        assert parsed["error"] is True
        assert parsed["error_code"] == "JSON_SERIALIZATION_ERROR"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_format_error_response_function(self):
        """Test the format_error_response convenience function."""
        response = format_error_response(
            error="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"},
            request_id="req-123"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "TEST_ERROR"
        assert response["message"] == "Test error"
        assert response["details"]["key"] == "value"
        assert response["request_id"] == "req-123"
    
    def test_format_error_json_function(self):
        """Test the format_error_json convenience function."""
        json_string = format_error_json(
            error="Test error",
            error_code="TEST_ERROR",
            indent=2
        )
        
        # Verify it's valid JSON
        parsed = json.loads(json_string)
        assert parsed["error"] is True
        assert parsed["error_code"] == "TEST_ERROR"
        assert parsed["message"] == "Test error"
        
        # Verify indentation
        assert "\n" in json_string
    
    def test_format_error_response_with_exception(self):
        """Test formatting error response with an exception object."""
        exception = ValueError("Invalid input")
        
        response = format_error_response(
            error=exception,
            error_code="VALIDATION_ERROR"
        )
        
        assert response["error"] is True
        assert response["error_code"] == "VALIDATION_ERROR"
        assert response["message"] == "Invalid input"
        assert response["details"]["exception_type"] == "ValueError"
    
    def test_format_error_response_with_custom_error(self):
        """Test formatting error response with custom error."""
        custom_error = DataValidationError(
            message="Invalid field",
            field_name="spot_price",
            field_value=-1.0
        )
        
        response = format_error_response(error=custom_error)
        
        assert response["error"] is True
        assert response["error_code"] == "DATA_VALIDATION_ERROR"
        assert response["message"] == "Invalid field"
        assert response["details"]["field_name"] == "spot_price"
        assert response["details"]["field_value"] == "-1.0"


class TestTimestampHandling:
    """Test cases for timestamp handling in error responses."""
    
    def test_timestamp_format(self):
        """Test that timestamps are in ISO format."""
        formatter = ErrorResponseFormatter()
        
        response = formatter.format_error_response("Test error")
        
        # Verify timestamp is present and in ISO format
        assert "timestamp" in response
        timestamp_str = response["timestamp"]
        
        # Should be able to parse as ISO format
        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        assert isinstance(parsed_timestamp, datetime)
    
    def test_custom_timestamp(self):
        """Test using a custom timestamp."""
        formatter = ErrorResponseFormatter()
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        response = formatter.format_error_response(
            error="Test error",
            timestamp=custom_time
        )
        
        assert response["timestamp"] == "2024-01-01T12:00:00+00:00"


class TestErrorResponseIntegration:
    """Integration tests for error response formatting."""
    
    def test_complete_error_workflow(self):
        """Test complete error formatting workflow."""
        # Create a chain of errors
        root_cause = ConnectionError("Network connection failed")
        network_error = WebScrapingError(
            message="Failed to scrape data",
            url="https://example.com",
            status_code=503,
            original_error=root_cause
        )
        
        formatter = ErrorResponseFormatter(include_stack_trace=True)
        response = formatter.format_error_response(
            error=network_error,
            request_id="req-integration-test"
        )
        
        # Verify all components are present
        assert response["error"] is True
        assert response["error_code"] == "WEB_SCRAPING_ERROR"
        assert response["message"] == "Failed to scrape data"
        assert response["details"]["url"] == "https://example.com"
        assert response["details"]["status_code"] == 503
        assert response["request_id"] == "req-integration-test"
        assert "timestamp" in response
        assert "stack_trace" in response
        assert response["original_error"]["type"] == "ConnectionError"
        assert response["original_error"]["message"] == "Network connection failed"
        
        # Convert to JSON and verify it's valid
        json_string = formatter.to_json_string(response, indent=2)
        parsed = json.loads(json_string)
        assert parsed == response
    
    def test_multiple_error_types_consistency(self):
        """Test that different error types produce consistent response structure."""
        formatter = ErrorResponseFormatter()
        
        # Test different error types
        errors = [
            "String error",
            ValueError("Generic exception"),
            WebScrapingError("Web scraping failed"),
            BedrockServiceError("Bedrock failed"),
            DataValidationError("Validation failed")
        ]
        
        responses = []
        for error in errors:
            response = formatter.format_error_response(error)
            responses.append(response)
        
        # Verify all responses have consistent structure
        required_fields = ["error", "error_code", "message", "timestamp", "details"]
        
        for response in responses:
            for field in required_fields:
                assert field in response
            assert response["error"] is True
            assert isinstance(response["error_code"], str)
            assert isinstance(response["message"], str)
            assert isinstance(response["details"], dict)