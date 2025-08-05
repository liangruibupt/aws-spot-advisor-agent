"""
Error response formatting utilities for AWS Spot Price Analyzer.

This module provides utilities for creating structured error responses
that can be returned to clients in a consistent format.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union

from src.utils.exceptions import SpotAnalyzerBaseError


logger = logging.getLogger(__name__)


class ErrorResponseFormatter:
    """
    Formatter for creating structured error responses.
    
    Provides methods to format various types of errors into consistent
    JSON responses that can be returned to clients.
    """
    
    def __init__(self, include_stack_trace: bool = False):
        """
        Initialize the error response formatter.
        
        Args:
            include_stack_trace: Whether to include stack traces in error responses
        """
        self.include_stack_trace = include_stack_trace
    
    def format_error_response(
        self,
        error: Union[Exception, str],
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Format an error into a structured response.
        
        Args:
            error: Exception or error message
            error_code: Optional error code
            details: Additional error details
            request_id: Optional request identifier
            timestamp: Error timestamp (defaults to current time)
            
        Returns:
            Structured error response dictionary
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Handle SpotAnalyzerBaseError instances
        if isinstance(error, SpotAnalyzerBaseError):
            return self._format_custom_error(error, request_id, timestamp)
        
        # Handle other exceptions
        if isinstance(error, Exception):
            return self._format_generic_exception(error, error_code, details, request_id, timestamp)
        
        # Handle string error messages
        return self._format_string_error(str(error), error_code, details, request_id, timestamp)
    
    def format_validation_error(
        self,
        field_errors: Dict[str, List[str]],
        message: str = "Validation failed",
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format validation errors into a structured response.
        
        Args:
            field_errors: Dictionary mapping field names to error messages
            message: Overall error message
            request_id: Optional request identifier
            
        Returns:
            Structured validation error response
        """
        return {
            "error": True,
            "error_code": "VALIDATION_ERROR",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "field_errors": field_errors,
                "total_errors": sum(len(errors) for errors in field_errors.values())
            },
            "request_id": request_id
        }
    
    def format_insufficient_data_error(
        self,
        required_count: int,
        available_count: int,
        criteria: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format insufficient data errors.
        
        Args:
            required_count: Number of items required
            available_count: Number of items available
            criteria: Filtering criteria that caused the shortage
            request_id: Optional request identifier
            
        Returns:
            Structured insufficient data error response
        """
        message = (
            f"Insufficient data available. Required: {required_count}, "
            f"Available: {available_count}"
        )
        
        details = {
            "required_count": required_count,
            "available_count": available_count,
            "shortage": required_count - available_count
        }
        
        if criteria:
            details["criteria"] = criteria
        
        return {
            "error": True,
            "error_code": "INSUFFICIENT_DATA_ERROR",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "request_id": request_id,
            "suggestions": [
                "Try relaxing the filtering criteria",
                "Check if the data source is available",
                "Retry the request later"
            ]
        }
    
    def format_service_unavailable_error(
        self,
        service_name: str,
        retry_after_seconds: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format service unavailable errors.
        
        Args:
            service_name: Name of the unavailable service
            retry_after_seconds: Suggested retry delay
            request_id: Optional request identifier
            
        Returns:
            Structured service unavailable error response
        """
        message = f"Service '{service_name}' is temporarily unavailable"
        
        details = {
            "service_name": service_name,
            "status": "unavailable"
        }
        
        if retry_after_seconds:
            details["retry_after_seconds"] = retry_after_seconds
            message += f". Please retry after {retry_after_seconds} seconds"
        
        return {
            "error": True,
            "error_code": "SERVICE_UNAVAILABLE_ERROR",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "request_id": request_id,
            "suggestions": [
                "Retry the request after the suggested delay",
                "Check service status page",
                "Contact support if the issue persists"
            ]
        }
    
    def format_rate_limit_error(
        self,
        service_name: str,
        retry_after_seconds: float,
        current_rate: Optional[float] = None,
        limit: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format rate limit errors.
        
        Args:
            service_name: Name of the rate-limited service
            retry_after_seconds: Required wait time before retry
            current_rate: Current request rate
            limit: Rate limit threshold
            request_id: Optional request identifier
            
        Returns:
            Structured rate limit error response
        """
        message = f"Rate limit exceeded for service '{service_name}'"
        
        details = {
            "service_name": service_name,
            "retry_after_seconds": retry_after_seconds
        }
        
        if current_rate is not None:
            details["current_rate"] = current_rate
        if limit is not None:
            details["rate_limit"] = limit
        
        return {
            "error": True,
            "error_code": "RATE_LIMIT_ERROR",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "request_id": request_id,
            "suggestions": [
                f"Wait {retry_after_seconds} seconds before retrying",
                "Reduce request frequency",
                "Consider implementing request batching"
            ]
        }
    
    def format_configuration_error(
        self,
        config_key: str,
        issue: str,
        expected_value: Optional[str] = None,
        current_value: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format configuration errors.
        
        Args:
            config_key: Configuration key that has an issue
            issue: Description of the configuration issue
            expected_value: Expected configuration value
            current_value: Current configuration value
            request_id: Optional request identifier
            
        Returns:
            Structured configuration error response
        """
        message = f"Configuration error for '{config_key}': {issue}"
        
        details = {
            "config_key": config_key,
            "issue": issue
        }
        
        if expected_value is not None:
            details["expected_value"] = expected_value
        if current_value is not None:
            details["current_value"] = current_value
        
        return {
            "error": True,
            "error_code": "CONFIGURATION_ERROR",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "request_id": request_id,
            "suggestions": [
                "Check configuration file",
                "Verify environment variables",
                "Consult documentation for correct configuration format"
            ]
        }
    
    def _format_custom_error(
        self,
        error: SpotAnalyzerBaseError,
        request_id: Optional[str],
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Format a custom SpotAnalyzerBaseError."""
        response = error.to_dict()
        response["timestamp"] = timestamp.isoformat()
        
        if request_id:
            response["request_id"] = request_id
        
        # Add stack trace if enabled
        if self.include_stack_trace:
            import traceback
            response["stack_trace"] = traceback.format_exception(
                type(error), error, error.__traceback__
            )
        
        return response
    
    def _format_generic_exception(
        self,
        error: Exception,
        error_code: Optional[str],
        details: Optional[Dict[str, Any]],
        request_id: Optional[str],
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Format a generic exception."""
        response = {
            "error": True,
            "error_code": error_code or type(error).__name__,
            "message": str(error),
            "timestamp": timestamp.isoformat(),
            "details": details or {}
        }
        
        if request_id:
            response["request_id"] = request_id
        
        # Add exception type information
        response["details"]["exception_type"] = type(error).__name__
        
        # Add stack trace if enabled
        if self.include_stack_trace:
            import traceback
            response["stack_trace"] = traceback.format_exception(
                type(error), error, error.__traceback__
            )
        
        return response
    
    def _format_string_error(
        self,
        error_message: str,
        error_code: Optional[str],
        details: Optional[Dict[str, Any]],
        request_id: Optional[str],
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Format a string error message."""
        return {
            "error": True,
            "error_code": error_code or "GENERIC_ERROR",
            "message": error_message,
            "timestamp": timestamp.isoformat(),
            "details": details or {},
            "request_id": request_id
        }
    
    def to_json_string(
        self,
        error_response: Dict[str, Any],
        indent: Optional[int] = None
    ) -> str:
        """
        Convert error response to JSON string.
        
        Args:
            error_response: Error response dictionary
            indent: JSON indentation (None for compact)
            
        Returns:
            JSON string representation
        """
        try:
            return json.dumps(error_response, indent=indent, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize error response to JSON: {e}")
            # Return a basic error response as a string to avoid recursive serialization issues
            fallback_response = {
                "error": True,
                "error_code": "JSON_SERIALIZATION_ERROR",
                "message": "Failed to serialize error response",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {"original_error": str(e)}
            }
            try:
                return json.dumps(fallback_response, indent=indent)
            except (TypeError, ValueError):
                # If even the fallback fails, return a minimal JSON string
                return '{"error": true, "error_code": "JSON_SERIALIZATION_ERROR", "message": "Critical serialization failure"}'


# Global error formatter instance
error_formatter = ErrorResponseFormatter()


def format_error_response(
    error: Union[Exception, str],
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to format error responses.
    
    Args:
        error: Exception or error message
        error_code: Optional error code
        details: Additional error details
        request_id: Optional request identifier
        
    Returns:
        Structured error response dictionary
    """
    return error_formatter.format_error_response(
        error=error,
        error_code=error_code,
        details=details,
        request_id=request_id
    )


def format_error_json(
    error: Union[Exception, str],
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    indent: Optional[int] = None
) -> str:
    """
    Convenience function to format error responses as JSON strings.
    
    Args:
        error: Exception or error message
        error_code: Optional error code
        details: Additional error details
        request_id: Optional request identifier
        indent: JSON indentation (None for compact)
        
    Returns:
        JSON string representation of error response
    """
    error_response = format_error_response(
        error=error,
        error_code=error_code,
        details=details,
        request_id=request_id
    )
    return error_formatter.to_json_string(error_response, indent=indent)