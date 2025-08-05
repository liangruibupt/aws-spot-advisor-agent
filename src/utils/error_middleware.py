"""
Error handling middleware for AWS Spot Price Analyzer.

This module provides centralized error handling patterns and middleware
for consistent error processing across all services.
"""

import logging
import functools
import traceback
import threading
from datetime import datetime, timezone
from typing import Callable, Any, Dict, Optional, Type, Union, List
from contextlib import contextmanager

from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    WebScrapingError,
    BedrockServiceError,
    DataValidationError,
    ServiceUnavailableError,
    NetworkError,
    RateLimitError,
    ConfigurationError
)
from src.utils.error_response import ErrorResponseFormatter
from src.utils.logging_config import log_error_with_context


logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handler for consistent error processing.
    
    Provides methods for handling different types of errors with
    appropriate logging, formatting, and response generation.
    """
    
    def __init__(
        self,
        service_name: str,
        include_stack_trace: bool = False,
        log_errors: bool = True
    ):
        """
        Initialize error handler.
        
        Args:
            service_name: Name of the service using this handler
            include_stack_trace: Whether to include stack traces in responses
            log_errors: Whether to log errors automatically
        """
        self.service_name = service_name
        self.include_stack_trace = include_stack_trace
        self.log_errors = log_errors
        self.error_formatter = ErrorResponseFormatter(include_stack_trace)
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.last_errors: List[Dict[str, Any]] = []
        self.max_last_errors = 100
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error with logging and formatting.
        
        Args:
            error: Exception that occurred
            operation: Name of the operation that failed
            request_id: Optional request ID for correlation
            context: Additional context information
            
        Returns:
            Formatted error response dictionary
        """
        # Record error statistics
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error record
        error_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "operation": operation,
            "error_type": error_type,
            "error_message": str(error),
            "request_id": request_id,
            "context": context or {}
        }
        
        # Add to recent errors (with size limit)
        self.last_errors.append(error_record)
        if len(self.last_errors) > self.max_last_errors:
            self.last_errors.pop(0)
        
        # Log error if enabled
        if self.log_errors:
            log_error_with_context(
                logger=logger,
                error=error,
                message=f"Error in {operation}",
                request_id=request_id,
                operation=operation,
                service=self.service_name,
                **(context or {})
            )
        
        # Format error response
        return self.error_formatter.format_error_response(
            error=error,
            request_id=request_id,
            details=context
        )
    
    def wrap_with_error_handling(
        self,
        operation_name: str,
        error_mappings: Optional[Dict[Type[Exception], Type[SpotAnalyzerBaseError]]] = None
    ) -> Callable:
        """
        Decorator to wrap functions with error handling.
        
        Args:
            operation_name: Name of the operation for logging
            error_mappings: Optional mapping of exception types to custom errors
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                request_id = kwargs.pop('_request_id', None)
                context = kwargs.pop('_error_context', {})
                
                try:
                    return func(*args, **kwargs)
                except SpotAnalyzerBaseError:
                    # Re-raise custom errors as-is
                    raise
                except Exception as e:
                    # Map to custom error if mapping provided
                    if error_mappings and type(e) in error_mappings:
                        custom_error_class = error_mappings[type(e)]
                        raise custom_error_class(
                            message=f"{operation_name} failed: {e}",
                            original_error=e
                        )
                    
                    # Handle as generic error
                    error_response = self.handle_error(
                        error=e,
                        operation=operation_name,
                        request_id=request_id,
                        context=context
                    )
                    
                    # Raise appropriate custom error
                    raise SpotAnalyzerBaseError(
                        message=f"{operation_name} failed: {e}",
                        details=error_response.get("details", {}),
                        original_error=e
                    )
            
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            "service": self.service_name,
            "total_errors": total_errors,
            "error_counts_by_type": self.error_counts.copy(),
            "recent_errors_count": len(self.last_errors),
            "last_error_time": (
                self.last_errors[-1]["timestamp"] if self.last_errors else None
            )
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors for debugging.
        
        Args:
            limit: Maximum number of recent errors to return
            
        Returns:
            List of recent error records
        """
        return self.last_errors[-limit:] if self.last_errors else []
    
    def clear_statistics(self) -> None:
        """Clear error statistics and recent errors."""
        self.error_counts.clear()
        self.last_errors.clear()


@contextmanager
def error_handling_context(
    operation: str,
    service_name: str = "spot-price-analyzer",
    request_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
):
    """
    Context manager for error handling.
    
    Args:
        operation: Name of the operation
        service_name: Name of the service
        request_id: Optional request ID
        context: Additional context
        reraise: Whether to re-raise exceptions after handling
        
    Yields:
        ErrorHandler instance for the context
    """
    handler = ErrorHandler(service_name)
    
    try:
        yield handler
    except Exception as e:
        handler.handle_error(
            error=e,
            operation=operation,
            request_id=request_id,
            context=context
        )
        
        if reraise:
            raise


def handle_service_errors(
    service_name: str,
    operation_name: Optional[str] = None,
    error_mappings: Optional[Dict[Type[Exception], Type[SpotAnalyzerBaseError]]] = None
) -> Callable:
    """
    Decorator for handling service-level errors.
    
    Args:
        service_name: Name of the service
        operation_name: Optional operation name (defaults to function name)
        error_mappings: Optional mapping of exception types to custom errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        operation = operation_name or func.__name__
        handler = ErrorHandler(service_name)
        
        return handler.wrap_with_error_handling(
            operation_name=operation,
            error_mappings=error_mappings
        )(func)
    
    return decorator


def create_error_mapping() -> Dict[Type[Exception], Type[SpotAnalyzerBaseError]]:
    """
    Create default error mapping for common exceptions.
    
    Returns:
        Dictionary mapping standard exceptions to custom errors
    """
    return {
        ConnectionError: NetworkError,
        TimeoutError: NetworkError,
        ValueError: DataValidationError,
        KeyError: DataValidationError,
        TypeError: DataValidationError,
        FileNotFoundError: ConfigurationError,
        PermissionError: ConfigurationError,
    }


class ErrorAggregator:
    """
    Aggregates errors from multiple sources for analysis.
    
    Useful for collecting errors from different services and
    providing consolidated error reporting.
    """
    
    def __init__(self, max_errors: int = 1000):
        """
        Initialize error aggregator.
        
        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()
    
    def add_error(
        self,
        service: str,
        operation: str,
        error: Exception,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an error to the aggregator.
        
        Args:
            service: Service name where error occurred
            operation: Operation name where error occurred
            error: Exception that occurred
            request_id: Optional request ID
            context: Additional context
        """
        error_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_id": request_id,
            "context": context or {}
        }
        
        with self._lock:
            # Add error record
            self.errors.append(error_record)
            
            # Maintain size limit
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
            
            # Update counts
            if service not in self.error_counts:
                self.error_counts[service] = {}
            
            error_type = type(error).__name__
            self.error_counts[service][error_type] = (
                self.error_counts[service].get(error_type, 0) + 1
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of aggregated errors.
        
        Returns:
            Dictionary with error summary statistics
        """
        with self._lock:
            total_errors = len(self.errors)
            
            # Calculate error rates by service
            service_totals = {}
            for service, error_types in self.error_counts.items():
                service_totals[service] = sum(error_types.values())
            
            # Find most common error types
            all_error_types: Dict[str, int] = {}
            for service_errors in self.error_counts.values():
                for error_type, count in service_errors.items():
                    all_error_types[error_type] = all_error_types.get(error_type, 0) + count
            
            return {
                "total_errors": total_errors,
                "errors_by_service": service_totals,
                "errors_by_type": all_error_types,
                "most_common_error": max(all_error_types.items(), key=lambda x: x[1]) if all_error_types else None,
                "services_with_errors": list(self.error_counts.keys()),
                "time_range": {
                    "oldest": self.errors[0]["timestamp"] if self.errors else None,
                    "newest": self.errors[-1]["timestamp"] if self.errors else None
                }
            }
    
    def get_errors_by_service(self, service: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get errors for a specific service.
        
        Args:
            service: Service name to filter by
            limit: Maximum number of errors to return
            
        Returns:
            List of error records for the service
        """
        with self._lock:
            service_errors = [
                error for error in self.errors
                if error["service"] == service
            ]
            return service_errors[-limit:] if service_errors else []
    
    def clear_errors(self, service: Optional[str] = None) -> None:
        """
        Clear errors, optionally for a specific service.
        
        Args:
            service: Optional service name to clear errors for
        """
        with self._lock:
            if service:
                # Clear errors for specific service
                self.errors = [
                    error for error in self.errors
                    if error["service"] != service
                ]
                if service in self.error_counts:
                    del self.error_counts[service]
            else:
                # Clear all errors
                self.errors.clear()
                self.error_counts.clear()


# Global error aggregator instance
global_error_aggregator = ErrorAggregator()


def report_error_to_aggregator(
    service: str,
    operation: str,
    error: Exception,
    request_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Report an error to the global error aggregator.
    
    Args:
        service: Service name where error occurred
        operation: Operation name where error occurred
        error: Exception that occurred
        request_id: Optional request ID
        context: Additional context
    """
    global_error_aggregator.add_error(
        service=service,
        operation=operation,
        error=error,
        request_id=request_id,
        context=context
    )