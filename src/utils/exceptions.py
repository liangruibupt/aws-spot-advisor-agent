"""
Custom exception classes for AWS Spot Price Analyzer.

This module defines custom exception classes for different error types
that can occur during spot price analysis operations.
"""

from typing import Optional, Dict, Any


class SpotAnalyzerBaseError(Exception):
    """
    Base exception class for all Spot Analyzer errors.
    
    Provides common functionality for error handling including
    error codes, details, and structured error responses.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the base error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for structured responses.
        
        Returns:
            Dictionary representation of the error
        """
        error_dict = {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }
        
        if self.original_error:
            error_dict["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            }
        
        return error_dict


class WebScrapingError(SpotAnalyzerBaseError):
    """Raised when web scraping operations fail."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(
            message=message,
            error_code="WEB_SCRAPING_ERROR",
            details=details,
            original_error=original_error
        )


class BedrockServiceError(SpotAnalyzerBaseError):
    """Raised when AWS Bedrock service operations fail."""
    
    def __init__(
        self,
        message: str,
        service_error_code: Optional[str] = None,
        region: Optional[str] = None,
        model_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if service_error_code:
            details["service_error_code"] = service_error_code
        if region:
            details["region"] = region
        if model_name:
            details["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_code="BEDROCK_SERVICE_ERROR",
            details=details,
            original_error=original_error
        )


class DataValidationError(SpotAnalyzerBaseError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if field_name:
            details["field_name"] = field_name
        if field_value is not None:
            details["field_value"] = str(field_value)
        if validation_rule:
            details["validation_rule"] = validation_rule
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            details=details,
            original_error=original_error
        )


class DataFilteringError(SpotAnalyzerBaseError):
    """Raised when data filtering operations fail."""
    
    def __init__(
        self,
        message: str,
        filter_type: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if filter_type:
            details["filter_type"] = filter_type
        if filter_criteria:
            details["filter_criteria"] = filter_criteria
        
        super().__init__(
            message=message,
            error_code="DATA_FILTERING_ERROR",
            details=details,
            original_error=original_error
        )


class InsufficientDataError(SpotAnalyzerBaseError):
    """Raised when insufficient data is available for analysis."""
    
    def __init__(
        self,
        message: str,
        required_count: Optional[int] = None,
        available_count: Optional[int] = None,
        criteria: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if required_count is not None:
            details["required_count"] = required_count
        if available_count is not None:
            details["available_count"] = available_count
        if criteria:
            details["criteria"] = criteria
        
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA_ERROR",
            details=details,
            original_error=original_error
        )


class ConfigurationError(SpotAnalyzerBaseError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)
        if expected_type:
            details["expected_type"] = expected_type
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            original_error=original_error
        )


class CacheError(SpotAnalyzerBaseError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if cache_key:
            details["cache_key"] = cache_key
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
            original_error=original_error
        )


class RankingError(SpotAnalyzerBaseError):
    """Raised when ranking operations fail."""
    
    def __init__(
        self,
        message: str,
        ranking_criteria: Optional[str] = None,
        data_count: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if ranking_criteria:
            details["ranking_criteria"] = ranking_criteria
        if data_count is not None:
            details["data_count"] = data_count
        
        super().__init__(
            message=message,
            error_code="RANKING_ERROR",
            details=details,
            original_error=original_error
        )


class FormattingError(SpotAnalyzerBaseError):
    """Raised when result formatting fails."""
    
    def __init__(
        self,
        message: str,
        format_type: Optional[str] = None,
        data_type: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if format_type:
            details["format_type"] = format_type
        if data_type:
            details["data_type"] = data_type
        
        super().__init__(
            message=message,
            error_code="FORMATTING_ERROR",
            details=details,
            original_error=original_error
        )


class RetryableError(SpotAnalyzerBaseError):
    """
    Base class for errors that can be retried.
    
    This exception indicates that the operation might succeed
    if retried after a delay.
    """
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
        if max_retries is not None:
            details["max_retries"] = max_retries
        
        super().__init__(
            message=message,
            error_code="RETRYABLE_ERROR",
            details=details,
            original_error=original_error
        )


class NetworkError(RetryableError):
    """Raised when network operations fail."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if url:
            details["url"] = url
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            original_error=original_error
        )
        self.error_code = "NETWORK_ERROR"
        self.details.update(details)


class ServiceUnavailableError(RetryableError):
    """Raised when external services are temporarily unavailable."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if service_name:
            details["service_name"] = service_name
        if status_code is not None:
            details["status_code"] = status_code
        
        super().__init__(
            message=message,
            original_error=original_error
        )
        self.error_code = "SERVICE_UNAVAILABLE_ERROR"
        self.details.update(details)


class RateLimitError(RetryableError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if service_name:
            details["service_name"] = service_name
        
        super().__init__(
            message=message,
            retry_after_seconds=retry_after_seconds,
            original_error=original_error
        )
        self.error_code = "RATE_LIMIT_ERROR"
        self.details.update(details)