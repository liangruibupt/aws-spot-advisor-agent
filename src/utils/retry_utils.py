"""
Retry utilities with exponential backoff for AWS Spot Price Analyzer.

This module provides retry decorators and utilities using the tenacity library
for handling transient failures in web scraping and AWS service operations.
"""

import logging
from typing import Type, Union, Tuple, Callable, Any
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    before_sleep_log,
    after_log
)
from botocore.exceptions import ClientError, BotoCoreError

from src.utils.exceptions import (
    RetryableError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError,
    BedrockServiceError,
    WebScrapingError
)


logger = logging.getLogger(__name__)


def is_retryable_aws_error(exception: Exception) -> bool:
    """
    Determine if an AWS error is retryable.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(exception, ClientError):
        error_code = exception.response.get('Error', {}).get('Code', '')
        
        # Retryable AWS error codes
        retryable_codes = {
            'Throttling',
            'ThrottlingException',
            'RequestLimitExceeded',
            'ServiceUnavailable',
            'InternalError',
            'InternalFailure',
            'ServiceTemporarilyUnavailable',
            'TooManyRequestsException',
            'ProvisionedThroughputExceededException'
        }
        
        return error_code in retryable_codes
    
    if isinstance(exception, BotoCoreError):
        # Network-related boto errors are generally retryable
        return True
    
    return False


def is_retryable_network_error(exception: Exception) -> bool:
    """
    Determine if a network error is retryable.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    # Check for specific network-related exceptions
    try:
        import requests
        from urllib3.exceptions import HTTPError
        
        retryable_exceptions = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            HTTPError,
            NetworkError,
            ServiceUnavailableError
        )
    except ImportError:
        # If requests is not available, use only our custom exceptions
        retryable_exceptions = (
            NetworkError,
            ServiceUnavailableError
        )
    
    if isinstance(exception, retryable_exceptions):
        return True
    
    # Check for HTTP status codes that are retryable
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        retryable_status_codes = {429, 500, 502, 503, 504}
        return exception.response.status_code in retryable_status_codes
    
    return False


def web_scraping_retry(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for web scraping operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        multiplier: Exponential backoff multiplier
        
    Returns:
        Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds
        ),
        retry=(
            retry_if_exception_type((NetworkError, WebScrapingError, ServiceUnavailableError)) |
            retry_if_exception(is_retryable_network_error)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


def aws_service_retry(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for AWS service operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        multiplier: Exponential backoff multiplier
        
    Returns:
        Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds
        ),
        retry=(
            retry_if_exception_type((BedrockServiceError, ServiceUnavailableError, RateLimitError)) |
            retry_if_exception(is_retryable_aws_error)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


def data_processing_retry(
    max_attempts: int = 2,
    min_wait_seconds: float = 0.5,
    max_wait_seconds: float = 10.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for data processing operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        multiplier: Exponential backoff multiplier
        
    Returns:
        Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds
        ),
        retry=retry_if_exception_type(RetryableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


def rate_limit_retry(
    max_attempts: int = 5,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 300.0,
    multiplier: float = 2.0
):
    """
    Retry decorator specifically for rate limit errors.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        multiplier: Exponential backoff multiplier
        
    Returns:
        Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds
        ),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


class RetryContext:
    """
    Context manager for retry operations with custom error handling.
    
    Provides a way to wrap operations with retry logic and custom
    error transformation.
    """
    
    def __init__(
        self,
        operation_name: str,
        max_attempts: int = 3,
        min_wait_seconds: float = 1.0,
        max_wait_seconds: float = 60.0,
        multiplier: float = 2.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,)
    ):
        """
        Initialize retry context.
        
        Args:
            operation_name: Name of the operation for logging
            max_attempts: Maximum number of retry attempts
            min_wait_seconds: Minimum wait time between retries
            max_wait_seconds: Maximum wait time between retries
            multiplier: Exponential backoff multiplier
            retryable_exceptions: Tuple of exception types that should be retried
        """
        self.operation_name = operation_name
        self.max_attempts = max_attempts
        self.min_wait_seconds = min_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.multiplier = multiplier
        self.retryable_exceptions = retryable_exceptions
        
        self.attempt_count = 0
        self.last_exception: Optional[Exception] = None
    
    def __enter__(self):
        """Enter the retry context."""
        self.attempt_count += 1
        logger.debug(f"Starting {self.operation_name} (attempt {self.attempt_count}/{self.max_attempts})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the retry context with error handling."""
        if exc_type is None:
            # Operation succeeded
            if self.attempt_count > 1:
                logger.info(f"{self.operation_name} succeeded after {self.attempt_count} attempts")
            return True
        
        # Operation failed
        self.last_exception = exc_val
        
        # Check if we should retry
        if (self.attempt_count < self.max_attempts and 
            exc_type and issubclass(exc_type, self.retryable_exceptions)):
            
            # Calculate wait time
            wait_time = min(
                self.min_wait_seconds * (self.multiplier ** (self.attempt_count - 1)),
                self.max_wait_seconds
            )
            
            logger.warning(
                f"{self.operation_name} failed (attempt {self.attempt_count}/{self.max_attempts}): "
                f"{exc_val}. Retrying in {wait_time:.1f} seconds..."
            )
            
            import time
            time.sleep(wait_time)
            
            # Suppress the exception to allow retry
            return True
        
        # Don't retry - let the exception propagate
        logger.error(
            f"{self.operation_name} failed after {self.attempt_count} attempts: {exc_val}"
        )
        return False


def with_retry(
    operation_name: str,
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    multiplier: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,)
):
    """
    Decorator to add retry logic to a function.
    
    Args:
        operation_name: Name of the operation for logging
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        multiplier: Exponential backoff multiplier
        retryable_exceptions: Tuple of exception types that should be retried
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    with RetryContext(
                        operation_name=f"{operation_name} ({func.__name__})",
                        max_attempts=max_attempts,
                        min_wait_seconds=min_wait_seconds,
                        max_wait_seconds=max_wait_seconds,
                        multiplier=multiplier,
                        retryable_exceptions=retryable_exceptions
                    ):
                        return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        raise
                    
                    # Calculate wait time for next attempt
                    wait_time = min(
                        min_wait_seconds * (multiplier ** attempt),
                        max_wait_seconds
                    )
                    
                    logger.warning(
                        f"{operation_name} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    
                    import time
                    time.sleep(wait_time)
                    continue
                except Exception:
                    # Non-retryable exception
                    raise
            
            # This should never be reached, but just in case
            raise RuntimeError(f"{operation_name} failed after {max_attempts} attempts")
        
        return wrapper
    return decorator


# Predefined retry decorators for common use cases
web_scraping_retry_default = web_scraping_retry()
aws_service_retry_default = aws_service_retry()
data_processing_retry_default = data_processing_retry()
rate_limit_retry_default = rate_limit_retry()


def create_custom_retry(
    exception_types: Tuple[Type[Exception], ...],
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    multiplier: float = 2.0
):
    """
    Create a custom retry decorator for specific exception types.
    
    Args:
        exception_types: Tuple of exception types to retry on
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        multiplier: Exponential backoff multiplier
        
    Returns:
        Custom retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait_seconds,
            max=max_wait_seconds
        ),
        retry=retry_if_exception_type(exception_types),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )