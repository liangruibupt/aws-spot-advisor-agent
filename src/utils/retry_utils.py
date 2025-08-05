"""
Retry utilities with exponential backoff for AWS Spot Price Analyzer.

This module provides retry decorators and utilities using the tenacity library
for handling transient failures in web scraping and AWS service operations.
"""

import logging
import threading
from typing import Type, Union, Tuple, Callable, Any, List, Optional, Dict
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


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for error handling.
    
    Provides automatic failure detection and recovery for external service calls.
    When failures exceed a threshold, the circuit opens and fails fast until
    a recovery period passes.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply circuit breaker to a function.
        
        Args:
            func: Function to wrap with circuit breaker
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN state")
                else:
                    raise ServiceUnavailableError(
                        message="Circuit breaker is OPEN - service unavailable",
                        service_name=func.__name__,
                        retry_after_seconds=self.recovery_timeout
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        import time
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        import time
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """
    Circuit breaker decorator factory.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker
        
    Returns:
        Circuit breaker decorator
    """
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )


class BulkheadPattern:
    """
    Bulkhead pattern implementation for resource isolation.
    
    Limits concurrent operations to prevent resource exhaustion
    and isolate failures in different parts of the system.
    """
    
    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0):
        """
        Initialize bulkhead.
        
        Args:
            max_concurrent: Maximum concurrent operations allowed
            timeout: Timeout for acquiring semaphore
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_count = 0
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply bulkhead pattern to a function.
        
        Args:
            func: Function to wrap with bulkhead
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_bulkhead(func, *args, **kwargs)
        return wrapper
    
    def _call_with_bulkhead(self, func: Callable, *args, **kwargs):
        """Execute function with bulkhead protection."""
        acquired = self.semaphore.acquire(timeout=self.timeout)
        if not acquired:
            raise ServiceUnavailableError(
                message=f"Bulkhead limit exceeded - max {self.max_concurrent} concurrent operations",
                service_name=func.__name__,
                retry_after_seconds=self.timeout
            )
        
        try:
            with self._lock:
                self.active_count += 1
            
            logger.debug(f"Bulkhead: {self.active_count}/{self.max_concurrent} active operations")
            return func(*args, **kwargs)
        finally:
            with self._lock:
                self.active_count -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current bulkhead statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "active_count": self.active_count,
            "available_slots": self.max_concurrent - self.active_count,
            "timeout": self.timeout
        }


def bulkhead(max_concurrent: int = 10, timeout: float = 30.0):
    """
    Bulkhead decorator factory.
    
    Args:
        max_concurrent: Maximum concurrent operations allowed
        timeout: Timeout for acquiring semaphore
        
    Returns:
        Bulkhead decorator
    """
    return BulkheadPattern(max_concurrent=max_concurrent, timeout=timeout)


class ErrorBudget:
    """
    Error budget implementation for SLA management.
    
    Tracks error rates and provides mechanisms to fail fast
    when error budgets are exhausted.
    """
    
    def __init__(
        self,
        error_threshold: float = 0.1,  # 10% error rate
        time_window: float = 3600.0,   # 1 hour window
        min_requests: int = 10         # Minimum requests before enforcing
    ):
        """
        Initialize error budget.
        
        Args:
            error_threshold: Maximum allowed error rate (0.0-1.0)
            time_window: Time window for error rate calculation (seconds)
            min_requests: Minimum requests before enforcing budget
        """
        self.error_threshold = error_threshold
        self.time_window = time_window
        self.min_requests = min_requests
        
        self.requests: List[Tuple[float, bool]] = []  # (timestamp, is_error)
        self._lock = threading.Lock()
    
    def record_request(self, is_error: bool = False) -> None:
        """
        Record a request outcome.
        
        Args:
            is_error: Whether the request resulted in an error
        """
        import time
        current_time = time.time()
        
        with self._lock:
            # Add new request
            self.requests.append((current_time, is_error))
            
            # Clean old requests outside time window
            cutoff_time = current_time - self.time_window
            self.requests = [
                (timestamp, error) for timestamp, error in self.requests
                if timestamp > cutoff_time
            ]
    
    def check_budget(self) -> Dict[str, Any]:
        """
        Check current error budget status.
        
        Returns:
            Dictionary with budget status information
        """
        with self._lock:
            total_requests = len(self.requests)
            error_requests = sum(1 for _, is_error in self.requests if is_error)
            
            if total_requests < self.min_requests:
                return {
                    "budget_exhausted": False,
                    "error_rate": 0.0,
                    "total_requests": total_requests,
                    "error_requests": error_requests,
                    "min_requests_met": False,
                    "threshold": self.error_threshold
                }
            
            error_rate = error_requests / total_requests
            budget_exhausted = error_rate > self.error_threshold
            
            return {
                "budget_exhausted": budget_exhausted,
                "error_rate": error_rate,
                "total_requests": total_requests,
                "error_requests": error_requests,
                "min_requests_met": True,
                "threshold": self.error_threshold,
                "remaining_budget": max(0, self.error_threshold - error_rate)
            }
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply error budget to a function.
        
        Args:
            func: Function to wrap with error budget
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check budget before execution
            budget_status = self.check_budget()
            if budget_status["budget_exhausted"]:
                raise ServiceUnavailableError(
                    message=f"Error budget exhausted - error rate {budget_status['error_rate']:.2%} exceeds threshold {self.error_threshold:.2%}",
                    service_name=func.__name__,
                    retry_after_seconds=self.time_window / 4  # Suggest waiting 1/4 of window
                )
            
            # Execute function and record outcome
            try:
                result = func(*args, **kwargs)
                self.record_request(is_error=False)
                return result
            except Exception as e:
                self.record_request(is_error=True)
                raise
        
        return wrapper


def error_budget(
    error_threshold: float = 0.1,
    time_window: float = 3600.0,
    min_requests: int = 10
):
    """
    Error budget decorator factory.
    
    Args:
        error_threshold: Maximum allowed error rate (0.0-1.0)
        time_window: Time window for error rate calculation (seconds)
        min_requests: Minimum requests before enforcing budget
        
    Returns:
        Error budget decorator
    """
    return ErrorBudget(
        error_threshold=error_threshold,
        time_window=time_window,
        min_requests=min_requests
    )