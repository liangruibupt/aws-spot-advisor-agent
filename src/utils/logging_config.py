"""
Logging configuration utilities for AWS Spot Price Analyzer.

This module provides enhanced logging configuration with structured logging,
error context tracking, and integration with the error handling system.
"""

import logging
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from pathlib import Path

from src.utils.exceptions import SpotAnalyzerBaseError


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that creates structured log entries with error context.
    
    This formatter adds additional context to log entries including error details,
    request IDs, and structured data for better observability.
    """
    
    def __init__(self, include_stack_trace: bool = False):
        """
        Initialize the structured formatter.
        
        Args:
            include_stack_trace: Whether to include stack traces in error logs
        """
        super().__init__()
        self.include_stack_trace = include_stack_trace
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured data.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message as JSON string
        """
        # Base log entry structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add thread and process info if available
        if hasattr(record, 'thread') and record.thread:
            log_entry["thread_id"] = record.thread
        if hasattr(record, 'process') and record.process:
            log_entry["process_id"] = record.process
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        # Add operation context if available
        if hasattr(record, 'operation'):
            log_entry["operation"] = record.operation
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'request_id', 'operation']:
                if not key.startswith('_'):
                    log_entry["extra"] = log_entry.get("extra", {})
                    log_entry["extra"][key] = value
        
        # Handle exceptions
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None
            }
            
            # Add stack trace if enabled
            if self.include_stack_trace:
                log_entry["exception"]["stack_trace"] = traceback.format_exception(*record.exc_info)
            
            # Add custom error details for SpotAnalyzerBaseError
            if isinstance(record.exc_info[1], SpotAnalyzerBaseError):
                error = record.exc_info[1]
                log_entry["exception"]["error_code"] = error.error_code
                log_entry["exception"]["details"] = error.details
                
                if error.original_error:
                    log_entry["exception"]["original_error"] = {
                        "type": type(error.original_error).__name__,
                        "message": str(error.original_error)
                    }
        
        # Add stack info if available
        if record.stack_info and self.include_stack_trace:
            log_entry["stack_info"] = record.stack_info
        
        try:
            return json.dumps(log_entry, default=str, separators=(',', ':'))
        except (TypeError, ValueError):
            # Fallback to simple format if JSON serialization fails
            return f"{log_entry['timestamp']} - {log_entry['logger']} - {log_entry['level']} - {log_entry['message']}"


class ErrorContextFilter(logging.Filter):
    """
    Logging filter that adds error context to log records.
    
    This filter enriches log records with additional context information
    that can be useful for debugging and monitoring.
    """
    
    def __init__(self, service_name: str = "spot-price-analyzer"):
        """
        Initialize the error context filter.
        
        Args:
            service_name: Name of the service for context
        """
        super().__init__()
        self.service_name = service_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context information to log record.
        
        Args:
            record: Log record to filter/enhance
            
        Returns:
            True to allow the record to be processed
        """
        # Add service name
        record.service = self.service_name
        
        # Add correlation ID if not present
        if not hasattr(record, 'request_id'):
            record.request_id = None
        
        return True


def setup_logging(
    log_level: str = 'INFO',
    structured: bool = False,
    include_stack_trace: bool = False,
    log_file: Optional[str] = None,
    service_name: str = "spot-price-analyzer"
) -> None:
    """
    Setup enhanced logging configuration for the application.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        include_stack_trace: Whether to include stack traces in error logs
        log_file: Optional file path for log output
        service_name: Name of the service for logging context
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    # Configure formatter
    if structured:
        formatter = StructuredFormatter(include_stack_trace=include_stack_trace)
    else:
        log_format = (
            '%(asctime)s - %(service)s - %(name)s - %(levelname)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        )
        if include_stack_trace:
            log_format += ' - %(pathname)s'
        formatter = logging.Formatter(log_format)
    
    # Configure handlers
    for handler in handlers:
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        handler.addFilter(ErrorContextFilter(service_name))
    
    # Setup root logger
    root_logger.setLevel(numeric_level)
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('tenacity').setLevel(logging.INFO)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={log_level}, structured={structured}, "
        f"stack_trace={include_stack_trace}, file={log_file}"
    )


def get_logger_with_context(
    name: str,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs
) -> logging.Logger:
    """
    Get a logger with additional context information.
    
    Args:
        name: Logger name
        request_id: Optional request ID for correlation
        operation: Optional operation name for context
        **kwargs: Additional context fields
        
    Returns:
        Logger instance with context
    """
    logger = logging.getLogger(name)
    
    # Create a custom adapter that adds context to all log calls
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Add context to extra
            extra = kwargs.get('extra', {})
            if request_id:
                extra['request_id'] = request_id
            if operation:
                extra['operation'] = operation
            
            # Add any additional context
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value
            
            kwargs['extra'] = extra
            return msg, kwargs
    
    context = kwargs.copy()
    if request_id:
        context['request_id'] = request_id
    if operation:
        context['operation'] = operation
    
    return ContextAdapter(logger, context)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    message: str,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    **context
) -> None:
    """
    Log an error with full context information.
    
    Args:
        logger: Logger instance to use
        error: Exception that occurred
        message: Log message
        request_id: Optional request ID for correlation
        operation: Optional operation name
        **context: Additional context fields
    """
    extra = context.copy()
    if request_id:
        extra['request_id'] = request_id
    if operation:
        extra['operation'] = operation
    
    # Add error details for custom exceptions
    if isinstance(error, SpotAnalyzerBaseError):
        extra['error_code'] = error.error_code
        extra['error_details'] = error.details
    
    logger.error(message, exc_info=error, extra=extra)


def setup_development_logging() -> None:
    """Setup logging configuration optimized for development."""
    setup_logging(
        log_level='DEBUG',
        structured=False,
        include_stack_trace=True,
        service_name="spot-price-analyzer-dev"
    )


def setup_production_logging(log_file: str = "/var/log/spot-price-analyzer.log") -> None:
    """
    Setup logging configuration optimized for production.
    
    Args:
        log_file: Path to log file
    """
    setup_logging(
        log_level='INFO',
        structured=True,
        include_stack_trace=False,
        log_file=log_file,
        service_name="spot-price-analyzer"
    )