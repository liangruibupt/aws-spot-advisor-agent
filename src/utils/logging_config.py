"""Logging configuration utilities."""

import logging
import sys
from typing import Optional


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging format
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup basic logging configuration
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)