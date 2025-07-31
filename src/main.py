#!/usr/bin/env python3
"""
AWS Spot Price Analyzer - Main Entry Point

This module provides the main entry point for the AWS Spot Price Analyzer service.
"""

import logging
import sys
from typing import List

from src.utils.config import load_config
from src.utils.logging_config import setup_logging


def main() -> None:
    """Main entry point for the AWS Spot Price Analyzer."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AWS Spot Price Analyzer")
    
    try:
        # TODO: Initialize and run the spot price analyzer
        # This will be implemented in later tasks
        logger.info("Spot Price Analyzer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Spot Price Analyzer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()