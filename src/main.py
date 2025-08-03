#!/usr/bin/env python3
"""
AWS Spot Price Analyzer - Main Entry Point

This module provides the main entry point for the AWS Spot Price Analyzer service.
"""

import json
import logging
import sys
from typing import List, Optional

from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.services.spot_price_analyzer import (
    SpotPriceAnalyzer,
    SpotPriceAnalyzerError,
    InsufficientRegionsError,
    ServiceFailureError
)


def main() -> None:
    """Main entry point for the AWS Spot Price Analyzer."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AWS Spot Price Analyzer")
    
    try:
        # Initialize the spot price analyzer
        analyzer = SpotPriceAnalyzer()
        logger.info("Spot Price Analyzer initialized successfully")
        
        # Perform spot price analysis
        logger.info("Starting spot price analysis...")
        result = analyzer.analyze_spot_prices()
        
        # Output results as JSON
        output = result.to_dict()
        print(json.dumps(output, indent=2))
        
        logger.info(f"Analysis complete: {len(result.results)} regions returned")
        
        if result.has_warnings:
            logger.warning(f"Analysis completed with {len(result.warnings)} warnings")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
        
    except InsufficientRegionsError as e:
        logger.error(f"Insufficient regions available: {e}")
        error_response = {
            "error": "insufficient_regions",
            "message": str(e),
            "results": [],
            "total_regions_analyzed": 0,
            "filtered_regions_count": 0
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)
        
    except ServiceFailureError as e:
        logger.error(f"Service failure during analysis: {e}")
        error_response = {
            "error": "service_failure",
            "message": str(e),
            "results": [],
            "total_regions_analyzed": 0,
            "filtered_regions_count": 0
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)
        
    except SpotPriceAnalyzerError as e:
        logger.error(f"Analysis error: {e}")
        error_response = {
            "error": "analysis_error",
            "message": str(e),
            "results": [],
            "total_regions_analyzed": 0,
            "filtered_regions_count": 0
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        error_response = {
            "error": "unexpected_error",
            "message": str(e),
            "results": [],
            "total_regions_analyzed": 0,
            "filtered_regions_count": 0
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)


def run_analysis_with_params(
    instance_types: Optional[List[str]] = None,
    max_interruption_rate: Optional[float] = None,
    top_count: int = 3,
    force_refresh: bool = False
) -> dict:
    """
    Run spot price analysis with custom parameters.
    
    Args:
        instance_types: List of instance types to analyze
        max_interruption_rate: Maximum acceptable interruption rate
        top_count: Number of top regions to return
        force_refresh: Force refresh of cached data
        
    Returns:
        Dictionary containing analysis results or error information
    """
    try:
        analyzer = SpotPriceAnalyzer()
        result = analyzer.analyze_spot_prices(
            instance_types=instance_types,
            max_interruption_rate=max_interruption_rate,
            top_count=top_count,
            force_refresh=force_refresh
        )
        return result.to_dict()
        
    except (InsufficientRegionsError, ServiceFailureError, SpotPriceAnalyzerError) as e:
        return {
            "error": type(e).__name__,
            "message": str(e),
            "results": [],
            "total_regions_analyzed": 0,
            "filtered_regions_count": 0
        }


if __name__ == "__main__":
    main()