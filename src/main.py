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
        
        # Perform spot price analysis with JSON formatting
        logger.info("Starting spot price analysis...")
        json_result = analyzer.analyze_spot_prices_json(include_summary=True)
        
        # Output results as formatted JSON
        json_string = analyzer.result_formatter.to_json_string(json_result, indent=2)
        print(json_string)
        
        # Log completion info
        result_count = len(json_result.get("results", []))
        logger.info(f"Analysis complete: {result_count} regions returned")
        
        if "warnings" in json_result and json_result["warnings"]:
            logger.warning(f"Analysis completed with {len(json_result['warnings'])} warnings")
            for warning in json_result["warnings"]:
                logger.warning(f"  - {warning}")
        
    except (InsufficientRegionsError, ServiceFailureError, SpotPriceAnalyzerError) as e:
        logger.error(f"Analysis error: {e}")
        # Use the result formatter for consistent error formatting
        analyzer = SpotPriceAnalyzer()
        error_response = analyzer.result_formatter.format_error_response(
            error_message=str(e),
            error_code=type(e).__name__
        )
        error_json = analyzer.result_formatter.to_json_string(error_response, indent=2)
        print(error_json)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Use the result formatter for consistent error formatting
        try:
            analyzer = SpotPriceAnalyzer()
            error_response = analyzer.result_formatter.format_error_response(
                error_message=f"Unexpected error: {e}",
                error_code="UNEXPECTED_ERROR"
            )
            error_json = analyzer.result_formatter.to_json_string(error_response, indent=2)
            print(error_json)
        except:
            # Fallback if formatter fails
            fallback_error = {
                "error": {
                    "message": f"Unexpected error: {e}",
                    "code": "UNEXPECTED_ERROR"
                }
            }
            print(json.dumps(fallback_error, indent=2))
        sys.exit(1)


def run_analysis_with_params(
    instance_types: Optional[List[str]] = None,
    max_interruption_rate: Optional[float] = None,
    top_count: int = 3,
    force_refresh: bool = False,
    include_summary: bool = False
) -> dict:
    """
    Run spot price analysis with custom parameters and return formatted JSON.
    
    Args:
        instance_types: List of instance types to analyze
        max_interruption_rate: Maximum acceptable interruption rate
        top_count: Number of top regions to return
        force_refresh: Force refresh of cached data
        include_summary: Include summary statistics in response
        
    Returns:
        Dictionary containing formatted analysis results or error information
    """
    try:
        analyzer = SpotPriceAnalyzer()
        return analyzer.analyze_spot_prices_json(
            instance_types=instance_types,
            max_interruption_rate=max_interruption_rate,
            top_count=top_count,
            force_refresh=force_refresh,
            include_summary=include_summary
        )
        
    except Exception as e:
        # Return formatted error response
        analyzer = SpotPriceAnalyzer()
        return analyzer.result_formatter.format_error_response(
            error_message=str(e),
            error_code=type(e).__name__,
            details={
                "instance_types": instance_types,
                "max_interruption_rate": max_interruption_rate,
                "top_count": top_count
            }
        )


def run_analysis_json_string(
    instance_types: Optional[List[str]] = None,
    max_interruption_rate: Optional[float] = None,
    top_count: int = 3,
    force_refresh: bool = False,
    include_summary: bool = False,
    indent: Optional[int] = 2
) -> str:
    """
    Run spot price analysis and return formatted JSON string.
    
    Args:
        instance_types: List of instance types to analyze
        max_interruption_rate: Maximum acceptable interruption rate
        top_count: Number of top regions to return
        force_refresh: Force refresh of cached data
        include_summary: Include summary statistics in response
        indent: Number of spaces for JSON indentation
        
    Returns:
        JSON string containing analysis results or error information
    """
    try:
        analyzer = SpotPriceAnalyzer()
        return analyzer.analyze_spot_prices_json_string(
            instance_types=instance_types,
            max_interruption_rate=max_interruption_rate,
            top_count=top_count,
            force_refresh=force_refresh,
            include_summary=include_summary,
            indent=indent
        )
        
    except Exception as e:
        # Return formatted error as JSON string
        analyzer = SpotPriceAnalyzer()
        error_response = analyzer.result_formatter.format_error_response(
            error_message=f"Failed to run analysis: {e}",
            error_code="ANALYSIS_EXECUTION_ERROR"
        )
        return analyzer.result_formatter.to_json_string(error_response, indent=indent)


if __name__ == "__main__":
    main()