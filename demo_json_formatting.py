#!/usr/bin/env python3
"""
Demonstration script for JSON response formatting functionality.

This script shows how the ResultFormatter service structures output according
to requirements with proper timestamp formatting, currency denomination,
and percentage formatting.
"""

import json
from datetime import datetime, timezone
from src.services.result_formatter import ResultFormatter
from src.models.spot_data import SpotPriceResult, AnalysisResponse


def main():
    """Demonstrate JSON formatting functionality."""
    print("=== AWS Spot Price Analyzer - JSON Formatting Demo ===\n")
    
    # Initialize the formatter
    formatter = ResultFormatter()
    
    # Create sample data
    timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
    
    # Sample spot price results
    results = [
        SpotPriceResult(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.5678,
            currency="USD",
            interruption_rate=0.0234,
            rank=1,
            data_timestamp=timestamp
        ),
        SpotPriceResult(
            region="us-west-2",
            instance_type="p5.48xlarge",
            spot_price=13.1234,
            currency="USD",
            interruption_rate=0.0456,
            rank=2,
            data_timestamp=timestamp
        ),
        SpotPriceResult(
            region="eu-west-1",
            instance_type="p5en.48xlarge",
            spot_price=14.7890,
            currency="USD",
            interruption_rate=0.0123,
            rank=3,
            data_timestamp=timestamp
        )
    ]
    
    # Create analysis response
    analysis_response = AnalysisResponse(
        results=results,
        total_regions_analyzed=15,
        filtered_regions_count=8,
        data_collection_timestamp=timestamp,
        warnings=["Only 8 regions met the criteria (interruption rate < 5.0%)"]
    )
    
    # Demonstrate different formatting options
    print("1. Complete Analysis Response (formatted JSON):")
    print("=" * 50)
    formatted_response = formatter.format_analysis_response(analysis_response)
    json_string = formatter.to_json_string(formatted_response, indent=2)
    print(json_string)
    
    print("\n\n2. Results Only (without metadata):")
    print("=" * 50)
    results_only = formatter.format_spot_price_results(results)
    results_json = formatter.to_json_string({"results": results_only}, indent=2)
    print(results_json)
    
    print("\n\n3. Summary Statistics:")
    print("=" * 50)
    summary_stats = formatter.format_summary_statistics(analysis_response)
    summary_json = formatter.to_json_string(summary_stats, indent=2)
    print(summary_json)
    
    print("\n\n4. Error Response Example:")
    print("=" * 50)
    error_response = formatter.format_error_response(
        error_message="Insufficient regions meet the criteria",
        error_code="INSUFFICIENT_REGIONS",
        details={
            "max_interruption_rate": "5.0%",
            "regions_found": 2,
            "regions_required": 3
        }
    )
    error_json = formatter.to_json_string(error_response, indent=2)
    print(error_json)
    
    print("\n\n5. Compact JSON (no indentation):")
    print("=" * 50)
    compact_json = formatter.to_json_string(formatted_response)
    print(compact_json)
    
    print("\n\n6. Validation Example:")
    print("=" * 50)
    try:
        is_valid = formatter.validate_json_structure(formatted_response)
        print(f"JSON structure validation: {'PASSED' if is_valid else 'FAILED'}")
    except Exception as e:
        print(f"JSON structure validation: FAILED - {e}")
    
    print("\n\n=== Key Features Demonstrated ===")
    print("✓ Timestamp formatting in ISO 8601 format with UTC timezone")
    print("✓ Currency denomination (USD) with amount/currency structure")
    print("✓ Interruption rates formatted as percentages with 2 decimal places")
    print("✓ Structured JSON output according to API requirements")
    print("✓ Error response formatting with consistent structure")
    print("✓ Summary statistics with price and interruption rate analysis")
    print("✓ JSON structure validation")
    print("✓ Compact and pretty-printed JSON options")


if __name__ == "__main__":
    main()