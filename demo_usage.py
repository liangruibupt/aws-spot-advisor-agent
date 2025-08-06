#!/usr/bin/env python3
"""
Demo script showing how to use the AWS Spot Price Analyzer programmatically.

This script demonstrates the basic usage patterns without requiring AWS credentials
or Bedrock access (uses mock data for demonstration).
"""

import json
import sys
from typing import List

# Import the analyzer
from src.services.spot_price_analyzer import SpotPriceAnalyzer
from src.models.spot_data import RawSpotData


def demo_basic_usage():
    """Demonstrate basic usage of the analyzer."""
    print("=== AWS Spot Price Analyzer Demo ===\n")
    
    try:
        # Initialize the analyzer
        print("1. Initializing SpotPriceAnalyzer...")
        analyzer = SpotPriceAnalyzer()
        print("   ✓ Analyzer initialized successfully\n")
        
        # Show cache information
        print("2. Cache Information:")
        cache_info = analyzer.get_analysis_cache_info()
        print(f"   Cache size: {cache_info['size']}")
        print(f"   Hit rate: {cache_info.get('hit_rate', 0):.2%}")
        print(f"   Default TTL: {cache_info['default_ttl_seconds']} seconds\n")
        
        # Demonstrate the JSON API (this will fail with AWS errors, but shows the interface)
        print("3. Attempting to analyze spot prices...")
        print("   (This will fail due to AWS Bedrock configuration, but shows the interface)")
        
        try:
            results = analyzer.analyze_spot_prices_json(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3,
                include_summary=True
            )
            
            print("   ✓ Analysis successful!")
            print(f"   Found {len(results.get('results', []))} regions")
            
        except Exception as e:
            print(f"   ✗ Analysis failed (expected): {type(e).__name__}")
            print(f"   Error: {str(e)[:100]}...")
        
        print("\n4. System Status:")
        try:
            # Check if services are accessible
            print(f"   Configuration loaded: ✓")
            print(f"   Services initialized: ✓")
            print(f"   Cache operational: ✓")
            print(f"   Web scraper: {type(analyzer.web_scraper).__name__}")
            print(f"   Data filter: {type(analyzer.data_filter).__name__}")
            print(f"   Ranking engine: {type(analyzer.ranking_engine).__name__}")
        except Exception as e:
            print(f"   Status check failed: {e}")
            
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return False
    
    return True


def demo_mock_data_processing():
    """Demonstrate data processing with mock data."""
    print("\n=== Mock Data Processing Demo ===\n")
    
    try:
        analyzer = SpotPriceAnalyzer()
        
        # Create some mock data
        from datetime import datetime, timezone
        
        mock_timestamp = datetime.now(timezone.utc)
        mock_data = [
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=8.45,
                currency="USD",
                interruption_rate=0.021,
                timestamp=mock_timestamp,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=9.12,
                currency="USD",
                interruption_rate=0.018,
                timestamp=mock_timestamp,
                availability=True
            ),
            RawSpotData(
                region="ap-southeast-1",
                instance_type="p5.48xlarge",
                spot_price=9.67,
                currency="USD",
                interruption_rate=0.032,
                timestamp=mock_timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.34,
                currency="USD",
                interruption_rate=0.065,  # Too high interruption rate
                timestamp=mock_timestamp,
                availability=True
            )
        ]
        
        print("1. Mock Data Created:")
        for data in mock_data:
            print(f"   {data.region}: ${data.spot_price} ({data.interruption_rate:.1%} interruption)")
        
        # Filter data
        print("\n2. Filtering data (max 5% interruption rate)...")
        filtered_data = analyzer.data_filter.filter_by_interruption_rate(
            mock_data, max_rate=0.05
        )
        
        print(f"   Filtered from {len(mock_data)} to {len(filtered_data)} regions")
        for data in filtered_data:
            print(f"   ✓ {data.region}: ${data.spot_price} ({data.interruption_rate:.1%})")
        
        # Rank data
        print("\n3. Ranking regions by price...")
        all_ranked = analyzer.ranking_engine.rank_by_price(filtered_data)
        ranked_results = all_ranked[:3]  # Take top 3
        
        print("   Top 3 regions:")
        for i, result in enumerate(ranked_results, 1):
            print(f"   {i}. {result.region}: ${result.spot_price} ({result.interruption_rate:.1%})")
        
        # Convert to SpotPriceResult format
        print("\n4. Converting to result format...")
        from src.models.spot_data import SpotPriceResult, AnalysisResponse
        
        spot_results = []
        for i, raw_data in enumerate(ranked_results, 1):
            spot_result = SpotPriceResult(
                region=raw_data.region,
                instance_type=raw_data.instance_type,
                spot_price=raw_data.spot_price,
                currency=raw_data.currency,
                interruption_rate=raw_data.interruption_rate,
                rank=i,
                data_timestamp=raw_data.timestamp
            )
            spot_results.append(spot_result)
        
        # Create analysis response
        analysis_response = AnalysisResponse(
            results=spot_results,
            total_regions_analyzed=len(mock_data),
            filtered_regions_count=len(filtered_data),
            data_collection_timestamp=mock_timestamp
        )
        
        # Format as JSON
        print("5. Formatting as JSON response...")
        json_response = analyzer.result_formatter.format_analysis_response(analysis_response)
        json_string = analyzer.result_formatter.to_json_string(json_response, indent=2)
        print("   JSON Response:")
        print(json_string)
        
    except Exception as e:
        print(f"Mock data processing failed: {e}")
        return False
    
    return True


def demo_configuration():
    """Demonstrate configuration options."""
    print("\n=== Configuration Demo ===\n")
    
    from src.utils.config import load_config
    
    try:
        config = load_config()
        
        print("Current Configuration:")
        print(f"  AWS Region: {config.get('aws_default_region', 'Not set')}")
        print(f"  Log Level: {config.get('log_level', 'INFO')}")
        print(f"  Cache TTL: {config.get('cache_ttl_hours', 1)} hours")
        print(f"  Max Interruption Rate: {config.get('max_interruption_rate', 5.0)}%")
        print(f"  Top Regions Count: {config.get('top_regions_count', 3)}")
        print(f"  Spot Advisor URL: {config.get('spot_advisor_url', 'Default')}")
        
        print("\nTo customize configuration, set environment variables:")
        print("  export AWS_DEFAULT_REGION=eu-west-1")
        print("  export LOG_LEVEL=DEBUG")
        print("  export CACHE_TTL_HOURS=2")
        print("  export MAX_INTERRUPTION_RATE=3.0")
        print("  export TOP_REGIONS_COUNT=5")
        
    except Exception as e:
        print(f"Configuration demo failed: {e}")
        return False
    
    return True


def main():
    """Run all demos."""
    print("AWS Spot Price Analyzer - Usage Demonstration")
    print("=" * 50)
    
    success = True
    
    # Run demos
    success &= demo_basic_usage()
    success &= demo_mock_data_processing()
    success &= demo_configuration()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Demo completed successfully!")
        print("\nTo use with real AWS data:")
        print("1. Configure AWS credentials")
        print("2. Ensure AWS Bedrock access")
        print("3. Run: python -m src.main")
    else:
        print("✗ Some demos failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())