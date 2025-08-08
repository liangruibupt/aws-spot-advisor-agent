#!/usr/bin/env python3
"""
Test what interruption rates are actually being assigned.
"""

from src.services.aws_spot_price_service import AwsSpotPriceService
from src.services.bedrock_agent_service import BedrockAgentService

def test_real_rates():
    """Test what interruption rates are actually being assigned."""
    
    print("🔍 Testing real interruption rates...")
    
    try:
        # Create services
        bedrock_service = BedrockAgentService()
        aws_service = AwsSpotPriceService(bedrock_service=bedrock_service)
        
        # Test instance types
        instance_types = ['p5en.48xlarge', 'p5.48xlarge']
        
        print("📊 Getting spot prices with real interruption rates...")
        spot_data = aws_service.get_spot_prices(instance_types=instance_types, max_results=10)
        
        print(f"Found {len(spot_data)} records:")
        for data in spot_data[:5]:  # Show first 5
            print(f"  {data.region} - {data.instance_type}: ${data.spot_price:.2f}, interruption: {data.interruption_rate:.3f} ({data.interruption_rate*100:.1f}%)")
        
        # Test cached rates
        cached_rates = aws_service.get_cached_interruption_rates()
        print(f"\n📋 Cached interruption rates:")
        for key, rate in cached_rates.items():
            print(f"  {key}: {rate:.3f} ({rate*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_rates()