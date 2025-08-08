#!/usr/bin/env python3
"""
Test web scraping functionality directly.
"""

import logging
from src.services.bedrock_agent_service import BedrockAgentService
from src.utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_web_scraping():
    """Test web scraping directly."""
    
    # Load config
    config = load_config()
    spot_advisor_url = config.get('spot_advisor_url', 'https://aws.amazon.com/ec2/spot/instance-advisor/')
    
    print(f"🔍 Testing web scraping for URL: {spot_advisor_url}")
    
    try:
        # Create service
        service = BedrockAgentService()
        print(f"✅ BedrockAgentService created with model: {service.model_name}")
        
        # Test connection first
        print("🔧 Testing connection...")
        if not service.test_connection():
            print("❌ Connection test failed")
            return
        print("✅ Connection test passed")
        
        # Test web scraping
        print("🌐 Starting web scraping...")
        content = service.execute_web_scraping(spot_advisor_url)
        
        print(f"📄 Content length: {len(content)} characters")
        print(f"📄 First 500 characters:")
        print("-" * 50)
        print(content[:500])
        print("-" * 50)
        
        # Test parsing
        print("🔍 Testing spot data parsing...")
        instance_types = ["p5en.48xlarge", "p5.48xlarge"]
        spot_data = service.parse_spot_data(content, instance_types)
        
        print(f"📊 Found {len(spot_data)} spot data records")
        for i, data in enumerate(spot_data[:5]):  # Show first 5
            print(f"  {i+1}. {data.region} - {data.instance_type}: ${data.spot_price} ({data.interruption_rate*100:.1f}% interruption)")
        
        if len(spot_data) == 0:
            print("⚠️  No spot data found - this explains the error!")
            print("🔍 Let's check if the content contains expected keywords...")
            
            keywords = ["p5en.48xlarge", "p5.48xlarge", "spot", "price", "region", "interruption"]
            for keyword in keywords:
                count = content.lower().count(keyword.lower())
                print(f"   '{keyword}': {count} occurrences")
        
    except Exception as e:
        print(f"❌ Error during web scraping test: {e}")
        logger.exception("Web scraping test failed")

if __name__ == "__main__":
    test_web_scraping()