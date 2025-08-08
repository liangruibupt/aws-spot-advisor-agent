#!/usr/bin/env python3
"""
Test interruption rate scraping directly.
"""

from src.services.bedrock_agent_service import BedrockAgentService

def test_interruption_scraping():
    """Test what the Bedrock agent actually returns when scraping interruption rates."""
    
    print("🔍 Testing interruption rate scraping...")
    
    try:
        # Create Bedrock service
        bedrock_service = BedrockAgentService()
        
        # Test URL
        spot_advisor_url = "https://aws.amazon.com/ec2/spot/instance-advisor/"
        
        # Create specific instructions for getting interruption rate data using browser tool
        scraping_instructions = """
        CRITICAL: This is a DYNAMIC web application. You MUST use the browser tool, NOT http_request.
        
        STEP 1: Use the browser tool to navigate to the URL:
        browser(action="navigate", url="https://aws.amazon.com/ec2/spot/instance-advisor/")
        
        STEP 2: Wait for the page to fully load (it uses JavaScript to load data dynamically)
        
        STEP 3: Look for the search box and search for "p5" to filter P5 instances
        
        STEP 4: Set the region to "US East (N. Virginia)" if there's a region selector
        
        STEP 5: Find the data table with these columns:
        - Instance Type
        - vCPU
        - Memory GiB  
        - Savings over On-Demand
        - Frequency of interruption
        
        STEP 6: Find the rows for:
        - p5en.48xlarge
        - p5.48xlarge
        
        STEP 7: Extract the EXACT text from the "Frequency of interruption" column for each instance type.
        This will be ranges like "10-15%", ">20%", "<5%", etc.
        
        STEP 8: Return the extracted data in this format:
        {
            "p5en.48xlarge": "EXACT_FREQUENCY_TEXT",
            "p5.48xlarge": "EXACT_FREQUENCY_TEXT",
            "data_source": "AWS Spot Instance Advisor",
            "method_used": "browser_tool"
        }
        
        CRITICAL REQUIREMENTS:
        - You MUST use the browser tool because this is a dynamic web application
        - Do NOT use http_request - it won't work for dynamic content
        - Do NOT make up or estimate data
        - Return exactly what you see in the interruption frequency column
        
        If you cannot use the browser tool, say "BROWSER_TOOL_NOT_AVAILABLE" and explain why.
        """
        
        print("🌐 Scraping AWS Spot Instance Advisor...")
        content = bedrock_service.execute_web_scraping(
            url=spot_advisor_url,
            custom_instructions=scraping_instructions
        )
        
        print(f"📄 Raw content from agent ({len(content)} chars):")
        print("=" * 80)
        print(content)
        print("=" * 80)
        
        # Test parsing
        print("\n🔍 Testing parsing...")
        from src.services.aws_spot_price_service import AwsSpotPriceService
        service = AwsSpotPriceService()
        
        p5en_rate = service._parse_interruption_rate_response(content, "p5en.48xlarge", "us-east-1")
        p5_rate = service._parse_interruption_rate_response(content, "p5.48xlarge", "us-east-1")
        
        print(f"📊 Parsed rates:")
        print(f"  p5en.48xlarge: {p5en_rate:.3f} ({p5en_rate*100:.1f}%)")
        print(f"  p5.48xlarge: {p5_rate:.3f} ({p5_rate*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interruption_scraping()