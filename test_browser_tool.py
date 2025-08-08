#!/usr/bin/env python3
"""
Test browser tool availability.
"""

import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel

def test_browser_tool():
    """Test if browser tool is available and working."""
    
    print("🔍 Testing browser tool availability...")
    
    try:
        # Check available browser implementations
        print("1. Checking browser implementations:")
        from strands_tools.browser.agent_core_browser import AgentCoreBrowser
        from strands_tools.browser import LocalChromiumBrowser
        print("   ✅ AgentCoreBrowser and LocalChromiumBrowser available")
        
        # Use AgentCoreBrowser (recommended for AWS Bedrock AgentCore)
        browser_tool = AgentCoreBrowser()
        print("   ✅ AgentCoreBrowser instance created")
        
        # Create Bedrock model
        session = boto3.Session()
        bedrock_model = BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            boto_session=session,
            temperature=0.1,
        )
        
        # Create agent with AgentCoreBrowser
        agent = Agent(
            model=bedrock_model,
            tools=[browser_tool],
            callback_handler=None
        )
        
        print("✅ Agent created with AgentCoreBrowser")
        
        # Test AWS Spot Instance Advisor page specifically
        print("\n🧪 Test: AWS Spot Instance Advisor")
        response = agent("""
        CRITICAL: Use the browser tool to navigate to https://aws.amazon.com/ec2/spot/instance-advisor/
        
        Steps:
        1. Navigate to the URL using the browser tool
        2. Wait for the page to load completely (it's a dynamic web application)
        3. Search for "p5" in the search box to filter P5 instances
        4. Look at the table and find the "Frequency of interruption" column
        5. Tell me the EXACT interruption rates shown for:
           - p5en.48xlarge
           - p5.48xlarge
        
        The interruption rates should be ranges like "10-15%", ">20%", etc.
        
        Do NOT make up data - use the browser tool to see the actual page content.
        Return exactly what you see on the page.
        """)
        print(f"📄 Response: {response}")
        
        # Test AWS Spot Instance Advisor page
        print("\n🧪 Test 2: AWS Spot Instance Advisor")
        aws_response = agent("""
        Use the browser tool to navigate to https://aws.amazon.com/ec2/spot/instance-advisor/
        
        Wait for the page to fully load, then:
        1. Look for a search box or filter to find P5 instances
        2. Search for "p5" to filter the results
        3. Find the data table showing instance types
        4. Look for columns like "Instance Type", "Frequency of interruption", etc.
        5. Tell me exactly what you see for p5.48xlarge and p5en.48xlarge instances
        6. Report the exact interruption frequency values shown
        
        Be very specific about what data is displayed on the page.
        """)
        
        print(f"📄 AWS Response: {aws_response}")
        
        # Test with more specific instructions
        print("\n🧪 Test 3: Specific P5 instance data extraction")
        specific_response = agent("""
        Use the browser tool to navigate to https://aws.amazon.com/ec2/spot/instance-advisor/
        
        Your task is to extract the EXACT interruption frequency data for these instances:
        - p5.48xlarge
        - p5en.48xlarge
        
        Steps:
        1. Navigate to the URL using browser tool
        2. Wait for dynamic content to load
        3. Use any search/filter functionality to find P5 instances
        4. Locate the "Frequency of interruption" column
        5. Extract the exact values (like "10-15%", ">20%", "<5%", etc.)
        
        Return the data in this format:
        {
            "p5.48xlarge": "EXACT_VALUE_FROM_PAGE",
            "p5en.48xlarge": "EXACT_VALUE_FROM_PAGE",
            "method": "browser_tool",
            "page_loaded": true/false
        }
        """)
        
        print(f"📄 Specific Response: {specific_response}")
        
        # Test with region-specific instructions
        print("\n🧪 Test 4: Region-specific data (US East N. Virginia)")
        region_response = agent("""
        Use the browser tool to navigate to https://aws.amazon.com/ec2/spot/instance-advisor/
        
        CRITICAL: Make sure you're looking at the correct region data.
        
        Steps:
        1. Navigate to the URL using browser tool
        2. Look for a region selector dropdown (usually shows "US East (N. Virginia)" or similar)
        3. Ensure the region is set to "US East (N. Virginia)" / "us-east-1"
        4. Search for "p5" in the search box to filter results
        5. Look at the table and find the exact rows for:
           - p5.48xlarge
           - p5en.48xlarge
        6. Check the "Frequency of interruption" column values
        
        Pay special attention to:
        - Are there visual indicators (like colored bars) showing interruption levels?
        - What exact text/ranges are shown (like "10-15%", ">20%", "<5%")?
        - Are there any tooltips or additional details when hovering over the data?
        
        Return detailed information about what you see, including any visual elements.
        """)
        
        print(f"📄 Region Response: {region_response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_browser_tool()