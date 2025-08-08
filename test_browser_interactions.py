#!/usr/bin/env python3
"""
Test browser interactions with AWS Spot Instance Advisor.
"""

import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel

def test_browser_interactions():
    """Test browser interactions with AWS Spot Instance Advisor page."""
    
    print("🔍 Testing browser interactions with AWS Spot Instance Advisor...")
    
    # Create Bedrock model
    session = boto3.Session()
    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        boto_session=session,
        temperature=0.1,
    )
    
    try:
        # Method 1: Test AgentCoreBrowser
        print("\n1. Testing AgentCoreBrowser:")
        from strands_tools.browser.agent_core_browser import AgentCoreBrowser
        
        browser_instance = AgentCoreBrowser()
        print(f"   ✅ AgentCoreBrowser created: {browser_instance}")
        
        agent = Agent(
            model=bedrock_model,
            tools=[browser_instance.browser],  # Use the .browser method as the tool
            callback_handler=None
        )
        
        print("   ✅ Agent created with AgentCoreBrowser")
        
        # Test detailed interaction
        response = agent("""
        Use the browser tool to interact with https://aws.amazon.com/ec2/spot/instance-advisor/
        
        DETAILED STEPS:
        1. Navigate to the URL
        2. Wait for the page to fully load
        3. Verify the Region dropdown shows "US East (N. Virginia)"
        4. Verify the OS dropdown shows "Linux"  
        5. In the search box (with magnifying glass icon), type "p5"
        6. Wait for the results to filter and show "2 matches"
        7. Look at the results table with columns:
           - Instance Type
           - vCPU
           - Memory GiB
           - Savings over On-Demand
           - Frequency of interruption
        8. Extract the EXACT data for:
           - p5en.48xlarge: Savings % and Interruption rate
           - p5.48xlarge: Savings % and Interruption rate
        
        Return the data in this format:
        {
            "p5en.48xlarge": {
                "savings_over_ondemand": "XX%",
                "frequency_of_interruption": "XX-XX%" or ">XX%"
            },
            "p5.48xlarge": {
                "savings_over_ondemand": "XX%", 
                "frequency_of_interruption": "XX-XX%" or ">XX%"
            }
        }
        
        CRITICAL: Use the browser tool to actually interact with the page elements.
        """)
        
        print(f"📄 AgentCoreBrowser Response:")
        print(response)
        
    except Exception as e:
        print(f"❌ AgentCoreBrowser failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Method 2: Test LocalChromiumBrowser
        print("\n2. Testing LocalChromiumBrowser:")
        from strands_tools.browser import LocalChromiumBrowser
        
        local_browser = LocalChromiumBrowser()
        print(f"   ✅ LocalChromiumBrowser created: {local_browser}")
        
        agent2 = Agent(
            model=bedrock_model,
            tools=[local_browser.browser],  # Use the .browser method as the tool
            callback_handler=None
        )
        
        print("   ✅ Agent created with LocalChromiumBrowser")
        
        # Test the same interaction
        response2 = agent2("""
        Use the browser tool to interact with https://aws.amazon.com/ec2/spot/instance-advisor/
        
        STEP-BY-STEP INTERACTION:
        1. Navigate to the URL and wait for full page load
        2. Confirm the page shows the AWS EC2 Spot Instance Advisor interface
        3. Check that Region is set to "US East (N. Virginia)"
        4. Check that OS is set to "Linux"
        5. Click on the search box (with magnifying glass icon)
        6. Type "p5" in the search box
        7. Wait for the table to filter and show results
        8. Read the table data for both P5 instances:
           - p5en.48xlarge row: get "Savings over On-Demand" and "Frequency of interruption"
           - p5.48xlarge row: get "Savings over On-Demand" and "Frequency of interruption"
        
        The interruption rates should be visual bars with text like "10-15%" or ">20%".
        
        Return exactly what you see in the table.
        """)
        
        print(f"📄 LocalChromiumBrowser Response:")
        print(response2)
        
    except Exception as e:
        print(f"❌ LocalChromiumBrowser failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_browser_interactions()