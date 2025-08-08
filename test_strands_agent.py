#!/usr/bin/env python3
"""
Simple test to debug Strands Agent issues.
"""

import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools.http_request import http_request

def test_strands_agent():
    """Test Strands Agent with different model configurations."""
    
    # Test models that we know work with direct bedrock-runtime calls
    test_models = [
        'us.anthropic.claude-3-7-sonnet-20250219-v1:0',  # Inference profile for Sonnet 3.7
        'us.anthropic.claude-sonnet-4-20250514-v1:0',  # Inference profile for Sonnet 4.0
    ]
    
    for model_id in test_models:
        print(f"\n🧪 Testing Strands Agent with: {model_id}")
        
        try:
            # Create Bedrock model
            session = boto3.Session()
            bedrock_model = BedrockModel(
                model_id=model_id,
                boto_session=session,
                temperature=0.3,
            )
            
            # Create agent
            agent = Agent(
                model=bedrock_model,
                tools=[http_request],
                callback_handler=None  # Disable default callback handler
            )
            
            print(f"   ✅ Agent created successfully")
            
            # Test http_request tool with a simple URL first
            print("   🌐 Testing http_request tool with simple URL...")
            test_url = "https://aws.amazon.com/ec2/spot/instance-advisor/"
            response = agent(f"Use the http_request tool to fetch content from {test_url} and show me what you get.")
            
            print(f"   Response type: {type(response)}")
            print(f"   Response: {response}")
            
            if response and hasattr(response, 'message'):
                print(f"   Message: {response.message}")
                
                # Extract text from response
                response_text = ""
                if isinstance(response.message, dict) and 'content' in response.message:
                    for item in response.message['content']:
                        if isinstance(item, dict) and 'text' in item:
                            response_text += item['text']
                
                print(f"   📄 Response text ({len(response_text)} chars):")
                print("   " + "-" * 50)
                print("   " + response_text[:500] + ("..." if len(response_text) > 500 else ""))
                print("   " + "-" * 50)
                
                # Check if the tool was actually used
                if "httpbin" in response_text.lower() or "json" in response_text.lower():
                    print(f"   ✅ SUCCESS: {model_id} successfully used http_request tool!")
                    
                    # Now test with AWS Spot Advisor
                    print("   🌐 Testing with AWS Spot Advisor URL...")
                    aws_response = agent("Use the http_request tool to fetch https://aws.amazon.com/ec2/spot/instance-advisor/ and extract any pricing data you find.")
                    
                    if aws_response and hasattr(aws_response, 'message'):
                        aws_text = ""
                        if isinstance(aws_response.message, dict) and 'content' in aws_response.message:
                            for item in aws_response.message['content']:
                                if isinstance(item, dict) and 'text' in item:
                                    aws_text += item['text']
                        
                        print(f"   📄 AWS response ({len(aws_text)} chars):")
                        print("   " + "-" * 50)
                        print("   " + aws_text[:500] + ("..." if len(aws_text) > 500 else ""))
                        print("   " + "-" * 50)
                    
                    return model_id
                else:
                    print(f"   ⚠️  Tool doesn't seem to be working properly")
            else:
                print(f"   ⚠️  No message in response")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            print(f"   Error type: {type(e).__name__}")
            
    return None

if __name__ == "__main__":
    working_model = test_strands_agent()
    if working_model:
        print(f"\n✅ Working model found: {working_model}")
    else:
        print(f"\n❌ No working models found")