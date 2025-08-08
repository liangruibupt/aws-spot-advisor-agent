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
            
            # Test simple call
            response = agent("Make a GET request to https://aws.amazon.com/ec2/spot/instance-advisor/ and show me the response")
            
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
                
                if "successful" in response_text.lower():
                    print(f"   ✅ SUCCESS: {model_id} works with Strands Agent!")
                    return model_id
                else:
                    print(f"   ⚠️  Response doesn't contain 'successful': {response_text[:100]}...")
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