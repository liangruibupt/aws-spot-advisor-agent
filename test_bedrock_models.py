#!/usr/bin/env python3
"""
Test script to find working Claude model identifiers in AWS Bedrock.
Includes support for inference profiles for cross-region inference.
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError
from src.services.bedrock_agent_service import BedrockAgentService
from src.utils.config import load_config


# Load configuration
config = load_config()
        
aws_region = config.get('bedrock_region', 'us-east-1')
model_name = config.get('bedrock_model_id', 'anthropic.claude-sonnet-4-20250514-v1:0')
        
# Fallback models to try if the primary model fails (based on available models)
fallback_model = config.get('bedrock_fallback_model_id', 'anthropic.claude-3-7-sonnet-20250219-v1:0')
        
# List of Claude model identifiers to test (inference profiles that should work)
CLAUDE_MODELS_TO_TEST = [
    model_name,  # Primary model from config
    fallback_model,  # Fallback model from config
    
    # Inference profiles (these should work)
    'us.anthropic.claude-3-7-sonnet-20250219-v1:0',     # Claude 3.7 Sonnet
    'us.anthropic.claude-opus-4-1-20250805-v1:0',       # Claude Opus 4.1 (latest)
    'us.anthropic.claude-opus-4-20250514-v1:0',         # Claude Opus 4
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0',     # Claude 3.5 Sonnet v2
    'us.anthropic.claude-3-5-sonnet-20240620-v1:0',     # Claude 3.5 Sonnet
    'us.anthropic.claude-3-sonnet-20240229-v1:0',       # Claude 3 Sonnet (stable)
]

# Global variable to store inference profiles
inference_profiles = []


def get_inference_profiles():
    """Get available Bedrock inference profiles."""
    global inference_profiles
    try:
        bedrock = boto3.client('bedrock', region_name=aws_region)
        
        print("🔍 Getting available inference profiles...")
        
        response = bedrock.list_inference_profiles(typeEquals='SYSTEM_DEFINED')
        inference_profiles = response.get('inferenceProfileSummaries', [])
        
        if inference_profiles:
            print(f"   Found {len(inference_profiles)} inference profile(s)")
            for profile in inference_profiles:
                print(f"   - {profile.get('inferenceProfileId')}")
        else:
            print("   No inference profiles found")
            
        return inference_profiles
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"   ❌ Error getting inference profiles ({error_code}): {e}")
        return []
    except Exception as e:
        print(f"   ❌ Unexpected error getting inference profiles: {e}")
        return []


def test_model_with_bedrock_runtime(model_id: str, aws_region: str) -> str:
    """
    Test model access using Bedrock Runtime directly.
    Returns the working model ID (could be original or inference profile).
    """
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
        
        user_message = "Hello, can you respond with 'Connection successful'?"
        conversation = [{
            "role": "user",
            "content": [{"text": user_message}],
        }]
        
        # First try with original model ID
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={"maxTokens": 50, "temperature": 0.5, "topP": 0.9},
            )
            
            # Extract response text
            response_text = response["output"]["message"]["content"][0]["text"]
            if "successful" in response_text.lower():
                print(f"   ✅ Direct model access successful")
                return model_id
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            print(f"   ⚠️  Direct model access failed ({error_code})")
            
            # If ValidationException, try with inference profile
            if error_code == 'ValidationException':
                print(f"   🔄 Trying with inference profiles...")
                
                # Look for matching inference profile
                for profile in inference_profiles:
                    profile_id = profile.get('inferenceProfileId', '')
                    if model_id in profile_id or any(model_id in str(model.get('modelId', '')) for model in profile.get('models', [])):
                        try:
                            print(f"   🧪 Testing inference profile: {profile_id}")
                            response = bedrock_runtime.converse(
                                modelId=profile_id,
                                messages=conversation,
                                inferenceConfig={"maxTokens": 50, "temperature": 0.5, "topP": 0.9},
                            )
                            
                            response_text = response["output"]["message"]["content"][0]["text"]
                            if "successful" in response_text.lower():
                                print(f"   ✅ Inference profile access successful")
                                return profile_id
                                
                        except ClientError as profile_error:
                            profile_error_code = profile_error.response.get('Error', {}).get('Code', 'Unknown')
                            print(f"   ❌ Inference profile failed ({profile_error_code})")
                            continue
            
            # If no inference profile worked, return None
            return None
            
    except Exception as e:
        print(f"   ❌ Runtime test error: {e}")
        return None


def test_model(model_id: str, aws_region: str) -> bool:
    """Test if a model identifier works with detailed debugging."""
    try:
        print(f"🧪 Testing model: {model_id}")
        print(f"   Region: {aws_region}")
        
        # First test with Bedrock Runtime directly
        working_model_id = test_model_with_bedrock_runtime(model_id, aws_region)
        
        if not working_model_id:
            print(f"❌ FAILED: {model_id} - no working model ID found")
            return False
        
        # Now test with BedrockAgentService using the working model ID
        print(f"   🔧 Testing with BedrockAgentService using: {working_model_id}")
        
        # Create service with the working model ID
        service = BedrockAgentService(model_name=working_model_id, region_name=aws_region)
        print(f"   Service created with model: {service.model_name}")
        
        # Get agent info
        agent_info = service.get_agent_info()
        print(f"   Agent info: {agent_info}")
        
        # Test connection
        print("   Testing connection...")
        success = service.test_connection()
        
        if success:
            print(f"✅ SUCCESS: {model_id} works with model ID: {working_model_id}")
            return True
        else:
            print(f"❌ FAILED: {model_id} - BedrockAgentService connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {model_id}")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)[:200]}...")
        
        # Check if it's a specific AWS error
        if hasattr(e, 'response'):
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            print(f"   AWS Error Code: {error_code}")
        
        return False


def main():
    """Test all model identifiers and find working ones."""
    print("🔍 Testing Claude model identifiers in AWS Bedrock...")
    print("=" * 60)
    
    # First get inference profiles
    get_inference_profiles()
    print("-" * 40)
    
    working_models = []
    
    for model_id in CLAUDE_MODELS_TO_TEST:
        if test_model(model_id, aws_region):
            working_models.append(model_id)
        print("-" * 40)
    
    print("\n📊 RESULTS:")
    print("=" * 60)
    
    if working_models:
        print(f"✅ Found {len(working_models)} working model(s):")
        for i, model in enumerate(working_models, 1):
            print(f"  {i}. {model}")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use this model ID: {working_models[0]}")
        print(f"   Set environment variable: export BEDROCK_MODEL_ID='{working_models[0]}'")
        
    else:
        print("❌ No working models found!")
        print("   This could be due to:")
        print("   - AWS credentials not configured")
        print("   - AWS Bedrock not enabled in your region")
        print("   - Model access not granted in your AWS account")
        print("   - Network connectivity issues")
    
    return 0 if working_models else 1


if __name__ == "__main__":
    sys.exit(main())