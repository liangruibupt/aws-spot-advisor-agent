#!/usr/bin/env python3
"""
Test script to find working Claude model identifiers in AWS Bedrock.
"""

import os
import sys
from src.services.bedrock_agent_service import BedrockAgentService
from src.utils.config import load_config


# Load configuration
config = load_config()
        
aws_region = config.get('bedrock_region', 'us-east-1')
model_name = config.get('bedrock_model_id', 'anthropic.claude-sonnet-4-20250514-v1:0')
        
# Fallback models to try if the primary model fails (based on available models)
fallback_model = config.get('bedrock_fallback_model_id', 'anthropic.claude-3-7-sonnet-20250219-v1:0')
        
# List of Claude model identifiers to test (from most recent to older)
CLAUDE_MODELS_TO_TEST = [
    model_name,
    fallback_model
]

def test_model(model_id: str, aws_region: str) -> bool:
    """Test if a model identifier works."""
    try:
        print(f"Testing model: {model_id}")
        service = BedrockAgentService(model_name=model_id, region_name=aws_region)
        
        # Test connection
        success = service.test_connection()
        if success:
            print(f"✅ SUCCESS: {model_id} works!")
            return True
        else:
            print(f"❌ FAILED: {model_id} - connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {model_id} - {str(e)[:100]}...")
        return False

def main():
    """Test all model identifiers and find working ones."""
    print("🔍 Testing Claude model identifiers in AWS Bedrock...")
    print("=" * 60)
    
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