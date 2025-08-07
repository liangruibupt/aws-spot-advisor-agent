#!/usr/bin/env python3
"""
Check what Claude models are actually available in your AWS Bedrock account.
"""

import boto3
import json
from botocore.exceptions import ClientError

def list_available_models():
    """List all available models in AWS Bedrock."""
    try:
        # Create Bedrock client
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        
        print("🔍 Checking available models in AWS Bedrock...")
        print("=" * 60)
        
        # List foundation models
        response = bedrock.list_foundation_models()
        
        # Filter for Anthropic Claude models
        claude_models = []
        for model in response.get('modelSummaries', []):
            if 'anthropic' in model.get('modelId', '').lower() and 'claude' in model.get('modelId', '').lower():
                claude_models.append({
                    'modelId': model.get('modelId'),
                    'modelName': model.get('modelName'),
                    'providerName': model.get('providerName'),
                    'inputModalities': model.get('inputModalities', []),
                    'outputModalities': model.get('outputModalities', []),
                    'responseStreamingSupported': model.get('responseStreamingSupported', False)
                })
        
        if claude_models:
            print(f"✅ Found {len(claude_models)} Claude model(s):")
            print()
            
            for i, model in enumerate(claude_models, 1):
                print(f"{i}. Model ID: {model['modelId']}")
                print(f"   Name: {model['modelName']}")
                print(f"   Provider: {model['providerName']}")
                print(f"   Streaming: {model['responseStreamingSupported']}")
                print("-" * 40)
            
            # Recommend the most recent model
            latest_model = claude_models[-1]  # Usually the last one is most recent
            print(f"💡 RECOMMENDED MODEL:")
            print(f"   export BEDROCK_MODEL_ID='{latest_model['modelId']}'")
            
        else:
            print("❌ No Claude models found!")
            print("   This could mean:")
            print("   - No Claude models are enabled in your AWS account")
            print("   - You don't have permission to list models")
            print("   - Claude models aren't available in your region")
        
        return claude_models
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"❌ AWS Error ({error_code}): {e}")
        
        if error_code == 'AccessDeniedException':
            print("   → You don't have permission to list Bedrock models")
            print("   → Try: aws iam attach-user-policy --user-name YOUR_USER --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess")
        elif error_code == 'UnauthorizedOperation':
            print("   → Bedrock service might not be available in your region")
        
        return []
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def test_model_access():
    """Test if we can access Bedrock runtime."""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("\n🔧 Testing Bedrock Runtime access...")
        
        # This will fail but tells us if we have runtime access
        try:
            bedrock_runtime.invoke_model(
                modelId='test-model',
                body=json.dumps({'test': 'test'})
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'ValidationException':
                print("✅ Bedrock Runtime access: OK (invalid model test)")
                return True
            elif error_code == 'AccessDeniedException':
                print("❌ Bedrock Runtime access: DENIED")
                return False
            else:
                print(f"⚠️  Bedrock Runtime access: Unknown ({error_code})")
                return True
                
    except Exception as e:
        print(f"❌ Bedrock Runtime test failed: {e}")
        return False

def main():
    """Main function."""
    print("AWS Bedrock Model Checker")
    print("=" * 60)
    
    # Test runtime access
    runtime_ok = test_model_access()
    
    # List available models
    models = list_available_models()
    
    print("\n📋 SUMMARY:")
    print("=" * 60)
    print(f"Bedrock Runtime Access: {'✅ OK' if runtime_ok else '❌ DENIED'}")
    print(f"Available Claude Models: {len(models)}")
    
    if not models and runtime_ok:
        print("\n💡 NEXT STEPS:")
        print("1. Enable Claude models in AWS Bedrock console")
        print("2. Go to: https://console.aws.amazon.com/bedrock/")
        print("3. Navigate to 'Model access' and request access to Claude models")
        print("4. Wait for approval (usually instant for Claude)")

if __name__ == "__main__":
    main()