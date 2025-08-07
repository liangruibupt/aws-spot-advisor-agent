#!/usr/bin/env python3
"""
Check what Claude models are actually available in your AWS Bedrock account.
"""

import boto3
import json
from botocore.exceptions import ClientError

inference_profiles = []

def list_available_models():
    """List all available models in AWS Bedrock."""
    try:
        # Create Bedrock client
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        
        print("🔍 Checking available models in AWS Bedrock...")
        print("=" * 60)
        
        # List foundation models
        response = bedrock.list_foundation_models(byProvider='Anthropic')
        
        # Filter for Anthropic Claude models
        claude_models = []
        for model in response.get('modelSummaries', []):
            if 'active' in model.get('modelLifecycle', '').get('status', '').lower():
                test_result = test_model_access(model.get('modelId'))
                if test_result == '' or test_result is None:
                    continue
                claude_models.append({
                    'modelId': test_result,
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

def test_model_access(model_id: str):
    """Test if we can access Bedrock runtime."""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        print(f"\n🔧 Testing Bedrock Runtime access {model_id}...")
        
        user_message = "Explain the concept of quantum entanglement in simple terms."
        conversation = [{
            "role": "user",
            "content": [{"text": user_message}],
            }
        ]
        # Tells us if we have runtime access
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
            )
            # Extract and print the response text.
            response_text = response["output"]["message"]["content"][0]["text"]
            print(f"Invoke response {response_text}")
            print(f"✅ Invoke {model_id} OK")
            return model_id

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            # print (f"❌ Bedrock Runtime access: {e.response}")
            if error_code == 'ValidationException':
                # Test with inference profile (if available)
                result = test_model_access_inference_profile(model_id)
                if not result:
                    return None
                print("✅ Bedrock Runtime access: OK")
                return result
            elif error_code == 'ResourceNotFoundException':
                print("❌ Bedrock Runtime access: ResourceNotFoundException")
                return None
            elif error_code == 'AccessDeniedException':
                print("❌ Bedrock Runtime access: DENIED")
                return None
            else:
                print(f"⚠️  Bedrock Runtime access: Unknown ({error_code})")
                return None
                
    except Exception as e:
        print(f"❌ Bedrock Runtime test failed: {e}")
        return None

def test_model_access_inference_profile(model_id: str):
    """Test if we can access Bedrock runtime."""
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("\n🔧 Testing Bedrock Runtime access with inference profile...")
        
        user_message = "Explain the concept of quantum entanglement in simple terms."
        conversation = [{
            "role": "user",
            "content": [{"text": user_message}],
            }
        ]
        
        for profile in inference_profiles:
            if model_id in profile.get('inferenceProfileId'):
                inference_profile_id = profile.get('inferenceProfileId')
                print(f"   Using inference profile: {inference_profile_id} for {model_id}")
        
                try:
                    response = bedrock_runtime.converse(
                        modelId=inference_profile_id,
                        messages=conversation,
                        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
                    )
                    # Extract and print the response text.
                    response_text = response["output"]["message"]["content"][0]["text"]
                    print(f"Invoke response {response_text}")
                    print(f"✅ Invoke {inference_profile_id} OK")
                    return inference_profile_id

                except ClientError as e:
                    error_msg = e.response.get('Error', {})
                    print (f"❌ Bedrock Runtime access with inference profile: {error_msg}")
                    return None
                
    except Exception as e:
        print(f"❌ Bedrock Runtime test failed: {e}")
        return None
    
def get_inference_profiles():
    """Get available Bedrock inference profiles."""
    try:
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        
        print("\n🔍 Checking available inference profiles...")
        print("=" * 60)
        
        response = bedrock.list_inference_profiles(typeEquals='SYSTEM_DEFINED')
        profiles = response.get('inferenceProfileSummaries', [])
        
        if profiles:
            print(f"✅ Found {len(profiles)} inference profile(s):")
            print()
            
            for i, profile in enumerate(profiles, 1):
                print(f"{i}. Profile Name: {profile.get('inferenceProfileName')}")
                print(f"   Profile ID: {profile.get('inferenceProfileId')}")
                if 'models' in profile:
                    print(f"   Models: {', '.join([m.get('modelId', 'Unknown') for m in profile.get('models', [])])}")
                print("-" * 40)
                
        else:
            print("❌ No inference profiles found!")
            
        return profiles
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"❌ AWS Error getting inference profiles ({error_code}): {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error getting inference profiles: {e}")
        return []

def main():
    """Main function."""
    print("AWS Bedrock Model Checker")
    print("=" * 60)
    
    # List available models
    models = list_available_models()
    
    print("\n📋 SUMMARY:")
    print("=" * 60)
    print(f"Available Claude Models: {len(models)}")
    
    if not models:
        print("\n💡 NEXT STEPS:")
        print("1. Enable Claude models in AWS Bedrock console")
        print("2. Go to: https://console.aws.amazon.com/bedrock/")
        print("3. Navigate to 'Model access' and request access to Claude models")
        print("4. Wait for approval (usually instant for Claude)")

if __name__ == "__main__":
    inference_profiles = get_inference_profiles()
    main()