#!/usr/bin/env python3
"""
Test AWS EC2 API for spot pricing data.
"""

import boto3
from datetime import datetime, timedelta, timezone
import json

def get_spot_prices(ec2_client, instance_types=None, max_results=50, start_time=None, end_time=None):
    """Get spot price history from AWS API."""
    params = {'MaxResults': max_results}
    
    if instance_types:
        params['InstanceTypes'] = instance_types
    if start_time:
        params['StartTime'] = start_time
    if end_time:
        params['EndTime'] = end_time
    
    print(params)
    response = ec2_client.describe_spot_price_history(**params)
    return response.get('SpotPrices', [])

def check_instance_type(ec2_client, instance_types=None, max_results=20):
    print("🔍 Checking if specified instance types exist...")
    try:
        if instance_types:
            response = ec2_client.describe_instance_types(InstanceTypes=instance_types)
        else:
            response = ec2_client.describe_instance_types(MaxResults=max_results)
        existing_types = [it['InstanceType'] for it in response.get('InstanceTypes', [])]
            
        if not existing_types:
            print("⚠️  None of the specified instance types exist!")
            return False
        else:
            print(f"✅ Existing instance types: {existing_types}")
            return True
                
    except Exception as e:
        print(f"⚠️  Error checking instance types: {e}")
        print("   Instance types might not exist or not be available in this region")

def test_aws_spot_pricing_api():
    """Test AWS EC2 API for spot pricing data."""
    
    print("🔍 Testing AWS EC2 API for spot pricing data...")
    
    try:
        # Create EC2 client
        ec2 = boto3.client('ec2', region_name='us-east-1')
        
        # Target instance types (as specified by user)
        target_instances = ['p5en.48xlarge', 'p5.48xlarge']
        print(f"🎯 Target instance types: {target_instances}")
        
        # Check if these specific instance types exist
        check_instance_type(ec2, target_instances)
        
        # Define time range for spot price history
        end_time = datetime(2025, 8, 8)  # Current date
        start_time = datetime(2024, 1, 1)  # Start from beginning of 2024 to catch any historical data
        
        print(f"🕐 Checking spot prices from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Check what P5 instances are available in spot pricing
        all_prices = get_spot_prices(ec2, instance_types=target_instances, max_results=20, start_time=start_time, end_time=end_time)
        
        p5_instances = set()
        for price in all_prices:
            if price['InstanceType'].startswith('p5'):
                p5_instances.add(price['InstanceType'])
        
        print(f"📊 P5 instances with spot pricing: {sorted(p5_instances) if p5_instances else 'None'}")
        
        
        # Try checking other regions
        print("\n🔍 Checking other regions...")
        regions_to_check = ['us-west-2', 'eu-west-1', 'ap-southeast-1']
            
        for region in regions_to_check:
            try:
                regional_ec2 = boto3.client('ec2', region_name=region)
                check_instance_type(regional_ec2, target_instances)
                regional_prices = get_spot_prices(
                    regional_ec2, 
                    instance_types=target_instances, 
                    start_time=start_time, 
                    end_time=end_time, 
                    max_results=20
                )
                if regional_prices:
                    print(f"  ✅ {region}: Found {sorted(p5_instances)}")
                    break
                else:
                    print(f"  ❌ {region}: No records")
            except Exception as e:
                print(f"  ❌ {region}: Error - {e}")
        
        return price_data
        
    except Exception as e:
        print(f"❌ Error testing AWS spot pricing API: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    data = test_aws_spot_pricing_api()
    if data:
        print(f"\n✅ Successfully retrieved spot pricing data for {len(data)} region/instance combinations")
        
        # Convert to the format expected by our application
        regions_data = []
        for key, price_info in data.items():
            regions_data.append({
                "region": price_info['region'],
                "instance_type": price_info['instance_type'],
                "spot_price": price_info['spot_price'],
                "currency": price_info['currency'],
                "interruption_rate": 0.05,  # Default - would need to be scraped separately
                "availability": True
            })
        
        result = {"regions": regions_data}
        print(f"\n📄 JSON format:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Failed to retrieve spot pricing data")