#!/usr/bin/env python3
"""
Test AWS EC2 API for spot pricing data.
"""

import boto3
from datetime import datetime, timedelta, timezone
import json

def get_spot_prices(ec2_client, instance_types=None, max_results=50, start_time=None, end_time=None):
    """Get spot price history from AWS API."""
    params = {
        'MaxResults': max_results, 
        'DryRun': False,
        'ProductDescriptions': ['Linux/UNIX']
    }
    
    if instance_types:
        params['InstanceTypes'] = instance_types
    if start_time:
        params['StartTime'] = start_time
    if end_time:
        params['EndTime'] = end_time
    
    print("API Parameters:", params)
    response = ec2_client.describe_spot_price_history(**params)
    return response.get('SpotPriceHistory', [])

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
        end_time = datetime(2025, 8, 7)  # Current date
        start_time = datetime(2025, 8, 1)  # Start from August 1st
        
        print(f"🕐 Checking spot prices from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # # Direct API call for debugging
        # print("🔍 Making direct API call...")
        # response = ec2.describe_spot_price_history(
        #     DryRun=False,
        #     StartTime=start_time, 
        #     EndTime=end_time,
        #     InstanceTypes=target_instances,
        #     ProductDescriptions=['Linux/UNIX'],
        #     MaxResults=20
        # )
        # direct_prices = response.get('SpotPriceHistory', [])
        # print(f"📊 Direct API call found {len(direct_prices)} spot price records")
        
        # Get spot prices using helper function
        print("🔍 Getting spot prices...")
        all_prices = get_spot_prices(ec2, instance_types=target_instances, max_results=20, start_time=start_time, end_time=end_time)
        print(f"📊 Found {len(all_prices)} spot price records")
        
        # Use the helper function results
        prices_to_use = all_prices
        
        if prices_to_use:
            print("✅ Found spot price data!")
            for i, price in enumerate(prices_to_use[:3]):
                print(f"  {i+1}. {price['AvailabilityZone']} - {price['InstanceType']}: ${price['SpotPrice']} at {price['Timestamp']}")
        else:
            print("⚠️  No spot price data found")
        
        
        # Initialize price_data dictionary
        price_data = {}
        
        # Process the found prices
        if prices_to_use:
            print("\n📋 Processing spot price data...")
            for price in prices_to_use:
                region = price['AvailabilityZone'][:-1]  # Remove AZ suffix to get region
                instance_type = price['InstanceType']
                spot_price = float(price['SpotPrice'])
                timestamp = price['Timestamp']
                
                key = f"{region}-{instance_type}"
                if key not in price_data or price_data[key]['timestamp'] < timestamp:
                    price_data[key] = {
                        'region': region,
                        'instance_type': instance_type,
                        'spot_price': spot_price,
                        'currency': 'USD',
                        'timestamp': timestamp,
                        'availability_zone': price['AvailabilityZone']
                    }
            
            print(f"📊 Processed {len(price_data)} unique region/instance combinations:")
            for key, data in sorted(price_data.items()):
                print(f"  {data['region']} - {data['instance_type']}: ${data['spot_price']:.4f}/hour")
        else:
            print("\n🔍 No data found in us-east-1, checking other regions...")
            regions_to_check = ['us-west-2', 'eu-west-1', 'ap-southeast-1']
            
            for region in regions_to_check:
                try:
                    print(f"\n🔍 Checking {region}...")
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
                        print(f"  ✅ {region}: Found {len(regional_prices)} records")
                        # Show sample data
                        for price in regional_prices[:2]:
                            print(f"    {price['AvailabilityZone']} - {price['InstanceType']}: ${price['SpotPrice']}")
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