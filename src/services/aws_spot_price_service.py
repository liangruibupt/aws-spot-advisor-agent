"""
AWS Spot Price Service for direct API access to spot pricing data.

This service provides direct access to AWS EC2 spot pricing data using the
AWS API instead of web scraping, which is more reliable and efficient.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from src.models.spot_data import RawSpotData
from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError,
    DataValidationError
)
from src.utils.retry_utils import aws_service_retry
from src.utils.config import load_config


logger = logging.getLogger(__name__)


class AwsSpotPriceServiceError(SpotAnalyzerBaseError):
    """Base exception for AWS Spot Price Service errors."""
    pass


class AwsSpotPriceService:
    """
    Service for retrieving AWS EC2 spot pricing data directly from AWS API.
    
    This service provides a more reliable alternative to web scraping by
    using the AWS EC2 API directly to get spot price history.
    """

    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize the AWS Spot Price Service.
        
        Args:
            region_name: AWS region for EC2 service (uses config default if None)
        """
        # Load configuration
        config = load_config()
        
        self.region_name = region_name or config.get('aws_default_region', 'us-east-1')
        
        # Initialize EC2 client
        self._ec2_client = None
        
        logger.info(f"AwsSpotPriceService initialized for region: {self.region_name}")

    def _get_ec2_client(self):
        """Get or create EC2 client."""
        if self._ec2_client is None:
            try:
                self._ec2_client = boto3.client('ec2', region_name=self.region_name)
                logger.debug(f"EC2 client created for region: {self.region_name}")
            except Exception as e:
                logger.error(f"Failed to create EC2 client: {e}")
                raise AwsSpotPriceServiceError(
                    message=f"Failed to create EC2 client: {e}",
                    region=self.region_name,
                    original_error=e
                )
        return self._ec2_client

    @aws_service_retry(max_attempts=3)
    def get_spot_prices(
        self, 
        instance_types: List[str], 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[RawSpotData]:
        """
        Get spot price history from AWS API.
        
        Args:
            instance_types: List of instance types to get prices for
            start_time: Start time for price history (defaults to 7 days ago)
            end_time: End time for price history (defaults to now)
            max_results: Maximum number of results to return
            
        Returns:
            List of RawSpotData objects
            
        Raises:
            AwsSpotPriceServiceError: If API call fails
            DataValidationError: If input validation fails
        """
        if not instance_types:
            raise DataValidationError(
                message="Instance types list cannot be empty",
                field_name="instance_types",
                field_value=instance_types
            )
        
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(days=7)
        
        logger.info(
            f"Getting spot prices for {instance_types} from {start_time.strftime('%Y-%m-%d')} "
            f"to {end_time.strftime('%Y-%m-%d')}"
        )
        
        try:
            ec2_client = self._get_ec2_client()
            
            # Prepare API parameters
            params = {
                'InstanceTypes': instance_types,
                'ProductDescriptions': ['Linux/UNIX'],
                'StartTime': start_time,
                'EndTime': end_time,
                'MaxResults': max_results,
                'DryRun': False
            }
            
            logger.debug(f"API Parameters: {params}")
            
            # Call AWS API
            response = ec2_client.describe_spot_price_history(**params)
            spot_price_history = response.get('SpotPriceHistory', [])
            
            logger.info(f"Retrieved {len(spot_price_history)} spot price records")
            
            # Convert to RawSpotData objects
            raw_data = self._convert_to_raw_spot_data(spot_price_history)
            
            logger.info(f"Converted to {len(raw_data)} RawSpotData objects")
            return raw_data
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            
            logger.error(f"AWS client error getting spot prices: {e}")
            
            # Handle specific AWS error types
            if error_code in ['Throttling', 'ThrottlingException', 'TooManyRequestsException']:
                raise RateLimitError(
                    message=f"Rate limit exceeded for EC2 service: {e}",
                    service_name="AWS EC2",
                    original_error=e
                )
            elif error_code in ['ServiceUnavailable', 'InternalError']:
                raise ServiceUnavailableError(
                    message=f"EC2 service unavailable: {e}",
                    service_name="AWS EC2",
                    status_code=status_code,
                    original_error=e
                )
            else:
                raise AwsSpotPriceServiceError(
                    message=f"AWS API error: {e}",
                    service_error_code=error_code,
                    region=self.region_name,
                    original_error=e
                )
                
        except BotoCoreError as e:
            logger.error(f"Boto core error getting spot prices: {e}")
            raise NetworkError(
                message=f"Network error during API call: {e}",
                original_error=e
            )
            
        except Exception as e:
            logger.error(f"Unexpected error getting spot prices: {e}")
            raise AwsSpotPriceServiceError(
                message=f"Failed to get spot prices: {e}",
                region=self.region_name,
                original_error=e
            )

    def _convert_to_raw_spot_data(self, spot_price_history: List[Dict[str, Any]]) -> List[RawSpotData]:
        """
        Convert AWS API response to RawSpotData objects.
        
        Args:
            spot_price_history: List of spot price records from AWS API
            
        Returns:
            List of RawSpotData objects
        """
        raw_data = []
        
        for record in spot_price_history:
            try:
                # Extract region from availability zone (remove last character)
                availability_zone = record.get('AvailabilityZone', '')
                region = availability_zone[:-1] if availability_zone else self.region_name
                
                # Create RawSpotData object
                raw_spot_data = RawSpotData(
                    region=region,
                    instance_type=record.get('InstanceType', ''),
                    spot_price=float(record.get('SpotPrice', 0.0)),
                    currency='USD',  # AWS spot prices are always in USD
                    interruption_rate=0.03,  # Default 3% - conservative estimate for most instances
                    timestamp=record.get('Timestamp', datetime.now(timezone.utc)),
                    availability=True  # Assume available if in spot price history
                )
                
                raw_data.append(raw_spot_data)
                
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Error converting spot price record {record}: {e}")
                continue
        
        return raw_data

    def get_latest_spot_prices(self, instance_types: List[str]) -> Dict[str, RawSpotData]:
        """
        Get the latest spot price for each instance type.
        
        Args:
            instance_types: List of instance types to get prices for
            
        Returns:
            Dictionary mapping instance type to latest RawSpotData
        """
        # Get recent spot prices (last 24 hours)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        all_prices = self.get_spot_prices(
            instance_types=instance_types,
            start_time=start_time,
            end_time=end_time,
            max_results=200
        )
        
        # Group by instance type and get latest for each
        latest_prices = {}
        for price in all_prices:
            instance_type = price.instance_type
            if (instance_type not in latest_prices or 
                price.timestamp > latest_prices[instance_type].timestamp):
                latest_prices[instance_type] = price
        
        logger.info(f"Found latest prices for {len(latest_prices)} instance types")
        return latest_prices

    def check_instance_types_exist(self, instance_types: List[str]) -> Dict[str, bool]:
        """
        Check if instance types exist in the current region.
        
        Args:
            instance_types: List of instance types to check
            
        Returns:
            Dictionary mapping instance type to existence boolean
        """
        try:
            ec2_client = self._get_ec2_client()
            
            response = ec2_client.describe_instance_types(InstanceTypes=instance_types)
            existing_types = {it['InstanceType'] for it in response.get('InstanceTypes', [])}
            
            result = {instance_type: instance_type in existing_types 
                     for instance_type in instance_types}
            
            logger.info(f"Instance type existence check: {result}")
            return result
            
        except ClientError as e:
            logger.warning(f"Error checking instance types: {e}")
            # Return False for all if check fails
            return {instance_type: False for instance_type in instance_types}

    def get_supported_instance_types(self) -> List[str]:
        """
        Get list of supported instance types for spot pricing.
        
        Note: This returns a basic list. In practice, most EC2 instance types
        support spot pricing, but some newer or specialized types may not.
        
        Returns:
            List of commonly supported instance type patterns
        """
        # Return the instance types we know work
        return [
            "p5en.48xlarge",
            "p5.48xlarge",
            "p4d.24xlarge",
            "p3.2xlarge",
            "p3.8xlarge",
            "p3.16xlarge",
            "g4dn.xlarge",
            "g4dn.2xlarge",
            "g4dn.4xlarge",
            "g4dn.8xlarge",
            "g4dn.12xlarge",
            "g4dn.16xlarge",
            "m5.large",
            "m5.xlarge",
            "m5.2xlarge",
            "m5.4xlarge",
            "c5.large",
            "c5.xlarge",
            "c5.2xlarge",
            "c5.4xlarge"
        ]

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the service configuration.
        
        Returns:
            Dictionary containing service information
        """
        return {
            "service_name": "AwsSpotPriceService",
            "region": self.region_name,
            "data_source": "AWS EC2 API",
            "api_method": "describe_spot_price_history",
            "supports_real_time": True,
            "supports_historical": True
        }