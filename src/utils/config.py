"""Configuration management utilities."""

import os
from typing import Dict, Any

from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dict containing configuration values
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    config = {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'aws_default_region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'cache_ttl_hours': int(os.getenv('CACHE_TTL_HOURS', '1')),
        'max_interruption_rate': float(os.getenv('MAX_INTERRUPTION_RATE', '5.0')),
        'top_regions_count': int(os.getenv('TOP_REGIONS_COUNT', '3')),
        'spot_advisor_url': os.getenv(
            'SPOT_ADVISOR_URL', 
            'https://aws.amazon.com/ec2/spot/instance-advisor/'
        ),
    }
    
    return config