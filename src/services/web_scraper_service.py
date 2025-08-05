"""
Web scraping service for AWS Spot Instance Advisor.

This module provides web scraping functionality using the BedrockAgentService
to fetch and parse spot pricing data from the AWS EC2 Spot Instance Advisor.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from src.services.bedrock_agent_service import BedrockAgentService, BedrockAgentServiceError
from src.models.spot_data import RawSpotData
from src.utils.exceptions import (
    WebScrapingError,
    BedrockServiceError,
    DataValidationError,
    CacheError,
    NetworkError,
    ServiceUnavailableError
)
from src.utils.retry_utils import web_scraping_retry


logger = logging.getLogger(__name__)


# Keep the old exception for backward compatibility
class WebScraperServiceError(WebScrapingError):
    """Legacy exception for WebScraperService errors."""
    pass


class WebScraperService:
    """
    Service for scraping AWS Spot Instance Advisor data.
    
    This service uses BedrockAgentService to fetch and parse spot pricing data
    from the AWS EC2 Spot Instance Advisor website with caching and freshness validation.
    """

    # AWS Spot Instance Advisor URL
    SPOT_ADVISOR_URL = "https://aws.amazon.com/ec2/spot/instance-advisor/"
    
    # Cache TTL in seconds (1 hour)
    CACHE_TTL_SECONDS = 3600

    def __init__(
        self,
        bedrock_service: Optional[BedrockAgentService] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS
    ):
        """
        Initialize the Web Scraper Service.
        
        Args:
            bedrock_service: BedrockAgentService instance (optional, will create if None)
            cache_ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
        """
        self.bedrock_service = bedrock_service or BedrockAgentService()
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # In-memory cache for scraped data
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Supported instance types
        self.supported_instance_types = ["p5en.48xlarge", "p5.48xlarge"]

    @web_scraping_retry(max_attempts=3)
    def scrape_spot_data(
        self,
        instance_types: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> List[RawSpotData]:
        """
        Scrape spot pricing data from AWS Spot Instance Advisor.
        
        Args:
            instance_types: List of instance types to scrape (defaults to supported types)
            force_refresh: Force refresh of cached data
            
        Returns:
            List of RawSpotData objects containing spot pricing information
            
        Raises:
            WebScrapingError: If scraping fails
            DataValidationError: If input validation fails
            CacheError: If cache operations fail
        """
        if instance_types is None:
            instance_types = self.supported_instance_types.copy()
        
        # Validate instance types
        if not isinstance(instance_types, list):
            raise DataValidationError(
                message="Instance types must be a list",
                field_name="instance_types",
                field_value=instance_types,
                validation_rule="must be list"
            )
        
        invalid_types = [t for t in instance_types if t not in self.supported_instance_types]
        if invalid_types:
            logger.warning(f"Unsupported instance types requested: {invalid_types}")
            instance_types = [t for t in instance_types if t in self.supported_instance_types]
        
        if not instance_types:
            raise DataValidationError(
                message="No valid instance types specified",
                field_name="instance_types",
                field_value=instance_types,
                validation_rule="must contain supported instance types"
            )
        
        try:
            logger.info(f"Starting spot data scraping for instance types: {instance_types}")
            
            # Check cache first (unless force refresh)
            cache_key = self._get_cache_key(instance_types)
            if not force_refresh and self._is_cache_valid(cache_key):
                logger.info("Returning cached spot data")
                try:
                    cached_data: List[RawSpotData] = self._cache[cache_key]["data"]
                    return cached_data
                except KeyError as e:
                    logger.warning(f"Cache key error: {e}")
                    raise CacheError(
                        message=f"Failed to retrieve cached data: {e}",
                        cache_key=cache_key,
                        operation="retrieve",
                        original_error=e
                    )
            
            # Scrape fresh data
            raw_content = self._fetch_web_content()
            spot_data = self._parse_spot_data(raw_content, instance_types)
            
            # Update cache
            self._update_cache(cache_key, spot_data)
            
            logger.info(f"Successfully scraped {len(spot_data)} spot price records")
            return spot_data
            
        except (DataValidationError, CacheError):
            # Re-raise validation and cache errors as-is
            raise
        except (BedrockAgentServiceError, BedrockServiceError) as e:
            logger.error(f"Bedrock service error during scraping: {e}")
            raise WebScrapingError(
                message=f"Failed to scrape spot data: {e}",
                url=self.SPOT_ADVISOR_URL,
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unexpected error during scraping: {e}")
            raise WebScrapingError(
                message=f"Scraping failed: {e}",
                url=self.SPOT_ADVISOR_URL,
                original_error=e
            )

    def is_data_fresh(self, timestamp: datetime, max_age_hours: float = 1.0) -> bool:
        """
        Check if data timestamp is within acceptable freshness threshold.
        
        Args:
            timestamp: Data timestamp to check
            max_age_hours: Maximum age in hours (default: 1.0)
            
        Returns:
            True if data is fresh, False otherwise
        """
        if not timestamp:
            return False
        
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        current_time = datetime.now(timezone.utc)
        max_age = timedelta(hours=max_age_hours)
        
        age = current_time - timestamp
        is_fresh = age <= max_age
        
        logger.debug(f"Data age: {age}, max age: {max_age}, fresh: {is_fresh}")
        return is_fresh

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary containing cache information
        """
        entries: List[Dict[str, Any]] = []
        cache_info = {
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "entries": entries
        }
        
        current_time = datetime.now(timezone.utc)
        
        for cache_key, cache_entry in self._cache.items():
            entry_info = {
                "key": cache_key,
                "timestamp": cache_entry["timestamp"].isoformat(),
                "data_count": len(cache_entry["data"]),
                "age_seconds": (current_time - cache_entry["timestamp"]).total_seconds(),
                "is_valid": self._is_cache_valid(cache_key)
            }
            entries.append(entry_info)
        
        return cache_info

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache cleared")

    def _fetch_web_content(self) -> str:
        """
        Fetch web content from AWS Spot Instance Advisor.
        
        Returns:
            Raw web content as string
            
        Raises:
            WebScrapingError: If fetching fails
            NetworkError: If network issues occur
            ServiceUnavailableError: If service is unavailable
        """
        try:
            logger.info(f"Fetching content from {self.SPOT_ADVISOR_URL}")
            
            # Use BedrockAgentService to scrape the content
            content = self.bedrock_service.execute_web_scraping(self.SPOT_ADVISOR_URL)
            
            if not content:
                raise WebScrapingError(
                    message="No content returned from web scraping",
                    url=self.SPOT_ADVISOR_URL
                )
            
            logger.debug(f"Fetched {len(content)} characters of content")
            return content
            
        except (NetworkError, ServiceUnavailableError):
            # Re-raise network and service errors as-is
            raise
        except (BedrockAgentServiceError, BedrockServiceError) as e:
            logger.error(f"Failed to fetch web content: {e}")
            raise WebScrapingError(
                message=f"Web content fetch failed: {e}",
                url=self.SPOT_ADVISOR_URL,
                original_error=e
            )

    def _parse_spot_data(self, content: str, instance_types: List[str]) -> List[RawSpotData]:
        """
        Parse spot data from web content.
        
        Args:
            content: Raw web content
            instance_types: List of instance types to filter for
            
        Returns:
            List of RawSpotData objects
            
        Raises:
            WebScrapingError: If parsing fails
            DataValidationError: If data validation fails
        """
        try:
            logger.info("Parsing spot data from web content")
            
            # Use BedrockAgentService to parse the content
            spot_data = self.bedrock_service.parse_spot_data(content, instance_types)
            
            if not spot_data:
                logger.warning("No spot data found in web content")
                return []
            
            # Validate and filter the data
            valid_data = []
            invalid_count = 0
            
            for data in spot_data:
                try:
                    if self._validate_spot_data(data):
                        valid_data.append(data)
                    else:
                        invalid_count += 1
                        logger.warning(f"Invalid spot data filtered out: {data.region}")
                except Exception as e:
                    invalid_count += 1
                    logger.warning(f"Error validating spot data for {getattr(data, 'region', 'unknown')}: {e}")
            
            if invalid_count > 0:
                logger.info(f"Filtered out {invalid_count} invalid records")
            
            logger.info(f"Parsed {len(valid_data)} valid spot price records")
            return valid_data
            
        except DataValidationError:
            # Re-raise validation errors as-is
            raise
        except (BedrockAgentServiceError, BedrockServiceError) as e:
            logger.error(f"Failed to parse spot data: {e}")
            raise WebScrapingError(
                message=f"Spot data parsing failed: {e}",
                url=self.SPOT_ADVISOR_URL,
                original_error=e
            )

    def _validate_spot_data(self, data: RawSpotData) -> bool:
        """
        Validate a single spot data record.
        
        Args:
            data: RawSpotData object to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check required fields
            if not data.region or not data.instance_type:
                return False
            
            # Check price is positive
            if data.spot_price <= 0:
                return False
            
            # Check interruption rate is valid (0-1)
            if not (0 <= data.interruption_rate <= 1):
                return False
            
            # Check currency is USD
            if data.currency != "USD":
                return False
            
            # Check timestamp is recent (within last 24 hours)
            if not self.is_data_fresh(data.timestamp, max_age_hours=24.0):
                logger.warning(f"Spot data timestamp is too old: {data.timestamp}")
                # Don't reject old data, just warn
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating spot data: {e}")
            return False

    def _get_cache_key(self, instance_types: List[str]) -> str:
        """
        Generate cache key for instance types.
        
        Args:
            instance_types: List of instance types
            
        Returns:
            Cache key string
        """
        # Sort instance types for consistent cache keys
        sorted_types = sorted(instance_types)
        return "|".join(sorted_types)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self._cache:
            return False
        
        cache_entry = self._cache[cache_key]
        cache_timestamp: datetime = cache_entry["timestamp"]
        cache_age = datetime.now(timezone.utc) - cache_timestamp
        
        return cache_age.total_seconds() < self.cache_ttl_seconds

    def _update_cache(self, cache_key: str, data: List[RawSpotData]) -> None:
        """
        Update cache with new data.
        
        Args:
            cache_key: Cache key
            data: Spot data to cache
            
        Raises:
            CacheError: If cache update fails
        """
        try:
            self._cache[cache_key] = {
                "timestamp": datetime.now(timezone.utc),
                "data": data
            }
            
            logger.debug(f"Updated cache for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to update cache for key {cache_key}: {e}")
            raise CacheError(
                message=f"Failed to update cache: {e}",
                cache_key=cache_key,
                operation="update",
                original_error=e
            )

    def get_supported_instance_types(self) -> List[str]:
        """
        Get list of supported instance types.
        
        Returns:
            List of supported instance type strings
        """
        return self.supported_instance_types.copy()

    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is accessible and appropriate for scraping.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            
            # Check basic URL structure
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check if it's the expected AWS domain
            if "aws.amazon.com" not in parsed.netloc:
                logger.warning(f"URL is not from AWS domain: {url}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return False

    def get_last_scrape_time(self, instance_types: Optional[List[str]] = None) -> Optional[datetime]:
        """
        Get the timestamp of the last successful scrape for given instance types.
        
        Args:
            instance_types: List of instance types (defaults to supported types)
            
        Returns:
            Datetime of last scrape, or None if no cached data exists
        """
        if instance_types is None:
            instance_types = self.supported_instance_types
        
        cache_key = self._get_cache_key(instance_types)
        
        if cache_key in self._cache:
            timestamp: datetime = self._cache[cache_key]["timestamp"]
            return timestamp
        
        return None