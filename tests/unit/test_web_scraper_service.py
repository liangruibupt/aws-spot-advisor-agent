"""
Unit tests for WebScraperService.

Tests cover web scraping functionality, caching behavior, data validation,
and integration with BedrockAgentService using mocked responses.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.services.web_scraper_service import WebScraperService, WebScraperServiceError
from src.services.bedrock_agent_service import BedrockAgentService, BedrockAgentServiceError
from src.models.spot_data import RawSpotData


class TestWebScraperService:
    """Test cases for WebScraperService class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock BedrockAgentService
        self.mock_bedrock_service = Mock(spec=BedrockAgentService)
        self.service = WebScraperService(
            bedrock_service=self.mock_bedrock_service,
            cache_ttl_seconds=3600
        )

    def test_initialization_with_bedrock_service(self):
        """Test service initialization with provided BedrockAgentService."""
        mock_bedrock = Mock(spec=BedrockAgentService)
        service = WebScraperService(bedrock_service=mock_bedrock, cache_ttl_seconds=1800)
        
        assert service.bedrock_service == mock_bedrock
        assert service.cache_ttl_seconds == 1800
        assert service.supported_instance_types == ["p5en.48xlarge", "p5.48xlarge"]

    def test_initialization_without_bedrock_service(self):
        """Test service initialization without BedrockAgentService (creates default)."""
        with patch('src.services.web_scraper_service.BedrockAgentService') as mock_bedrock_class:
            mock_bedrock_instance = Mock()
            mock_bedrock_class.return_value = mock_bedrock_instance
            
            service = WebScraperService()
            
            assert service.bedrock_service == mock_bedrock_instance
            assert service.cache_ttl_seconds == WebScraperService.CACHE_TTL_SECONDS
            mock_bedrock_class.assert_called_once()

    def test_scrape_spot_data_success(self):
        """Test successful spot data scraping."""
        # Mock data
        timestamp = datetime.now(timezone.utc)
        expected_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.02,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        # Mock BedrockAgentService responses
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock web content"
        self.mock_bedrock_service.parse_spot_data.return_value = expected_data
        
        # Execute scraping
        result = self.service.scrape_spot_data()
        
        # Verify results
        assert len(result) == 2
        assert result[0].region == "us-east-1"
        assert result[1].region == "us-west-2"
        
        # Verify service calls
        self.mock_bedrock_service.execute_web_scraping.assert_called_once_with(
            WebScraperService.SPOT_ADVISOR_URL
        )
        self.mock_bedrock_service.parse_spot_data.assert_called_once_with(
            "mock web content",
            ["p5en.48xlarge", "p5.48xlarge"]
        )

    def test_scrape_spot_data_with_custom_instance_types(self):
        """Test scraping with custom instance types."""
        timestamp = datetime.now(timezone.utc)
        expected_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = expected_data
        
        result = self.service.scrape_spot_data(instance_types=["p5en.48xlarge"])
        
        assert len(result) == 1
        self.mock_bedrock_service.parse_spot_data.assert_called_once_with(
            "mock content",
            ["p5en.48xlarge"]
        )

    def test_scrape_spot_data_invalid_instance_types(self):
        """Test scraping with invalid instance types."""
        with pytest.raises(WebScraperServiceError, match="No valid instance types specified"):
            self.service.scrape_spot_data(instance_types=["invalid.type"])

    def test_scrape_spot_data_mixed_valid_invalid_types(self):
        """Test scraping with mix of valid and invalid instance types."""
        timestamp = datetime.now(timezone.utc)
        expected_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = expected_data
        
        # Should filter out invalid types and proceed with valid ones
        result = self.service.scrape_spot_data(instance_types=["p5en.48xlarge", "invalid.type"])
        
        assert len(result) == 1
        self.mock_bedrock_service.parse_spot_data.assert_called_once_with(
            "mock content",
            ["p5en.48xlarge"]
        )

    def test_scrape_spot_data_bedrock_service_error(self):
        """Test scraping with BedrockAgentService error."""
        self.mock_bedrock_service.execute_web_scraping.side_effect = BedrockAgentServiceError("Service error")
        
        with pytest.raises(WebScraperServiceError, match="Scraping failed"):
            self.service.scrape_spot_data()

    def test_scrape_spot_data_caching_behavior(self):
        """Test caching behavior during scraping."""
        timestamp = datetime.now(timezone.utc)
        expected_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = expected_data
        
        # First call should scrape
        result1 = self.service.scrape_spot_data()
        assert len(result1) == 1
        
        # Second call should use cache
        result2 = self.service.scrape_spot_data()
        assert len(result2) == 1
        assert result1 == result2
        
        # Should only call bedrock service once (first time)
        self.mock_bedrock_service.execute_web_scraping.assert_called_once()

    def test_scrape_spot_data_force_refresh(self):
        """Test force refresh bypasses cache."""
        timestamp = datetime.now(timezone.utc)
        expected_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = expected_data
        
        # First call
        self.service.scrape_spot_data()
        
        # Second call with force refresh
        self.service.scrape_spot_data(force_refresh=True)
        
        # Should call bedrock service twice
        assert self.mock_bedrock_service.execute_web_scraping.call_count == 2

    def test_is_data_fresh_recent_data(self):
        """Test data freshness check with recent data."""
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        
        assert self.service.is_data_fresh(recent_time) is True

    def test_is_data_fresh_old_data(self):
        """Test data freshness check with old data."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        
        assert self.service.is_data_fresh(old_time) is False

    def test_is_data_fresh_custom_max_age(self):
        """Test data freshness check with custom max age."""
        time_30_min_ago = datetime.now(timezone.utc) - timedelta(minutes=30)
        
        # Should be fresh with 1 hour max age
        assert self.service.is_data_fresh(time_30_min_ago, max_age_hours=1.0) is True
        
        # Should not be fresh with 15 minute max age
        assert self.service.is_data_fresh(time_30_min_ago, max_age_hours=0.25) is False

    def test_is_data_fresh_naive_datetime(self):
        """Test data freshness check with naive datetime (no timezone)."""
        naive_time = datetime.now() - timedelta(minutes=30)
        
        # Should handle naive datetime by assuming UTC
        assert self.service.is_data_fresh(naive_time) is True

    def test_is_data_fresh_none_timestamp(self):
        """Test data freshness check with None timestamp."""
        assert self.service.is_data_fresh(None) is False

    def test_get_cache_info_empty_cache(self):
        """Test cache info with empty cache."""
        info = self.service.get_cache_info()
        
        assert info["cache_entries"] == 0
        assert info["cache_ttl_seconds"] == 3600
        assert info["entries"] == []

    def test_get_cache_info_with_data(self):
        """Test cache info with cached data."""
        # Add some data to cache
        timestamp = datetime.now(timezone.utc)
        test_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = test_data
        
        # Populate cache
        self.service.scrape_spot_data()
        
        # Check cache info
        info = self.service.get_cache_info()
        
        assert info["cache_entries"] == 1
        assert len(info["entries"]) == 1
        assert info["entries"][0]["data_count"] == 1
        assert info["entries"][0]["is_valid"] is True

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Populate cache first
        timestamp = datetime.now(timezone.utc)
        test_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = test_data
        
        self.service.scrape_spot_data()
        assert self.service.get_cache_info()["cache_entries"] == 1
        
        # Clear cache
        self.service.clear_cache()
        assert self.service.get_cache_info()["cache_entries"] == 0

    def test_validate_spot_data_valid(self):
        """Test spot data validation with valid data."""
        timestamp = datetime.now(timezone.utc)
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp,
            availability=True
        )
        
        assert self.service._validate_spot_data(valid_data) is True

    def test_validate_spot_data_invalid_price(self):
        """Test spot data validation with invalid price."""
        timestamp = datetime.now(timezone.utc)
        
        # Create valid data first, then modify the price
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp,
            availability=True
        )
        
        # Modify the price to be invalid (bypass validation)
        valid_data.spot_price = -5.00
        
        assert self.service._validate_spot_data(valid_data) is False

    def test_validate_spot_data_invalid_interruption_rate(self):
        """Test spot data validation with invalid interruption rate."""
        timestamp = datetime.now(timezone.utc)
        
        # Create valid data first, then modify the interruption rate
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp,
            availability=True
        )
        
        # Modify the interruption rate to be invalid (bypass validation)
        valid_data.interruption_rate = 1.5
        
        assert self.service._validate_spot_data(valid_data) is False

    def test_validate_spot_data_invalid_currency(self):
        """Test spot data validation with invalid currency."""
        timestamp = datetime.now(timezone.utc)
        
        # Create valid data first, then modify the currency
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp,
            availability=True
        )
        
        # Modify the currency to be invalid (bypass validation)
        valid_data.currency = "EUR"
        
        assert self.service._validate_spot_data(valid_data) is False

    def test_validate_spot_data_empty_region(self):
        """Test spot data validation with empty region."""
        timestamp = datetime.now(timezone.utc)
        
        # Create valid data first, then modify the region
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp,
            availability=True
        )
        
        # Modify the region to be empty (bypass validation)
        valid_data.region = ""
        
        assert self.service._validate_spot_data(valid_data) is False

    def test_get_supported_instance_types(self):
        """Test getting supported instance types."""
        types = self.service.get_supported_instance_types()
        
        assert types == ["p5en.48xlarge", "p5.48xlarge"]
        # Ensure it returns a copy, not the original list
        types.append("test")
        assert self.service.supported_instance_types == ["p5en.48xlarge", "p5.48xlarge"]

    def test_validate_url_valid_aws_url(self):
        """Test URL validation with valid AWS URL."""
        valid_url = "https://aws.amazon.com/ec2/spot/instance-advisor/"
        
        assert self.service.validate_url(valid_url) is True

    def test_validate_url_invalid_domain(self):
        """Test URL validation with invalid domain."""
        invalid_url = "https://example.com/some/path"
        
        assert self.service.validate_url(invalid_url) is False

    def test_validate_url_malformed(self):
        """Test URL validation with malformed URL."""
        malformed_url = "not-a-url"
        
        assert self.service.validate_url(malformed_url) is False

    def test_get_last_scrape_time_no_cache(self):
        """Test getting last scrape time with no cached data."""
        result = self.service.get_last_scrape_time()
        
        assert result is None

    def test_get_last_scrape_time_with_cache(self):
        """Test getting last scrape time with cached data."""
        # Populate cache
        timestamp = datetime.now(timezone.utc)
        test_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = test_data
        
        self.service.scrape_spot_data()
        
        # Get last scrape time
        last_scrape = self.service.get_last_scrape_time()
        
        assert last_scrape is not None
        assert isinstance(last_scrape, datetime)

    def test_cache_key_generation(self):
        """Test cache key generation for different instance type combinations."""
        # Test single type
        key1 = self.service._get_cache_key(["p5en.48xlarge"])
        assert key1 == "p5en.48xlarge"
        
        # Test multiple types (should be sorted)
        key2 = self.service._get_cache_key(["p5.48xlarge", "p5en.48xlarge"])
        key3 = self.service._get_cache_key(["p5en.48xlarge", "p5.48xlarge"])
        assert key2 == key3  # Order shouldn't matter
        assert "|" in key2  # Should contain separator

    def test_cache_expiration(self):
        """Test cache expiration behavior."""
        # Create service with very short cache TTL
        short_ttl_service = WebScraperService(
            bedrock_service=self.mock_bedrock_service,
            cache_ttl_seconds=1  # 1 second TTL
        )
        
        timestamp = datetime.now(timezone.utc)
        test_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        self.mock_bedrock_service.execute_web_scraping.return_value = "mock content"
        self.mock_bedrock_service.parse_spot_data.return_value = test_data
        
        # First call should populate cache
        short_ttl_service.scrape_spot_data()
        
        # Simulate cache expiration by manually updating cache timestamp
        cache_key = short_ttl_service._get_cache_key(["p5en.48xlarge", "p5.48xlarge"])
        short_ttl_service._cache[cache_key]["timestamp"] = datetime.now(timezone.utc) - timedelta(seconds=2)
        
        # Next call should refresh cache (not use expired cache)
        short_ttl_service.scrape_spot_data()
        
        # Should have called bedrock service twice
        assert self.mock_bedrock_service.execute_web_scraping.call_count == 2


class TestWebScraperServiceIntegration:
    """Integration tests for WebScraperService."""

    def test_complete_scraping_workflow(self):
        """Test complete scraping workflow with realistic data."""
        # Create service with mock bedrock service
        mock_bedrock = Mock(spec=BedrockAgentService)
        service = WebScraperService(bedrock_service=mock_bedrock)
        
        # Mock realistic spot data
        timestamp = datetime.now(timezone.utc)
        mock_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=13.00,
                currency="USD",
                interruption_rate=0.02,
                timestamp=timestamp,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5.48xlarge",
                spot_price=14.00,
                currency="USD",
                interruption_rate=0.04,
                timestamp=timestamp,
                availability=True
            )
        ]
        
        mock_bedrock.execute_web_scraping.return_value = "mock html content"
        mock_bedrock.parse_spot_data.return_value = mock_data
        
        # Execute scraping
        result = service.scrape_spot_data()
        
        # Verify results
        assert len(result) == 3
        assert all(isinstance(item, RawSpotData) for item in result)
        assert all(item.currency == "USD" for item in result)
        assert all(0 <= item.interruption_rate <= 1 for item in result)
        
        # Verify cache was populated
        cache_info = service.get_cache_info()
        assert cache_info["cache_entries"] == 1
        assert cache_info["entries"][0]["data_count"] == 3

    def test_error_recovery_and_logging(self):
        """Test error recovery and logging behavior."""
        mock_bedrock = Mock(spec=BedrockAgentService)
        service = WebScraperService(bedrock_service=mock_bedrock)
        
        # Test BedrockAgentService error
        mock_bedrock.execute_web_scraping.side_effect = BedrockAgentServiceError("Network error")
        
        with pytest.raises(WebScraperServiceError):
            service.scrape_spot_data()
        
        # Test unexpected error
        mock_bedrock.execute_web_scraping.side_effect = Exception("Unexpected error")
        
        with pytest.raises(WebScraperServiceError):
            service.scrape_spot_data()