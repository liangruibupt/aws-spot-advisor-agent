"""
Unit tests for custom exception classes.

This module tests the custom exception classes and their functionality
for structured error handling in the AWS Spot Price Analyzer.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    WebScrapingError,
    BedrockServiceError,
    DataValidationError,
    DataFilteringError,
    InsufficientDataError,
    ConfigurationError,
    CacheError,
    RankingError,
    FormattingError,
    RetryableError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError
)


class TestSpotAnalyzerBaseError:
    """Test cases for the base error class."""
    
    def test_basic_error_creation(self):
        """Test basic error creation with message only."""
        error = SpotAnalyzerBaseError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code == "SpotAnalyzerBaseError"
        assert error.details == {}
        assert error.original_error is None
    
    def test_error_with_all_parameters(self):
        """Test error creation with all parameters."""
        original_error = ValueError("Original error")
        details = {"key": "value", "count": 42}
        
        error = SpotAnalyzerBaseError(
            message="Test error",
            error_code="CUSTOM_ERROR",
            details=details,
            original_error=original_error
        )
        
        assert error.message == "Test error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.details == details
        assert error.original_error == original_error
    
    def test_to_dict_method(self):
        """Test conversion to dictionary."""
        original_error = ValueError("Original error")
        details = {"key": "value"}
        
        error = SpotAnalyzerBaseError(
            message="Test error",
            error_code="CUSTOM_ERROR",
            details=details,
            original_error=original_error
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error"] is True
        assert error_dict["error_code"] == "CUSTOM_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == details
        assert error_dict["original_error"]["type"] == "ValueError"
        assert error_dict["original_error"]["message"] == "Original error"
    
    def test_to_dict_without_original_error(self):
        """Test dictionary conversion without original error."""
        error = SpotAnalyzerBaseError("Test error")
        error_dict = error.to_dict()
        
        assert "original_error" not in error_dict


class TestWebScrapingError:
    """Test cases for web scraping errors."""
    
    def test_web_scraping_error_creation(self):
        """Test web scraping error with URL and status code."""
        error = WebScrapingError(
            message="Failed to scrape",
            url="https://example.com",
            status_code=404
        )
        
        assert error.message == "Failed to scrape"
        assert error.error_code == "WEB_SCRAPING_ERROR"
        assert error.details["url"] == "https://example.com"
        assert error.details["status_code"] == 404
    
    def test_web_scraping_error_minimal(self):
        """Test web scraping error with minimal parameters."""
        error = WebScrapingError("Scraping failed")
        
        assert error.message == "Scraping failed"
        assert error.error_code == "WEB_SCRAPING_ERROR"
        assert error.details == {}


class TestBedrockServiceError:
    """Test cases for Bedrock service errors."""
    
    def test_bedrock_service_error_full(self):
        """Test Bedrock service error with all parameters."""
        error = BedrockServiceError(
            message="Bedrock failed",
            service_error_code="ThrottlingException",
            region="us-east-1",
            model_name="claude-3-sonnet"
        )
        
        assert error.message == "Bedrock failed"
        assert error.error_code == "BEDROCK_SERVICE_ERROR"
        assert error.details["service_error_code"] == "ThrottlingException"
        assert error.details["region"] == "us-east-1"
        assert error.details["model_name"] == "claude-3-sonnet"


class TestDataValidationError:
    """Test cases for data validation errors."""
    
    def test_data_validation_error_full(self):
        """Test data validation error with all parameters."""
        error = DataValidationError(
            message="Invalid field value",
            field_name="spot_price",
            field_value=-1.0,
            validation_rule="must be positive"
        )
        
        assert error.message == "Invalid field value"
        assert error.error_code == "DATA_VALIDATION_ERROR"
        assert error.details["field_name"] == "spot_price"
        assert error.details["field_value"] == "-1.0"
        assert error.details["validation_rule"] == "must be positive"


class TestInsufficientDataError:
    """Test cases for insufficient data errors."""
    
    def test_insufficient_data_error_full(self):
        """Test insufficient data error with all parameters."""
        criteria = {"max_interruption_rate": 0.05}
        
        error = InsufficientDataError(
            message="Not enough regions",
            required_count=3,
            available_count=1,
            criteria=criteria
        )
        
        assert error.message == "Not enough regions"
        assert error.error_code == "INSUFFICIENT_DATA_ERROR"
        assert error.details["required_count"] == 3
        assert error.details["available_count"] == 1
        assert error.details["criteria"] == criteria


class TestRetryableError:
    """Test cases for retryable errors."""
    
    def test_retryable_error_creation(self):
        """Test retryable error creation."""
        error = RetryableError(
            message="Temporary failure",
            retry_after_seconds=30.0,
            max_retries=3
        )
        
        assert error.message == "Temporary failure"
        assert error.error_code == "RETRYABLE_ERROR"
        assert error.details["retry_after_seconds"] == 30.0
        assert error.details["max_retries"] == 3


class TestNetworkError:
    """Test cases for network errors."""
    
    def test_network_error_creation(self):
        """Test network error creation."""
        error = NetworkError(
            message="Connection failed",
            url="https://example.com",
            timeout_seconds=30.0
        )
        
        assert error.message == "Connection failed"
        assert error.error_code == "NETWORK_ERROR"
        assert error.details["url"] == "https://example.com"
        assert error.details["timeout_seconds"] == 30.0


class TestServiceUnavailableError:
    """Test cases for service unavailable errors."""
    
    def test_service_unavailable_error_creation(self):
        """Test service unavailable error creation."""
        error = ServiceUnavailableError(
            message="Service down",
            service_name="AWS Bedrock",
            status_code=503
        )
        
        assert error.message == "Service down"
        assert error.error_code == "SERVICE_UNAVAILABLE_ERROR"
        assert error.details["service_name"] == "AWS Bedrock"
        assert error.details["status_code"] == 503


class TestRateLimitError:
    """Test cases for rate limit errors."""
    
    def test_rate_limit_error_creation(self):
        """Test rate limit error creation."""
        error = RateLimitError(
            message="Too many requests",
            service_name="AWS Bedrock",
            retry_after_seconds=60.0
        )
        
        assert error.message == "Too many requests"
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.details["service_name"] == "AWS Bedrock"
        assert error.details["retry_after_seconds"] == 60.0


class TestConfigurationError:
    """Test cases for configuration errors."""
    
    def test_configuration_error_creation(self):
        """Test configuration error creation."""
        error = ConfigurationError(
            message="Invalid config",
            config_key="max_interruption_rate",
            config_value="invalid",
            expected_type="float"
        )
        
        assert error.message == "Invalid config"
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.details["config_key"] == "max_interruption_rate"
        assert error.details["config_value"] == "invalid"
        assert error.details["expected_type"] == "float"


class TestCacheError:
    """Test cases for cache errors."""
    
    def test_cache_error_creation(self):
        """Test cache error creation."""
        error = CacheError(
            message="Cache operation failed",
            cache_key="test_key",
            operation="retrieve"
        )
        
        assert error.message == "Cache operation failed"
        assert error.error_code == "CACHE_ERROR"
        assert error.details["cache_key"] == "test_key"
        assert error.details["operation"] == "retrieve"


class TestRankingError:
    """Test cases for ranking errors."""
    
    def test_ranking_error_creation(self):
        """Test ranking error creation."""
        error = RankingError(
            message="Ranking failed",
            ranking_criteria="price",
            data_count=0
        )
        
        assert error.message == "Ranking failed"
        assert error.error_code == "RANKING_ERROR"
        assert error.details["ranking_criteria"] == "price"
        assert error.details["data_count"] == 0


class TestFormattingError:
    """Test cases for formatting errors."""
    
    def test_formatting_error_creation(self):
        """Test formatting error creation."""
        error = FormattingError(
            message="Format failed",
            format_type="JSON",
            data_type="SpotPriceResult"
        )
        
        assert error.message == "Format failed"
        assert error.error_code == "FORMATTING_ERROR"
        assert error.details["format_type"] == "JSON"
        assert error.details["data_type"] == "SpotPriceResult"


class TestErrorInheritance:
    """Test error inheritance relationships."""
    
    def test_inheritance_chain(self):
        """Test that errors inherit from base classes correctly."""
        # Test that specific errors inherit from base error
        assert issubclass(WebScrapingError, SpotAnalyzerBaseError)
        assert issubclass(BedrockServiceError, SpotAnalyzerBaseError)
        assert issubclass(DataValidationError, SpotAnalyzerBaseError)
        
        # Test that retryable errors inherit correctly
        assert issubclass(RetryableError, SpotAnalyzerBaseError)
        assert issubclass(NetworkError, RetryableError)
        assert issubclass(ServiceUnavailableError, RetryableError)
        assert issubclass(RateLimitError, RetryableError)
    
    def test_exception_isinstance_checks(self):
        """Test isinstance checks work correctly."""
        network_error = NetworkError("Network failed")
        
        assert isinstance(network_error, NetworkError)
        assert isinstance(network_error, RetryableError)
        assert isinstance(network_error, SpotAnalyzerBaseError)
        assert isinstance(network_error, Exception)
    
    def test_error_code_inheritance(self):
        """Test that error codes are set correctly in inheritance."""
        network_error = NetworkError("Network failed")
        service_error = ServiceUnavailableError("Service down")
        
        assert network_error.error_code == "NETWORK_ERROR"
        assert service_error.error_code == "SERVICE_UNAVAILABLE_ERROR"


class TestErrorChaining:
    """Test error chaining with original errors."""
    
    def test_error_chaining(self):
        """Test chaining errors with original exceptions."""
        original = ValueError("Original problem")
        chained = WebScrapingError(
            message="Scraping failed due to validation",
            original_error=original
        )
        
        assert chained.original_error == original
        
        error_dict = chained.to_dict()
        assert error_dict["original_error"]["type"] == "ValueError"
        assert error_dict["original_error"]["message"] == "Original problem"
    
    def test_multiple_error_chaining(self):
        """Test multiple levels of error chaining."""
        root_error = ConnectionError("Network connection failed")
        network_error = NetworkError(
            message="Network operation failed",
            original_error=root_error
        )
        scraping_error = WebScrapingError(
            message="Web scraping failed",
            original_error=network_error
        )
        
        assert scraping_error.original_error == network_error
        assert network_error.original_error == root_error
        
        # Test dictionary representation includes original error
        error_dict = scraping_error.to_dict()
        assert error_dict["original_error"]["type"] == "NetworkError"