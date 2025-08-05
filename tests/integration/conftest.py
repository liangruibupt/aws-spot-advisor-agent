"""
Configuration and fixtures for integration tests.

This module provides shared fixtures and configuration for integration tests
across the AWS Spot Price Analyzer system.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.services.spot_price_analyzer import SpotPriceAnalyzer
from src.services.web_scraper_service import WebScraperService
from src.services.bedrock_agent_service import BedrockAgentService
from src.services.data_filter_service import DataFilterService
from src.services.ranking_engine import RankingEngine
from src.services.result_formatter import ResultFormatter
from tests.fixtures.mock_responses import MockResponses, TestDataFactory, ErrorScenarios


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def mock_responses():
    """Provide mock responses for testing."""
    return MockResponses()


@pytest.fixture(scope="session")
def test_data_factory():
    """Provide test data factory for creating test objects."""
    return TestDataFactory()


@pytest.fixture(scope="session")
def error_scenarios():
    """Provide error scenarios for testing."""
    return ErrorScenarios()


@pytest.fixture
def real_services():
    """Create real service instances for integration testing."""
    bedrock_service = BedrockAgentService(
        region_name="us-east-1",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    web_scraper = WebScraperService(bedrock_service)
    data_filter = DataFilterService()
    ranking_engine = RankingEngine()
    result_formatter = ResultFormatter()
    
    return {
        'bedrock_service': bedrock_service,
        'web_scraper': web_scraper,
        'data_filter': data_filter,
        'ranking_engine': ranking_engine,
        'result_formatter': result_formatter
    }


@pytest.fixture
def analyzer_with_real_services(real_services):
    """Create SpotPriceAnalyzer with real service instances."""
    return SpotPriceAnalyzer(
        web_scraper=real_services['web_scraper'],
        data_filter=real_services['data_filter'],
        ranking_engine=real_services['ranking_engine'],
        bedrock_service=real_services['bedrock_service'],
        result_formatter=real_services['result_formatter']
    )


@pytest.fixture
def mock_successful_agent_response(mock_responses):
    """Create mock successful agent response."""
    return mock_responses.create_mock_agent_response(
        mock_responses.get_successful_spot_data_response()
    )


@pytest.fixture
def mock_high_interruption_agent_response(mock_responses):
    """Create mock agent response with high interruption rates."""
    return mock_responses.create_mock_agent_response(
        mock_responses.get_high_interruption_response()
    )


@pytest.fixture
def mock_limited_regions_agent_response(mock_responses):
    """Create mock agent response with limited regions."""
    return mock_responses.create_mock_agent_response(
        mock_responses.get_limited_regions_response()
    )


@pytest.fixture
def mock_corrupted_data_agent_response(mock_responses):
    """Create mock agent response with corrupted data."""
    return mock_responses.create_mock_agent_response(
        mock_responses.get_corrupted_data_response()
    )


@pytest.fixture
def mock_empty_agent_response(mock_responses):
    """Create mock agent response with empty data."""
    return mock_responses.create_mock_agent_response(
        mock_responses.get_empty_response()
    )


@pytest.fixture
def mock_malformed_agent_response(mock_responses):
    """Create mock agent response with malformed JSON."""
    return mock_responses.create_mock_agent_response(
        mock_responses.get_malformed_json_response()
    )


@pytest.fixture
def sample_raw_spot_data(test_data_factory):
    """Provide sample raw spot data for testing."""
    return test_data_factory.create_raw_spot_data_list()


@pytest.fixture
def high_interruption_spot_data(test_data_factory):
    """Provide spot data with high interruption rates."""
    return test_data_factory.create_high_interruption_data_list()


@pytest.fixture
def mixed_validity_spot_data(test_data_factory):
    """Provide spot data with mixed validity."""
    return test_data_factory.create_mixed_validity_data_list()


@pytest.fixture
def mock_bedrock_agent():
    """Create mock Bedrock agent for testing."""
    with patch('src.services.bedrock_agent_service.Agent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def mock_bedrock_agent_class():
    """Create mock Bedrock agent class for testing."""
    with patch('src.services.bedrock_agent_service.Agent') as mock_agent_class:
        yield mock_agent_class


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all caches before each test."""
    # This fixture runs automatically before each test
    yield
    # Cleanup after test if needed


@pytest.fixture
def integration_test_config():
    """Provide configuration for integration tests."""
    return {
        'default_instance_types': ['p5en.48xlarge', 'p5.48xlarge'],
        'default_max_interruption_rate': 0.05,
        'default_top_count': 3,
        'test_timeout_seconds': 30,
        'retry_attempts': 3,
        'cache_ttl_seconds': 3600
    }


@pytest.fixture
def aws_error_responses(error_scenarios):
    """Provide AWS error responses for testing."""
    return {
        'throttling': error_scenarios.get_throttling_error(),
        'service_unavailable': error_scenarios.get_service_unavailable_error(),
        'unauthorized': error_scenarios.get_unauthorized_error(),
        'network': error_scenarios.get_network_error(),
        'timeout': error_scenarios.get_timeout_error()
    }


class IntegrationTestHelper:
    """Helper class for integration tests."""

    @staticmethod
    def assert_valid_analysis_response(response, expected_count=None):
        """Assert that analysis response is valid."""
        from src.models.spot_data import AnalysisResponse
        
        assert isinstance(response, AnalysisResponse)
        assert isinstance(response.results, list)
        assert isinstance(response.total_regions_analyzed, int)
        assert isinstance(response.filtered_regions_count, int)
        assert isinstance(response.data_collection_timestamp, datetime)
        
        if expected_count is not None:
            assert len(response.results) == expected_count
        
        # Verify results are sorted by price
        if len(response.results) > 1:
            prices = [r.spot_price for r in response.results]
            assert prices == sorted(prices)

    @staticmethod
    def assert_valid_json_response(response):
        """Assert that JSON response is valid."""
        assert isinstance(response, dict)
        assert "results" in response
        assert "metadata" in response
        
        results = response["results"]
        assert isinstance(results, list)
        
        for result in results:
            assert "region" in result
            assert "instance_type" in result
            assert "spot_price" in result
            assert "interruption_rate" in result
            assert "rank" in result

    @staticmethod
    def assert_error_response(response):
        """Assert that error response is valid."""
        assert isinstance(response, dict)
        assert "error" in response
        assert "error_type" in response
        assert "timestamp" in response

    @staticmethod
    def create_mock_agent_with_responses(responses):
        """Create mock agent that returns different responses on successive calls."""
        mock_agent = Mock()
        mock_agent.side_effect = responses
        return mock_agent


@pytest.fixture
def integration_test_helper():
    """Provide integration test helper."""
    return IntegrationTestHelper()


# Pytest markers for different test categories
pytest.mark.integration = pytest.mark.integration
pytest.mark.bedrock = pytest.mark.bedrock
pytest.mark.error_handling = pytest.mark.error_handling
pytest.mark.end_to_end = pytest.mark.end_to_end


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "bedrock: mark test as bedrock integration test"
    )
    config.addinivalue_line(
        "markers", "error_handling: mark test as error handling test"
    )
    config.addinivalue_line(
        "markers", "end_to_end: mark test as end-to-end workflow test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add specific markers based on test file names
        if "bedrock" in str(item.fspath):
            item.add_marker(pytest.mark.bedrock)
        elif "error" in str(item.fspath):
            item.add_marker(pytest.mark.error_handling)
        elif "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.end_to_end)