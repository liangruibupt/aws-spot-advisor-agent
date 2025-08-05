"""
Unit tests for SpotPriceAnalyzer service.

This module contains comprehensive unit tests for the main orchestration
service that coordinates all components of the spot price analysis workflow.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from typing import List

from src.services.spot_price_analyzer import (
    SpotPriceAnalyzer,
    SpotPriceAnalyzerError,
    InsufficientRegionsError,
    ServiceFailureError
)
from src.utils.exceptions import (
    InsufficientDataError,
    ConfigurationError
)
from src.models.spot_data import RawSpotData, SpotPriceResult, AnalysisResponse
from src.services.web_scraper_service import WebScraperServiceError
from src.services.data_filter_service import DataFilterServiceError
from src.services.bedrock_agent_service import BedrockAgentServiceError
from src.services.result_formatter import ResultFormatter


class TestSpotPriceAnalyzer:
    """Test cases for SpotPriceAnalyzer class."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        mock_bedrock = Mock()
        mock_web_scraper = Mock()
        mock_data_filter = Mock()
        mock_ranking_engine = Mock()
        mock_result_formatter = Mock()
        
        return {
            'bedrock': mock_bedrock,
            'web_scraper': mock_web_scraper,
            'data_filter': mock_data_filter,
            'ranking_engine': mock_ranking_engine,
            'result_formatter': mock_result_formatter
        }

    @pytest.fixture
    def analyzer(self, mock_services):
        """Create SpotPriceAnalyzer instance with mocked services."""
        return SpotPriceAnalyzer(
            web_scraper=mock_services['web_scraper'],
            data_filter=mock_services['data_filter'],
            ranking_engine=mock_services['ranking_engine'],
            bedrock_service=mock_services['bedrock'],
            result_formatter=mock_services['result_formatter']
        )

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw spot data for testing."""
        timestamp = datetime.now(timezone.utc)
        return [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=10.50,
                currency="USD",
                interruption_rate=0.02,
                timestamp=timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=11.25,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=9.75,
                currency="USD",
                interruption_rate=0.04,
                timestamp=timestamp,
                availability=True
            )
        ]

    @pytest.fixture
    def sample_spot_results(self):
        """Create sample spot price results for testing."""
        timestamp = datetime.now(timezone.utc)
        return [
            SpotPriceResult(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=9.75,
                currency="USD",
                interruption_rate=0.04,
                rank=1,
                data_timestamp=timestamp
            ),
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=10.50,
                currency="USD",
                interruption_rate=0.02,
                rank=2,
                data_timestamp=timestamp
            ),
            SpotPriceResult(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=11.25,
                currency="USD",
                interruption_rate=0.03,
                rank=3,
                data_timestamp=timestamp
            )
        ]

    def test_init_default_services(self):
        """Test initialization with default services."""
        with patch('src.services.spot_price_analyzer.BedrockAgentService') as mock_bedrock, \
             patch('src.services.spot_price_analyzer.WebScraperService') as mock_web_scraper, \
             patch('src.services.spot_price_analyzer.DataFilterService') as mock_data_filter, \
             patch('src.services.spot_price_analyzer.RankingEngine') as mock_ranking_engine:
            
            analyzer = SpotPriceAnalyzer()
            
            # Verify services were created
            mock_bedrock.assert_called_once()
            mock_web_scraper.assert_called_once()
            mock_data_filter.assert_called_once()
            mock_ranking_engine.assert_called_once()
            
            # Verify default configuration
            assert analyzer.instance_types == ["p5en.48xlarge", "p5.48xlarge"]
            assert analyzer.max_interruption_rate == 0.05

    def test_init_with_custom_services(self, mock_services):
        """Test initialization with custom services."""
        analyzer = SpotPriceAnalyzer(
            web_scraper=mock_services['web_scraper'],
            data_filter=mock_services['data_filter'],
            ranking_engine=mock_services['ranking_engine'],
            bedrock_service=mock_services['bedrock']
        )
        
        assert analyzer.web_scraper == mock_services['web_scraper']
        assert analyzer.data_filter == mock_services['data_filter']
        assert analyzer.ranking_engine == mock_services['ranking_engine']
        assert analyzer.bedrock_service == mock_services['bedrock']

    def test_analyze_spot_prices_success(self, analyzer, mock_services, sample_raw_data, sample_spot_results):
        """Test successful spot price analysis workflow."""
        # Setup mocks
        mock_services['web_scraper'].scrape_spot_data.return_value = sample_raw_data
        mock_services['data_filter'].filter_and_validate.return_value = sample_raw_data
        mock_services['ranking_engine'].rank_and_get_top.return_value = sample_spot_results
        mock_services['data_filter'].get_filter_statistics.return_value = {}
        
        # Execute analysis
        result = analyzer.analyze_spot_prices()
        
        # Verify service calls
        mock_services['web_scraper'].scrape_spot_data.assert_called_once()
        mock_services['data_filter'].filter_and_validate.assert_called_once()
        mock_services['ranking_engine'].rank_and_get_top.assert_called_once()
        
        # Verify result
        assert isinstance(result, AnalysisResponse)
        assert len(result.results) == 3
        assert result.total_regions_analyzed == 3
        assert result.filtered_regions_count == 3
        assert isinstance(result.data_collection_timestamp, datetime)

    def test_analyze_spot_prices_with_custom_parameters(self, analyzer, mock_services, sample_raw_data, sample_spot_results):
        """Test analysis with custom parameters."""
        custom_instance_types = ["p5.48xlarge"]
        custom_max_rate = 0.03
        custom_top_count = 2
        
        # Setup mocks
        mock_services['web_scraper'].scrape_spot_data.return_value = sample_raw_data
        mock_services['data_filter'].filter_and_validate.return_value = sample_raw_data[:2]
        mock_services['ranking_engine'].rank_and_get_top.return_value = sample_spot_results[:2]
        mock_services['data_filter'].get_filter_statistics.return_value = {}
        
        # Execute analysis
        result = analyzer.analyze_spot_prices(
            instance_types=custom_instance_types,
            max_interruption_rate=custom_max_rate,
            top_count=custom_top_count,
            force_refresh=True
        )
        
        # Verify service calls with custom parameters
        mock_services['web_scraper'].scrape_spot_data.assert_called_once_with(
            instance_types=custom_instance_types,
            force_refresh=True
        )
        mock_services['data_filter'].filter_and_validate.assert_called_once_with(
            data=sample_raw_data,
            max_interruption_rate=custom_max_rate
        )
        mock_services['ranking_engine'].rank_and_get_top.assert_called_once_with(
            sample_raw_data[:2],
            custom_top_count
        )
        
        # Verify result
        assert len(result.results) == 2

    def test_analyze_spot_prices_web_scraper_error(self, analyzer, mock_services):
        """Test analysis failure due to web scraper error."""
        # Setup mock to raise error
        mock_services['web_scraper'].scrape_spot_data.side_effect = WebScraperServiceError("Scraping failed")
        
        # Execute and verify error
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices()
        
        assert "Failed to scrape spot data" in str(exc_info.value)

    def test_analyze_spot_prices_bedrock_error(self, analyzer, mock_services):
        """Test analysis failure due to Bedrock service error."""
        # Setup mock to raise error
        mock_services['web_scraper'].scrape_spot_data.side_effect = BedrockAgentServiceError("Bedrock failed")
        
        # Execute and verify error
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices()
        
        assert "Web scraping service failed" in str(exc_info.value)

    def test_analyze_spot_prices_data_filter_error(self, analyzer, mock_services, sample_raw_data):
        """Test analysis failure due to data filter error."""
        # Setup mocks
        mock_services['web_scraper'].scrape_spot_data.return_value = sample_raw_data
        mock_services['data_filter'].filter_and_validate.side_effect = DataFilterServiceError("Filtering failed")
        
        # Execute and verify error
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices()
        
        assert "Failed to filter data" in str(exc_info.value)

    def test_analyze_spot_prices_no_data_scraped(self, analyzer, mock_services):
        """Test analysis failure when no data is scraped."""
        # Setup mock to return empty data
        mock_services['web_scraper'].scrape_spot_data.return_value = []
        
        # Execute and verify error
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices()
        
        assert "No spot data retrieved" in str(exc_info.value)

    def test_analyze_spot_prices_insufficient_regions(self, analyzer, mock_services, sample_raw_data):
        """Test analysis with insufficient regions after filtering."""
        # Setup mocks - no regions pass filtering
        mock_services['web_scraper'].scrape_spot_data.return_value = sample_raw_data
        mock_services['data_filter'].filter_and_validate.return_value = []
        
        # Execute and verify error
        with pytest.raises(InsufficientDataError) as exc_info:
            analyzer.analyze_spot_prices()
        
        assert "No regions meet the criteria" in str(exc_info.value)

    def test_analyze_spot_prices_with_warnings(self, analyzer, mock_services, sample_raw_data, sample_spot_results):
        """Test analysis that generates warnings."""
        # Setup mocks - only 2 regions pass filtering
        filtered_data = sample_raw_data[:2]
        results = sample_spot_results[:2]
        
        mock_services['web_scraper'].scrape_spot_data.return_value = sample_raw_data
        mock_services['data_filter'].filter_and_validate.return_value = filtered_data
        mock_services['ranking_engine'].rank_and_get_top.return_value = results
        mock_services['data_filter'].get_filter_statistics.return_value = {
            "validation_excluded_count": 1
        }
        
        # Execute analysis
        result = analyzer.analyze_spot_prices()
        
        # Verify warnings are added
        assert result.has_warnings
        assert len(result.warnings) >= 2  # Should have warnings about region count and exclusions

    def test_get_analysis_status(self, analyzer, mock_services):
        """Test getting analysis status."""
        # Setup mocks
        last_scrape_time = datetime.now(timezone.utc)
        mock_services['web_scraper'].get_last_scrape_time.return_value = last_scrape_time
        mock_services['web_scraper'].get_cache_info.return_value = {
            "cache_entries": 2,
            "cache_ttl_seconds": 3600
        }
        mock_services['data_filter'].get_filter_statistics.return_value = {
            "filtered_count": 5
        }
        
        # Execute
        status = analyzer.get_analysis_status()
        
        # Verify status structure
        assert "configuration" in status
        assert "cache_status" in status
        assert "last_filter_statistics" in status
        assert "services_initialized" in status
        
        # Verify configuration
        config = status["configuration"]
        assert config["instance_types"] == ["p5en.48xlarge", "p5.48xlarge"]
        assert config["max_interruption_rate"] == 0.05
        assert config["max_interruption_rate_percentage"] == "5.0%"
        
        # Verify cache status
        cache_status = status["cache_status"]
        assert cache_status["last_scrape_time"] == last_scrape_time.isoformat()
        assert cache_status["cache_entries"] == 2
        assert cache_status["cache_ttl_seconds"] == 3600
        
        # Verify services initialized
        services = status["services_initialized"]
        assert all(services.values())  # All services should be initialized

    def test_update_configuration_instance_types(self, analyzer, mock_services):
        """Test updating instance types configuration."""
        new_types = ["p5.48xlarge"]
        
        # Setup mock
        mock_services['web_scraper'].get_supported_instance_types.return_value = ["p5en.48xlarge", "p5.48xlarge"]
        
        # Execute
        analyzer.update_configuration(instance_types=new_types)
        
        # Verify update
        assert analyzer.instance_types == new_types

    def test_update_configuration_max_interruption_rate(self, analyzer, mock_services):
        """Test updating max interruption rate configuration."""
        new_rate = 0.03
        
        # Execute
        analyzer.update_configuration(max_interruption_rate=new_rate)
        
        # Verify update
        assert analyzer.max_interruption_rate == new_rate
        mock_services['data_filter'].set_max_interruption_rate.assert_called_once_with(new_rate)

    def test_update_configuration_invalid_instance_types(self, analyzer, mock_services):
        """Test updating configuration with invalid instance types."""
        # Setup mock
        mock_services['web_scraper'].get_supported_instance_types.return_value = ["p5en.48xlarge", "p5.48xlarge"]
        
        # Test empty list
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(instance_types=[])
        assert "non-empty list" in str(exc_info.value)
        
        # Test unsupported types
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(instance_types=["invalid.type"])
        assert "Unsupported instance types" in str(exc_info.value)

    def test_update_configuration_invalid_interruption_rate(self, analyzer):
        """Test updating configuration with invalid interruption rate."""
        # Test negative rate
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(max_interruption_rate=-0.1)
        assert "between 0.0 and 1.0" in str(exc_info.value)
        
        # Test rate > 1.0
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(max_interruption_rate=1.5)
        assert "between 0.0 and 1.0" in str(exc_info.value)

    def test_clear_cache(self, analyzer, mock_services):
        """Test clearing all caches."""
        # Execute
        analyzer.clear_cache()
        
        # Verify service calls
        mock_services['web_scraper'].clear_cache.assert_called_once()
        mock_services['data_filter'].clear_statistics.assert_called_once()

    def test_get_supported_instance_types(self, analyzer, mock_services):
        """Test getting supported instance types."""
        expected_types = ["p5en.48xlarge", "p5.48xlarge"]
        mock_services['web_scraper'].get_supported_instance_types.return_value = expected_types
        
        # Execute
        result = analyzer.get_supported_instance_types()
        
        # Verify
        assert result == expected_types
        mock_services['web_scraper'].get_supported_instance_types.assert_called_once()

    def test_validate_instance_types(self, analyzer, mock_services):
        """Test validating instance types."""
        supported_types = ["p5en.48xlarge", "p5.48xlarge"]
        test_types = ["p5en.48xlarge", "invalid.type", "p5.48xlarge"]
        
        mock_services['web_scraper'].get_supported_instance_types.return_value = supported_types
        
        # Execute
        result = analyzer.validate_instance_types(test_types)
        
        # Verify
        assert result["valid"] == ["p5en.48xlarge", "p5.48xlarge"]
        assert result["invalid"] == ["invalid.type"]

    def test_scrape_spot_data_success(self, analyzer, mock_services, sample_raw_data):
        """Test successful spot data scraping."""
        instance_types = ["p5en.48xlarge"]
        mock_services['web_scraper'].scrape_spot_data.return_value = sample_raw_data
        
        # Execute
        result = analyzer._scrape_spot_data(instance_types, False)
        
        # Verify
        assert result == sample_raw_data
        mock_services['web_scraper'].scrape_spot_data.assert_called_once_with(
            instance_types=instance_types,
            force_refresh=False
        )

    def test_scrape_spot_data_failure(self, analyzer, mock_services):
        """Test spot data scraping failure."""
        mock_services['web_scraper'].scrape_spot_data.side_effect = WebScraperServiceError("Scraping failed")
        
        # Execute and verify error
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer._scrape_spot_data(["p5en.48xlarge"], False)
        
        assert "Failed to scrape spot data" in str(exc_info.value)

    def test_filter_data_success(self, analyzer, mock_services, sample_raw_data):
        """Test successful data filtering."""
        max_rate = 0.05
        mock_services['data_filter'].filter_and_validate.return_value = sample_raw_data
        
        # Execute
        result = analyzer._filter_data(sample_raw_data, max_rate)
        
        # Verify
        assert result == sample_raw_data
        mock_services['data_filter'].filter_and_validate.assert_called_once_with(
            data=sample_raw_data,
            max_interruption_rate=max_rate
        )

    def test_filter_data_failure(self, analyzer, mock_services, sample_raw_data):
        """Test data filtering failure."""
        mock_services['data_filter'].filter_and_validate.side_effect = DataFilterServiceError("Filtering failed")
        
        # Execute and verify error
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer._filter_data(sample_raw_data, 0.05)
        
        assert "Failed to filter data" in str(exc_info.value)

    def test_validate_sufficient_regions_success(self, analyzer, sample_raw_data):
        """Test successful region validation."""
        # Should not raise error with sufficient regions
        analyzer._validate_sufficient_regions(sample_raw_data, 3)

    def test_validate_sufficient_regions_no_regions(self, analyzer):
        """Test region validation with no regions."""
        with pytest.raises(InsufficientDataError) as exc_info:
            analyzer._validate_sufficient_regions([], 3)
        
        assert "No regions meet the criteria" in str(exc_info.value)

    def test_validate_sufficient_regions_fewer_than_requested(self, analyzer, sample_raw_data):
        """Test region validation with fewer regions than requested."""
        # Should not raise error, just log warning
        analyzer._validate_sufficient_regions(sample_raw_data[:2], 3)

    def test_rank_and_select_top(self, analyzer, mock_services, sample_raw_data, sample_spot_results):
        """Test ranking and selecting top regions."""
        mock_services['ranking_engine'].rank_and_get_top.return_value = sample_spot_results
        
        # Execute
        result = analyzer._rank_and_select_top(sample_raw_data, 3)
        
        # Verify
        assert result == sample_spot_results
        mock_services['ranking_engine'].rank_and_get_top.assert_called_once_with(sample_raw_data, 3)

    def test_create_analysis_response(self, analyzer, mock_services, sample_spot_results):
        """Test creating analysis response."""
        analysis_time = datetime.now(timezone.utc)
        mock_services['data_filter'].get_filter_statistics.return_value = {}
        
        # Execute
        result = analyzer._create_analysis_response(
            results=sample_spot_results,
            total_analyzed=5,
            filtered_count=3,
            analysis_time=analysis_time
        )
        
        # Verify
        assert isinstance(result, AnalysisResponse)
        assert result.results == sample_spot_results
        assert result.total_regions_analyzed == 5
        assert result.filtered_regions_count == 3
        assert result.data_collection_timestamp == analysis_time

    def test_create_analysis_response_with_warnings(self, analyzer, mock_services, sample_spot_results):
        """Test creating analysis response with warnings."""
        analysis_time = datetime.now(timezone.utc)
        mock_services['data_filter'].get_filter_statistics.return_value = {
            "validation_excluded_count": 2
        }
        
        # Execute with fewer than 3 filtered regions
        result = analyzer._create_analysis_response(
            results=sample_spot_results[:2],
            total_analyzed=5,
            filtered_count=2,
            analysis_time=analysis_time
        )
        
        # Verify warnings are added
        assert result.has_warnings
        assert len(result.warnings) >= 2  # Should have multiple warnings

    def test_unexpected_error_handling(self, analyzer, mock_services):
        """Test handling of unexpected errors during analysis."""
        # Setup mock to raise unexpected error
        mock_services['web_scraper'].scrape_spot_data.side_effect = RuntimeError("Unexpected error")
        
        # Execute and verify error
        with pytest.raises(SpotPriceAnalyzerError) as exc_info:
            analyzer.analyze_spot_prices()
        
        assert "Analysis failed" in str(exc_info.value)


class TestSpotPriceAnalyzerIntegration:
    """Integration-style tests for SpotPriceAnalyzer."""

    def test_full_workflow_with_real_services(self):
        """Test the full workflow with real service instances (mocked externally)."""
        with patch('src.services.spot_price_analyzer.BedrockAgentService') as mock_bedrock_cls, \
             patch('src.services.spot_price_analyzer.WebScraperService') as mock_web_scraper_cls, \
             patch('src.services.spot_price_analyzer.DataFilterService') as mock_data_filter_cls, \
             patch('src.services.spot_price_analyzer.RankingEngine') as mock_ranking_engine_cls:
            
            # Setup service instances
            mock_bedrock = Mock()
            mock_web_scraper = Mock()
            mock_data_filter = Mock()
            mock_ranking_engine = Mock()
            
            mock_bedrock_cls.return_value = mock_bedrock
            mock_web_scraper_cls.return_value = mock_web_scraper
            mock_data_filter_cls.return_value = mock_data_filter
            mock_ranking_engine_cls.return_value = mock_ranking_engine
            
            # Setup workflow data
            timestamp = datetime.now(timezone.utc)
            raw_data = [
                RawSpotData(
                    region="us-east-1",
                    instance_type="p5en.48xlarge",
                    spot_price=10.50,
                    currency="USD",
                    interruption_rate=0.02,
                    timestamp=timestamp,
                    availability=True
                )
            ]
            
            spot_results = [
                SpotPriceResult(
                    region="us-east-1",
                    instance_type="p5en.48xlarge",
                    spot_price=10.50,
                    currency="USD",
                    interruption_rate=0.02,
                    rank=1,
                    data_timestamp=timestamp
                )
            ]
            
            # Setup mock returns
            mock_web_scraper.scrape_spot_data.return_value = raw_data
            mock_data_filter.filter_and_validate.return_value = raw_data
            mock_ranking_engine.rank_and_get_top.return_value = spot_results
            mock_data_filter.get_filter_statistics.return_value = {}
            
            # Create analyzer and run analysis
            analyzer = SpotPriceAnalyzer()
            result = analyzer.analyze_spot_prices()
            
            # Verify the full workflow executed
            assert isinstance(result, AnalysisResponse)
            assert len(result.results) == 1
            assert result.results[0].region == "us-east-1"
            
            # Verify all services were called
            mock_web_scraper.scrape_spot_data.assert_called_once()
            mock_data_filter.filter_and_validate.assert_called_once()
            mock_ranking_engine.rank_and_get_top.assert_called_once()

    def test_error_propagation_through_workflow(self):
        """Test that errors propagate correctly through the workflow."""
        with patch('src.services.spot_price_analyzer.WebScraperService') as mock_web_scraper_cls:
            mock_web_scraper = Mock()
            mock_web_scraper_cls.return_value = mock_web_scraper
            
            # Setup error in web scraper
            mock_web_scraper.scrape_spot_data.side_effect = WebScraperServiceError("Network error")
            
            # Create analyzer and verify error propagation
            analyzer = SpotPriceAnalyzer()
            
            with pytest.raises(ServiceFailureError) as exc_info:
                analyzer.analyze_spot_prices()
            
            assert "Failed to scrape spot data" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)


class TestSpotPriceAnalyzerJSONFormatting:
    """Test cases for JSON formatting functionality in SpotPriceAnalyzer."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for JSON formatting tests."""
        mock_bedrock = Mock()
        mock_web_scraper = Mock()
        mock_data_filter = Mock()
        mock_ranking_engine = Mock()
        mock_result_formatter = Mock()
        
        return {
            'bedrock': mock_bedrock,
            'web_scraper': mock_web_scraper,
            'data_filter': mock_data_filter,
            'ranking_engine': mock_ranking_engine,
            'result_formatter': mock_result_formatter
        }

    @pytest.fixture
    def analyzer(self, mock_services):
        """Create SpotPriceAnalyzer instance with mocked services."""
        return SpotPriceAnalyzer(
            web_scraper=mock_services['web_scraper'],
            data_filter=mock_services['data_filter'],
            ranking_engine=mock_services['ranking_engine'],
            bedrock_service=mock_services['bedrock'],
            result_formatter=mock_services['result_formatter']
        )

    @pytest.fixture
    def sample_analysis_response(self):
        """Create sample AnalysisResponse for testing."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        
        results = [
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.5678,
                currency="USD",
                interruption_rate=0.0234,
                rank=1,
                data_timestamp=timestamp
            ),
            SpotPriceResult(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=13.1234,
                currency="USD",
                interruption_rate=0.0456,
                rank=2,
                data_timestamp=timestamp
            )
        ]
        
        return AnalysisResponse(
            results=results,
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=timestamp,
            warnings=["Test warning"]
        )

    def test_analyze_spot_prices_json_success(self, analyzer, mock_services, sample_analysis_response):
        """Test successful JSON analysis response."""
        # Mock the analyze_spot_prices method to return sample response
        with patch.object(analyzer, 'analyze_spot_prices', return_value=sample_analysis_response):
            # Mock the formatter
            expected_formatted = {
                "results": [{"region": "us-east-1", "price": 12.5678}],
                "metadata": {"total": 10}
            }
            mock_services['result_formatter'].format_analysis_response.return_value = expected_formatted
            
            # Call the JSON method
            result = analyzer.analyze_spot_prices_json(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )
            
            # Verify the formatter was called
            mock_services['result_formatter'].format_analysis_response.assert_called_once_with(sample_analysis_response)
            
            # Verify the result
            assert result == expected_formatted

    def test_analyze_spot_prices_json_with_summary(self, analyzer, mock_services, sample_analysis_response):
        """Test JSON analysis response with summary statistics."""
        # Mock the analyze_spot_prices method
        with patch.object(analyzer, 'analyze_spot_prices', return_value=sample_analysis_response):
            # Mock the formatter methods
            expected_formatted = {"results": [], "metadata": {}}
            expected_summary = {"summary": "stats"}
            
            mock_services['result_formatter'].format_analysis_response.return_value = expected_formatted
            mock_services['result_formatter'].format_summary_statistics.return_value = expected_summary
            
            # Call with summary enabled
            result = analyzer.analyze_spot_prices_json(include_summary=True)
            
            # Verify both formatter methods were called
            mock_services['result_formatter'].format_analysis_response.assert_called_once()
            mock_services['result_formatter'].format_summary_statistics.assert_called_once_with(sample_analysis_response)
            
            # Verify summary was added
            assert result["summary_statistics"] == expected_summary

    def test_analyze_spot_prices_json_error_handling(self, analyzer, mock_services):
        """Test JSON analysis response error handling."""
        # Mock analyze_spot_prices to raise an exception
        with patch.object(analyzer, 'analyze_spot_prices', side_effect=ServiceFailureError("Test error")):
            # Mock the error formatter
            expected_error = {"error": True, "message": "Test error"}
            with patch.object(analyzer.error_formatter, 'format_error_response', return_value=expected_error) as mock_error_formatter:
                
                # Call the JSON method
                result = analyzer.analyze_spot_prices_json()
                
                # Verify error formatter was called
                mock_error_formatter.assert_called_once()
            
            # Verify error response
            assert result == expected_error

    def test_analyze_spot_prices_json_string_success(self, analyzer, mock_services, sample_analysis_response):
        """Test successful JSON string analysis response."""
        # Mock the analyze_spot_prices_json method
        expected_json_dict = {"results": [], "metadata": {}}
        expected_json_string = '{"results": [], "metadata": {}}'
        
        with patch.object(analyzer, 'analyze_spot_prices_json', return_value=expected_json_dict):
            mock_services['result_formatter'].to_json_string.return_value = expected_json_string
            
            # Call the JSON string method
            result = analyzer.analyze_spot_prices_json_string(
                instance_types=["p5en.48xlarge"],
                indent=2
            )
            
            # Verify the JSON conversion was called
            mock_services['result_formatter'].to_json_string.assert_called_once_with(expected_json_dict, indent=2)
            
            # Verify the result
            assert result == expected_json_string

    def test_analyze_spot_prices_json_string_error_handling(self, analyzer, mock_services):
        """Test JSON string analysis response error handling."""
        # Mock analyze_spot_prices_json to raise an exception
        with patch.object(analyzer, 'analyze_spot_prices_json', side_effect=Exception("Test error")):
            # Mock the error formatter
            expected_error = {"error": True, "message": "Failed to generate JSON response: Test error"}
            expected_error_string = '{"error": true, "message": "Failed to generate JSON response: Test error"}'
            
            with patch.object(analyzer.error_formatter, 'format_error_response', return_value=expected_error) as mock_error_formatter:
                with patch.object(analyzer.error_formatter, 'to_json_string', return_value=expected_error_string) as mock_to_json:
                    
                    # Call the JSON string method
                    result = analyzer.analyze_spot_prices_json_string()
                    
                    # Verify error handling
                    mock_error_formatter.assert_called_once()
            mock_services['result_formatter'].to_json_string.assert_called_once_with(expected_error, indent=None)
            
            # Verify error response
            assert result == expected_error_string

    def test_format_results_only_success(self, analyzer, mock_services):
        """Test formatting results only without metadata."""
        # Create sample results
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        results = [
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.5678,
                currency="USD",
                interruption_rate=0.0234,
                rank=1,
                data_timestamp=timestamp
            )
        ]
        
        # Mock the formatter
        expected_formatted_results = [{"region": "us-east-1", "price": 12.5678}]
        mock_services['result_formatter'].format_spot_price_results.return_value = expected_formatted_results
        
        # Call the method
        result = analyzer.format_results_only(results)
        
        # Verify the formatter was called
        mock_services['result_formatter'].format_spot_price_results.assert_called_once_with(results)
        
        # Verify the result structure
        assert result == {"results": expected_formatted_results}

    def test_format_results_only_error_handling(self, analyzer, mock_services):
        """Test error handling in format_results_only."""
        # Create sample results
        results = [Mock()]
        
        # Mock the formatter to raise an exception
        mock_services['result_formatter'].format_spot_price_results.side_effect = Exception("Formatting error")
        
        # Mock the error formatter
        expected_error = {"error": True, "message": "Failed to format results: Formatting error"}
        with patch.object(analyzer.error_formatter, 'format_error_response', return_value=expected_error) as mock_error_formatter:
            
            # Call the method
            result = analyzer.format_results_only(results)
            
            # Verify error handling
            mock_error_formatter.assert_called_once()
            
            # Verify error response
            assert result == expected_error

    def test_json_formatting_integration_with_real_formatter(self):
        """Test JSON formatting integration with real ResultFormatter."""
        # Create analyzer with real formatter
        analyzer = SpotPriceAnalyzer()
        
        # Create sample data
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        results = [
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.5678,
                currency="USD",
                interruption_rate=0.0234,
                rank=1,
                data_timestamp=timestamp
            )
        ]
        
        # Test format_results_only with real formatter
        result = analyzer.format_results_only(results)
        
        # Verify structure
        assert "results" in result
        assert len(result["results"]) == 1
        
        # Verify formatting
        formatted_result = result["results"][0]
        assert formatted_result["region"] == "us-east-1"
        assert formatted_result["instance_type"] == "p5en.48xlarge"
        assert formatted_result["spot_price"]["amount"] == 12.5678
        assert formatted_result["spot_price"]["currency"] == "USD"
        assert formatted_result["interruption_rate"] == "2.34%"
        assert formatted_result["rank"] == 1
        assert formatted_result["data_timestamp"] == "2024-01-15T10:30:45+00:00"

    def test_json_string_conversion_with_real_formatter(self):
        """Test JSON string conversion with real ResultFormatter."""
        # Create analyzer with real formatter
        analyzer = SpotPriceAnalyzer()
        
        # Test data
        test_data = {
            "results": [
                {
                    "region": "us-east-1",
                    "price": 12.34
                }
            ],
            "metadata": {
                "total": 1
            }
        }
        
        # Convert to JSON string
        json_string = analyzer.result_formatter.to_json_string(test_data, indent=2)
        
        # Verify it's valid JSON
        import json
        parsed_back = json.loads(json_string)
        assert parsed_back == test_data
        
        # Verify formatting (should have newlines due to indent)
        assert "\n" in json_string
        assert "  " in json_string  # Should have indentation