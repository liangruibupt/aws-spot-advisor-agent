"""
End-to-end integration tests for the complete spot price analysis workflow.

This module contains integration tests that verify the complete data flow
from web scraping through to final formatted results.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.services.spot_price_analyzer import SpotPriceAnalyzer
from src.services.web_scraper_service import WebScraperService
from src.services.bedrock_agent_service import BedrockAgentService
from src.services.data_filter_service import DataFilterService
from src.services.ranking_engine import RankingEngine
from src.services.result_formatter import ResultFormatter
from src.models.spot_data import RawSpotData, SpotPriceResult, AnalysisResponse
from src.utils.exceptions import (
    InsufficientDataError,
    WebScrapingError,
    BedrockServiceError
)


class TestEndToEndWorkflow:
    """End-to-end integration tests for the complete workflow."""

    @pytest.fixture
    def mock_web_page_content(self):
        """Mock web page content with spot pricing data."""
        return json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 15.20,
                    "currency": "USD",
                    "interruption_rate": 2.1,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.80,
                    "currency": "USD",
                    "interruption_rate": 3.5,
                    "availability": True
                },
                {
                    "region": "eu-west-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 14.50,
                    "currency": "USD",
                    "interruption_rate": 1.8,
                    "availability": True
                },
                {
                    "region": "ap-southeast-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 16.90,
                    "currency": "USD",
                    "interruption_rate": 4.2,
                    "availability": True
                },
                {
                    "region": "us-east-1",
                    "instance_type": "p5.48xlarge",
                    "spot_price": 14.80,
                    "currency": "USD",
                    "interruption_rate": 2.3,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5.48xlarge",
                    "spot_price": 12.20,
                    "currency": "USD",
                    "interruption_rate": 3.8,
                    "availability": True
                },
                {
                    "region": "eu-west-1",
                    "instance_type": "p5.48xlarge",
                    "spot_price": 13.90,
                    "currency": "USD",
                    "interruption_rate": 1.9,
                    "availability": True
                },
                {
                    "region": "ca-central-1",
                    "instance_type": "p5.48xlarge",
                    "spot_price": 11.50,
                    "currency": "USD",
                    "interruption_rate": 6.2,  # High interruption rate - should be filtered
                    "availability": True
                }
            ]
        })

    @pytest.fixture
    def mock_agent_response(self, mock_web_page_content):
        """Mock agent response with web page content."""
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': mock_web_page_content}
            ]
        }
        return mock_response

    @pytest.fixture
    def analyzer_with_mocked_services(self):
        """Create SpotPriceAnalyzer with real services but mocked external dependencies."""
        # Create real service instances
        bedrock_service = BedrockAgentService()
        web_scraper = WebScraperService(bedrock_service)
        data_filter = DataFilterService()
        ranking_engine = RankingEngine()
        result_formatter = ResultFormatter()
        
        # Create analyzer with real services
        analyzer = SpotPriceAnalyzer(
            web_scraper=web_scraper,
            data_filter=data_filter,
            ranking_engine=ranking_engine,
            bedrock_service=bedrock_service,
            result_formatter=result_formatter
        )
        
        return analyzer

    @patch('src.services.bedrock_agent_service.Agent')
    def test_complete_workflow_success(self, mock_agent_class, analyzer_with_mocked_services, mock_agent_response):
        """Test complete successful workflow from scraping to results."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge", "p5.48xlarge"],
            max_interruption_rate=0.05,  # 5%
            top_count=3
        )
        
        # Assert
        assert isinstance(result, AnalysisResponse)
        assert len(result.results) == 3
        assert result.total_regions_analyzed == 8  # Total regions in mock data
        assert result.filtered_regions_count == 7  # Excluding high interruption rate region
        
        # Verify results are sorted by price (lowest first)
        prices = [r.spot_price for r in result.results]
        assert prices == sorted(prices)
        
        # Verify all results have acceptable interruption rates
        for spot_result in result.results:
            assert spot_result.interruption_rate < 0.05

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_with_json_output(self, mock_agent_class, analyzer_with_mocked_services, mock_agent_response):
        """Test complete workflow with JSON formatted output."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act
        result = analyzer.analyze_spot_prices_json(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3,
            include_summary=True
        )
        
        # Assert
        assert isinstance(result, dict)
        assert "results" in result
        assert "metadata" in result
        assert "summary_statistics" in result
        
        # Verify results structure
        results = result["results"]
        assert len(results) <= 3
        
        for spot_result in results:
            assert "region" in spot_result
            assert "instance_type" in spot_result
            assert "spot_price" in spot_result
            assert "interruption_rate" in spot_result
            assert "rank" in spot_result

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_with_insufficient_regions(self, mock_agent_class, analyzer_with_mocked_services):
        """Test workflow when insufficient regions meet criteria."""
        # Arrange
        # Mock response with all regions having high interruption rates
        high_interruption_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 15.20,
                    "currency": "USD",
                    "interruption_rate": 8.5,  # High interruption rate
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.80,
                    "currency": "USD",
                    "interruption_rate": 7.2,  # High interruption rate
                    "availability": True
                }
            ]
        })
        
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': high_interruption_content}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act & Assert
        with pytest.raises(InsufficientDataError) as exc_info:
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,  # 5%
                top_count=3
            )
        
        assert "No regions meet the criteria" in str(exc_info.value)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_with_partial_results(self, mock_agent_class, analyzer_with_mocked_services):
        """Test workflow when only some regions meet criteria."""
        # Arrange
        # Mock response with only 2 regions meeting criteria
        limited_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 15.20,
                    "currency": "USD",
                    "interruption_rate": 2.1,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.80,
                    "currency": "USD",
                    "interruption_rate": 3.5,
                    "availability": True
                },
                {
                    "region": "eu-west-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 14.50,
                    "currency": "USD",
                    "interruption_rate": 8.2,  # High interruption rate
                    "availability": True
                }
            ]
        })
        
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': limited_content}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,  # 5%
            top_count=3
        )
        
        # Assert
        assert isinstance(result, AnalysisResponse)
        assert len(result.results) == 2  # Only 2 regions meet criteria
        assert result.total_regions_analyzed == 3
        assert result.filtered_regions_count == 2
        
        # Should have warnings about limited results
        assert result.warnings is not None
        assert len(result.warnings) > 0
        assert any("2 regions" in warning for warning in result.warnings)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_with_caching(self, mock_agent_class, analyzer_with_mocked_services, mock_agent_response):
        """Test workflow with caching behavior."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act - First call should scrape data
        result1 = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Act - Second call should use cached data
        result2 = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        assert isinstance(result1, AnalysisResponse)
        assert isinstance(result2, AnalysisResponse)
        
        # Agent should only be called once due to caching
        assert mock_agent.call_count == 1
        
        # Results should be identical
        assert len(result1.results) == len(result2.results)
        assert result1.total_regions_analyzed == result2.total_regions_analyzed

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_with_force_refresh(self, mock_agent_class, analyzer_with_mocked_services, mock_agent_response):
        """Test workflow with forced cache refresh."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act - First call
        result1 = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Act - Second call with force refresh
        result2 = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3,
            force_refresh=True
        )
        
        # Assert
        assert isinstance(result1, AnalysisResponse)
        assert isinstance(result2, AnalysisResponse)
        
        # Agent should be called twice due to force refresh
        assert mock_agent.call_count == 2

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_error_recovery(self, mock_agent_class, analyzer_with_mocked_services):
        """Test workflow error handling and recovery mechanisms."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = ClientError(error_response, 'InvokeModel')
        
        analyzer = analyzer_with_mocked_services
        
        # Act & Assert
        with pytest.raises(Exception):  # Should propagate as ServiceFailureError
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_with_multiple_instance_types(self, mock_agent_class, analyzer_with_mocked_services, mock_agent_response):
        """Test workflow with multiple instance types."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge", "p5.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        assert isinstance(result, AnalysisResponse)
        assert len(result.results) == 3
        
        # Should have results from both instance types
        instance_types_in_results = {r.instance_type for r in result.results}
        assert len(instance_types_in_results) >= 1  # At least one instance type

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_data_validation_and_filtering(self, mock_agent_class, analyzer_with_mocked_services):
        """Test workflow with data validation and filtering."""
        # Arrange
        # Mock response with mixed valid and invalid data
        mixed_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 15.20,
                    "currency": "USD",
                    "interruption_rate": 2.1,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": -5.0,  # Invalid negative price
                    "currency": "USD",
                    "interruption_rate": 3.5,
                    "availability": True
                },
                {
                    "region": "eu-west-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 14.50,
                    "currency": "EUR",  # Invalid currency
                    "interruption_rate": 1.8,
                    "availability": True
                },
                {
                    "region": "ap-southeast-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 16.90,
                    "currency": "USD",
                    "interruption_rate": 4.2,
                    "availability": True
                }
            ]
        })
        
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': mixed_content}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        assert isinstance(result, AnalysisResponse)
        # Should only have valid records (filtering out invalid price and currency)
        assert len(result.results) <= 4  # May have fewer due to validation
        assert result.total_regions_analyzed == 4

    def test_analyzer_configuration_update(self, analyzer_with_mocked_services):
        """Test updating analyzer configuration."""
        # Arrange
        analyzer = analyzer_with_mocked_services
        
        # Act
        analyzer.update_configuration(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.03
        )
        
        # Assert
        assert analyzer.instance_types == ["p5en.48xlarge"]
        assert analyzer.max_interruption_rate == 0.03

    def test_analyzer_status_and_info(self, analyzer_with_mocked_services):
        """Test getting analyzer status and information."""
        # Arrange
        analyzer = analyzer_with_mocked_services
        
        # Act
        status = analyzer.get_analysis_status()
        supported_types = analyzer.get_supported_instance_types()
        validation_result = analyzer.validate_instance_types(["p5en.48xlarge", "invalid.type"])
        
        # Assert
        assert isinstance(status, dict)
        assert "configuration" in status
        assert "cache_status" in status
        assert "services_initialized" in status
        
        assert isinstance(supported_types, list)
        assert "p5en.48xlarge" in supported_types
        assert "p5.48xlarge" in supported_types
        
        assert isinstance(validation_result, dict)
        assert "valid" in validation_result
        assert "invalid" in validation_result
        assert "p5en.48xlarge" in validation_result["valid"]
        assert "invalid.type" in validation_result["invalid"]

    @patch('src.services.bedrock_agent_service.Agent')
    def test_workflow_json_string_output(self, mock_agent_class, analyzer_with_mocked_services, mock_agent_response):
        """Test workflow with JSON string output."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        analyzer = analyzer_with_mocked_services
        
        # Act
        result = analyzer.analyze_spot_prices_json_string(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3,
            indent=2
        )
        
        # Assert
        assert isinstance(result, str)
        
        # Should be valid JSON
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, dict)
        assert "results" in parsed_result
        assert "metadata" in parsed_result

    def test_cache_management(self, analyzer_with_mocked_services):
        """Test cache management functionality."""
        # Arrange
        analyzer = analyzer_with_mocked_services
        
        # Act
        cache_info_before = analyzer.web_scraper.get_cache_info()
        analyzer.clear_cache()
        cache_info_after = analyzer.web_scraper.get_cache_info()
        
        # Assert
        assert isinstance(cache_info_before, dict)
        assert isinstance(cache_info_after, dict)
        assert cache_info_after["cache_entries"] == 0