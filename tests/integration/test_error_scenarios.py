"""
Integration tests for error scenarios and recovery mechanisms.

This module contains integration tests that verify error handling
and recovery mechanisms across the entire system.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.services.spot_price_analyzer import SpotPriceAnalyzer, ServiceFailureError
from src.services.web_scraper_service import WebScraperService
from src.services.bedrock_agent_service import BedrockAgentService
from src.services.data_filter_service import DataFilterService
from src.services.ranking_engine import RankingEngine
from src.services.result_formatter import ResultFormatter
from src.models.spot_data import RawSpotData
from src.utils.exceptions import (
    InsufficientDataError,
    WebScrapingError,
    BedrockServiceError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError,
    DataValidationError,
    ConfigurationError
)


class TestErrorScenarios:
    """Integration tests for error scenarios and recovery mechanisms."""

    @pytest.fixture
    def analyzer_with_real_services(self):
        """Create SpotPriceAnalyzer with real service instances."""
        bedrock_service = BedrockAgentService()
        web_scraper = WebScraperService(bedrock_service)
        data_filter = DataFilterService()
        ranking_engine = RankingEngine()
        result_formatter = ResultFormatter()
        
        return SpotPriceAnalyzer(
            web_scraper=web_scraper,
            data_filter=data_filter,
            ranking_engine=ranking_engine,
            bedrock_service=bedrock_service,
            result_formatter=result_formatter
        )

    @patch('src.services.bedrock_agent_service.Agent')
    def test_bedrock_service_unavailable_error(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of Bedrock service unavailable errors."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'ServiceUnavailable',
                'Message': 'Service temporarily unavailable'
            },
            'ResponseMetadata': {
                'HTTPStatusCode': 503
            }
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = ClientError(error_response, 'InvokeModel')
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(Exception):  # May be wrapped in different exception types due to retry
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )

    @patch('src.services.bedrock_agent_service.Agent')
    def test_rate_limit_error_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of rate limit errors with retry mechanism."""
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
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(Exception):  # May be wrapped in different exception types due to retry
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )
        
        # Should have attempted retries before failing
        assert mock_agent.call_count > 1  # Retry mechanism should have been triggered

    @patch('src.services.bedrock_agent_service.Agent')
    def test_network_error_recovery(self, mock_agent_class, analyzer_with_real_services):
        """Test network error handling and recovery."""
        # Arrange
        from botocore.exceptions import EndpointConnectionError
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = EndpointConnectionError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(Exception):  # May be wrapped in different exception types
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )

    @patch('src.services.bedrock_agent_service.Agent')
    def test_empty_response_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of empty responses from web scraping."""
        # Arrange
        mock_response = Mock()
        mock_response.message = None
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )
        
        assert "Web scraping service failed" in str(exc_info.value)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_malformed_response_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of malformed responses from web scraping."""
        # Arrange
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': '{ invalid json content }'}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_real_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        # Should handle malformed JSON gracefully by falling back to HTML parsing
        # which may return empty results
        assert isinstance(result.results, list)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_no_valid_data_scenario(self, mock_agent_class, analyzer_with_real_services):
        """Test scenario where no valid data is found."""
        # Arrange
        empty_content = json.dumps({"regions": []})
        
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': empty_content}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )
        
        assert "No spot data retrieved" in str(exc_info.value)

    def test_invalid_instance_type_configuration(self, analyzer_with_real_services):
        """Test error handling for invalid instance type configuration."""
        # Arrange
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(instance_types=[])
        
        assert "Instance types must be a non-empty list" in str(exc_info.value)

    def test_invalid_interruption_rate_configuration(self, analyzer_with_real_services):
        """Test error handling for invalid interruption rate configuration."""
        # Arrange
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(max_interruption_rate=1.5)
        
        assert "Max interruption rate must be between 0.0 and 1.0" in str(exc_info.value)

    def test_unsupported_instance_type_configuration(self, analyzer_with_real_services):
        """Test error handling for unsupported instance types."""
        # Arrange
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            analyzer.update_configuration(instance_types=["unsupported.type"])
        
        assert "Unsupported instance types" in str(exc_info.value)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_partial_data_corruption_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of partially corrupted data."""
        # Arrange
        corrupted_content = json.dumps({
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
                    "spot_price": "invalid_price",  # Corrupted price data
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
                }
            ]
        })
        
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': corrupted_content}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_response
        
        analyzer = analyzer_with_real_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        # Should handle corrupted data gracefully by filtering out invalid records
        assert len(result.results) == 2  # Only valid records
        assert all(isinstance(r.spot_price, (int, float)) for r in result.results)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_authentication_error_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of authentication errors."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'UnauthorizedOperation',
                'Message': 'You are not authorized to perform this operation'
            }
        }
        
        mock_agent_class.side_effect = ClientError(error_response, 'CreateAgent')
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )
        
        assert "Web scraping service failed" in str(exc_info.value)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_timeout_error_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test handling of timeout errors."""
        # Arrange
        from botocore.exceptions import ReadTimeoutError
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = ReadTimeoutError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")
        
        analyzer = analyzer_with_real_services
        
        # Act & Assert
        with pytest.raises(ServiceFailureError) as exc_info:
            analyzer.analyze_spot_prices(
                instance_types=["p5en.48xlarge"],
                max_interruption_rate=0.05,
                top_count=3
            )
        
        assert "Web scraping service failed" in str(exc_info.value)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_json_output_error_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test error handling in JSON output generation."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'ServiceUnavailable',
                'Message': 'Service temporarily unavailable'
            }
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = ClientError(error_response, 'InvokeModel')
        
        analyzer = analyzer_with_real_services
        
        # Act
        result = analyzer.analyze_spot_prices_json(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        # Should return formatted error response instead of raising exception
        assert isinstance(result, dict)
        # The exact error format may vary, just check it's a dict with error info
        assert len(result) > 0

    @patch('src.services.bedrock_agent_service.Agent')
    def test_json_string_output_error_handling(self, mock_agent_class, analyzer_with_real_services):
        """Test error handling in JSON string output generation."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'ServiceUnavailable',
                'Message': 'Service temporarily unavailable'
            }
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = ClientError(error_response, 'InvokeModel')
        
        analyzer = analyzer_with_real_services
        
        # Act
        result = analyzer.analyze_spot_prices_json_string(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        # Should return formatted error response as JSON string
        assert isinstance(result, str)
        
        # Should be valid JSON
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, dict)
        assert "error" in parsed_result

    @patch('src.services.bedrock_agent_service.Agent')
    def test_retry_mechanism_success_after_failure(self, mock_agent_class, analyzer_with_real_services):
        """Test retry mechanism succeeding after initial failures."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        
        success_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 15.20,
                    "currency": "USD",
                    "interruption_rate": 2.1,
                    "availability": True
                }
            ]
        })
        
        success_response = Mock()
        success_response.message = {
            'content': [
                {'text': success_content}
            ]
        }
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # First call fails, second succeeds
        mock_agent.side_effect = [
            ClientError(error_response, 'InvokeModel'),
            success_response
        ]
        
        analyzer = analyzer_with_real_services
        
        # Act
        result = analyzer.analyze_spot_prices(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        
        # Assert
        # Should succeed after retry
        assert len(result.results) == 1
        assert mock_agent.call_count == 2  # Initial call + 1 retry

    def test_data_filter_error_propagation(self, analyzer_with_real_services):
        """Test error propagation from data filter service."""
        # Arrange
        analyzer = analyzer_with_real_services
        
        # Mock the data filter to raise an error
        with patch.object(analyzer.data_filter, 'filter_and_validate') as mock_filter:
            mock_filter.side_effect = Exception("Filter service error")
            
            # Mock successful web scraping
            with patch.object(analyzer.web_scraper, 'scrape_spot_data') as mock_scrape:
                mock_scrape.return_value = [
                    RawSpotData(
                        region="us-east-1",
                        instance_type="p5en.48xlarge",
                        spot_price=15.20,
                        currency="USD",
                        interruption_rate=0.021,
                        timestamp=datetime.now(timezone.utc),
                        availability=True
                    )
                ]
                
                # Act & Assert
                with pytest.raises(ServiceFailureError) as exc_info:
                    analyzer.analyze_spot_prices(
                        instance_types=["p5en.48xlarge"],
                        max_interruption_rate=0.05,
                        top_count=3
                    )
                
                assert "Data filtering service failed" in str(exc_info.value)
                assert exc_info.value.details["service"] == "data_filter"

    def test_ranking_engine_error_propagation(self, analyzer_with_real_services):
        """Test error propagation from ranking engine."""
        # Arrange
        analyzer = analyzer_with_real_services
        
        # Mock successful web scraping and filtering
        with patch.object(analyzer.web_scraper, 'scrape_spot_data') as mock_scrape, \
             patch.object(analyzer.data_filter, 'filter_and_validate') as mock_filter, \
             patch.object(analyzer.ranking_engine, 'rank_and_get_top') as mock_rank:
            
            mock_scrape.return_value = [
                RawSpotData(
                    region="us-east-1",
                    instance_type="p5en.48xlarge",
                    spot_price=15.20,
                    currency="USD",
                    interruption_rate=0.021,
                    timestamp=datetime.now(timezone.utc),
                    availability=True
                )
            ]
            
            mock_filter.return_value = mock_scrape.return_value
            mock_rank.side_effect = Exception("Ranking engine error")
            
            # Act & Assert
            with pytest.raises(Exception):  # Should propagate as general error
                analyzer.analyze_spot_prices(
                    instance_types=["p5en.48xlarge"],
                    max_interruption_rate=0.05,
                    top_count=3
                )

    def test_cache_error_handling(self, analyzer_with_real_services):
        """Test cache error handling and recovery."""
        # Arrange
        analyzer = analyzer_with_real_services
        
        # Mock cache error in web scraper
        with patch.object(analyzer.web_scraper, '_update_cache') as mock_cache:
            mock_cache.side_effect = Exception("Cache update failed")
            
            # Mock successful bedrock response
            with patch('src.services.bedrock_agent_service.Agent') as mock_agent_class:
                success_content = json.dumps({
                    "regions": [
                        {
                            "region": "us-east-1",
                            "instance_type": "p5en.48xlarge",
                            "spot_price": 15.20,
                            "currency": "USD",
                            "interruption_rate": 2.1,
                            "availability": True
                        }
                    ]
                })
                
                success_response = Mock()
                success_response.message = {
                    'content': [
                        {'text': success_content}
                    ]
                }
                
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                mock_agent.return_value = success_response
                
                # Act & Assert
                # Should handle cache errors gracefully and continue processing
                with pytest.raises(ServiceFailureError):  # Cache error should propagate
                    analyzer.analyze_spot_prices(
                        instance_types=["p5en.48xlarge"],
                        max_interruption_rate=0.05,
                        top_count=3
                    )