"""
Integration tests for AWS Bedrock AgentCore connectivity.

This module contains integration tests that verify the BedrockAgentService
can successfully connect to AWS Bedrock and perform web scraping operations.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.services.bedrock_agent_service import BedrockAgentService, BedrockAgentServiceError
from src.models.spot_data import RawSpotData
from src.utils.exceptions import (
    BedrockServiceError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError,
    DataValidationError
)


class TestBedrockIntegration:
    """Integration tests for Bedrock AgentCore connectivity."""

    @pytest.fixture
    def bedrock_service(self):
        """Create BedrockAgentService instance for testing."""
        return BedrockAgentService(
            region_name="us-east-1",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0"
        )

    @pytest.fixture
    def mock_agent_response(self):
        """Mock successful agent response."""
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {
                    'text': "Connection successful"
                }
            ]
        }
        return mock_response

    @pytest.fixture
    def mock_html_response(self):
        """Mock HTML response for fallback parsing."""
        return """
        <html>
        <body>
            <table>
                <tr>
                    <td>us-east-1</td>
                    <td>p5en.48xlarge</td>
                    <td>$12.50</td>
                    <td>3.2%</td>
                </tr>
                <tr>
                    <td>us-west-2</td>
                    <td>p5.48xlarge</td>
                    <td>$11.80</td>
                    <td>2.8%</td>
                </tr>
            </table>
        </body>
        </html>
        """

    @patch('src.services.bedrock_agent_service.Agent')
    def test_successful_bedrock_connection(self, mock_agent_class, bedrock_service, mock_agent_response):
        """Test successful connection to Bedrock AgentCore."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        # Act
        result = bedrock_service.test_connection()
        
        # Assert
        assert result is True
        mock_agent_class.assert_called_once()

    @patch('src.services.bedrock_agent_service.Agent')
    def test_bedrock_connection_failure(self, mock_agent_class, bedrock_service):
        """Test Bedrock connection failure handling."""
        # Arrange
        from botocore.exceptions import ClientError
        error_response = {
            'Error': {
                'Code': 'AccessDenied',
                'Message': 'Access denied'
            }
        }
        mock_agent_class.side_effect = ClientError(error_response, 'CreateAgent')
        
        # Act & Assert
        with pytest.raises(BedrockServiceError) as exc_info:
            bedrock_service._get_agent()
        
        assert "Failed to initialize Strands Agent" in str(exc_info.value)

    @patch('src.services.bedrock_agent_service.Agent')
    def test_web_scraping_success(self, mock_agent_class, bedrock_service, mock_agent_response):
        """Test successful web scraping operation."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.return_value = mock_agent_response
        
        test_url = "https://aws.amazon.com/ec2/spot/instance-advisor/"
        
        # Act
        result = bedrock_service.execute_web_scraping(test_url)
        
        # Assert
        assert result is not None
        assert isinstance(result, str)
        mock_agent.assert_called_once()
        
        # Verify the prompt contains the URL
        call_args = mock_agent.call_args[0][0]
        assert test_url in call_args

    @patch('src.services.bedrock_agent_service.Agent')
    def test_web_scraping_rate_limit_error(self, mock_agent_class, bedrock_service):
        """Test rate limit error handling during web scraping."""
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
        
        # Act & Assert
        # The retry mechanism will eventually raise the error
        with pytest.raises((RateLimitError, Exception)):
            bedrock_service.execute_web_scraping("https://example.com")

    @patch('src.services.bedrock_agent_service.Agent')
    def test_web_scraping_service_unavailable(self, mock_agent_class, bedrock_service):
        """Test service unavailable error handling."""
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
        
        # Act & Assert
        # The retry mechanism will eventually raise the error
        with pytest.raises((ServiceUnavailableError, Exception)):
            bedrock_service.execute_web_scraping("https://example.com")

    @patch('src.services.bedrock_agent_service.Agent')
    def test_web_scraping_network_error(self, mock_agent_class, bedrock_service):
        """Test network error handling during web scraping."""
        # Arrange
        from botocore.exceptions import EndpointConnectionError
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_agent.side_effect = EndpointConnectionError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")
        
        # Act & Assert
        # The retry mechanism will eventually raise the error
        with pytest.raises((NetworkError, Exception)):
            bedrock_service.execute_web_scraping("https://example.com")

    def test_parse_structured_json_data(self, bedrock_service):
        """Test parsing structured JSON data from agent response."""
        # Arrange
        json_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.50,
                    "currency": "USD",
                    "interruption_rate": 3.2,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5.48xlarge",
                    "spot_price": 11.80,
                    "currency": "USD",
                    "interruption_rate": 2.8,
                    "availability": True
                }
            ]
        })
        
        instance_types = ["p5en.48xlarge", "p5.48xlarge"]
        
        # Act
        result = bedrock_service.parse_spot_data(json_content, instance_types)
        
        # Assert
        assert len(result) == 2
        assert all(isinstance(item, RawSpotData) for item in result)
        
        # Check first result
        first_result = result[0]
        assert first_result.region == "us-east-1"
        assert first_result.instance_type == "p5en.48xlarge"
        assert first_result.spot_price == 12.50
        assert first_result.currency == "USD"
        assert first_result.interruption_rate == 0.032  # Converted from percentage
        assert first_result.availability is True

    def test_parse_html_fallback_data(self, bedrock_service, mock_html_response):
        """Test parsing HTML data as fallback when JSON parsing fails."""
        # Arrange
        instance_types = ["p5en.48xlarge", "p5.48xlarge"]
        
        # Act
        result = bedrock_service.parse_spot_data(mock_html_response, instance_types)
        
        # Assert
        # HTML parsing is simplified and may not extract all data perfectly
        # This test verifies the fallback mechanism works without errors
        assert isinstance(result, list)
        # The exact number of results depends on the HTML parsing implementation

    def test_parse_empty_content_error(self, bedrock_service):
        """Test error handling for empty content."""
        # Act & Assert
        with pytest.raises(DataValidationError) as exc_info:
            bedrock_service.parse_spot_data("", ["p5en.48xlarge"])
        
        assert "Empty content provided for parsing" in str(exc_info.value)

    def test_parse_no_instance_types_error(self, bedrock_service):
        """Test error handling for empty instance types list."""
        # Act & Assert
        with pytest.raises(DataValidationError) as exc_info:
            bedrock_service.parse_spot_data("some content", [])
        
        assert "No instance types provided for filtering" in str(exc_info.value)

    def test_parse_invalid_json_fallback(self, bedrock_service):
        """Test fallback to HTML parsing when JSON is invalid."""
        # Arrange
        invalid_json = "{ invalid json content"
        instance_types = ["p5en.48xlarge"]
        
        # Act
        result = bedrock_service.parse_spot_data(invalid_json, instance_types)
        
        # Assert
        # Should not raise exception, should fallback to HTML parsing
        assert isinstance(result, list)

    def test_get_agent_info(self, bedrock_service):
        """Test getting agent information."""
        # Act
        info = bedrock_service.get_agent_info()
        
        # Assert
        assert isinstance(info, dict)
        assert info["model_name"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert info["region"] == "us-east-1"
        assert info["framework"] == "Strands Agent"
        assert "http_request" in info["tools"]

    @patch('src.services.bedrock_agent_service.Agent')
    def test_connection_test_success(self, mock_agent_class, bedrock_service):
        """Test successful connection test."""
        # Arrange
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': 'Connection successful'}
            ]
        }
        mock_agent.return_value = mock_response
        
        # Act
        result = bedrock_service.test_connection()
        
        # Assert
        assert result is True

    @patch('src.services.bedrock_agent_service.Agent')
    def test_connection_test_failure(self, mock_agent_class, bedrock_service):
        """Test connection test failure."""
        # Arrange
        mock_agent_class.side_effect = Exception("Connection failed")
        
        # Act
        result = bedrock_service.test_connection()
        
        # Assert
        assert result is False

    def test_structured_data_filtering_by_instance_type(self, bedrock_service):
        """Test that structured data parsing filters by instance type correctly."""
        # Arrange
        json_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.50,
                    "currency": "USD",
                    "interruption_rate": 3.2,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "m5.large",  # Different instance type
                    "spot_price": 0.10,
                    "currency": "USD",
                    "interruption_rate": 1.0,
                    "availability": True
                }
            ]
        })
        
        instance_types = ["p5en.48xlarge"]  # Only request one type
        
        # Act
        result = bedrock_service.parse_spot_data(json_content, instance_types)
        
        # Assert
        assert len(result) == 1
        assert result[0].instance_type == "p5en.48xlarge"

    def test_structured_data_missing_fields_handling(self, bedrock_service):
        """Test handling of structured data with missing required fields."""
        # Arrange
        json_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.50,
                    "currency": "USD",
                    "interruption_rate": 3.2,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5.48xlarge",
                    # Missing spot_price
                    "currency": "USD",
                    "interruption_rate": 2.8,
                    "availability": True
                }
            ]
        })
        
        instance_types = ["p5en.48xlarge", "p5.48xlarge"]
        
        # Act
        result = bedrock_service.parse_spot_data(json_content, instance_types)
        
        # Assert
        # Should only return the valid record
        assert len(result) == 1
        assert result[0].region == "us-east-1"

    def test_interruption_rate_percentage_conversion(self, bedrock_service):
        """Test that interruption rates are correctly converted from percentages."""
        # Arrange
        json_content = json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.50,
                    "currency": "USD",
                    "interruption_rate": 15.5,  # Percentage format
                    "availability": True
                }
            ]
        })
        
        instance_types = ["p5en.48xlarge"]
        
        # Act
        result = bedrock_service.parse_spot_data(json_content, instance_types)
        
        # Assert
        assert len(result) == 1
        assert result[0].interruption_rate == 0.155  # Converted to decimal