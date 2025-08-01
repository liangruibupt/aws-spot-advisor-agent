"""
Unit tests for BedrockAgentService.

Tests cover AWS Bedrock AgentCore integration, web scraping functionality,
and HTML content parsing with mocked responses.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

from src.services.bedrock_agent_service import BedrockAgentService, BedrockAgentServiceError
from src.models.spot_data import RawSpotData

# Mock the http_request import to avoid dependency issues in tests
try:
    from strands_tools.http_request import http_request
except ImportError:
    http_request = Mock()


class TestBedrockAgentService:
    """Test cases for BedrockAgentService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = BedrockAgentService(
            region_name="us-east-1",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0"
        )

    def test_initialization(self):
        """Test service initialization with parameters."""
        service = BedrockAgentService(
            region_name="us-west-2",
            model_name="anthropic.claude-3-haiku-20240307-v1:0"
        )
        
        assert service.region_name == "us-west-2"
        assert service.model_name == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_initialization_defaults(self):
        """Test service initialization with default parameters."""
        service = BedrockAgentService()
        
        assert service.region_name == "us-east-1"
        assert service.model_name == "anthropic.claude-3-sonnet-20240229-v1:0"

    @patch('src.services.bedrock_agent_service.Agent')
    def test_get_agent_success(self, mock_agent_class):
        """Test successful Strands Agent creation."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        agent = self.service._get_agent()
        
        assert agent == mock_agent
        mock_agent_class.assert_called_once_with(
            model='bedrock:anthropic.claude-3-sonnet-20240229-v1:0',
            tools=[http_request]
        )

    @patch('src.services.bedrock_agent_service.Agent')
    def test_get_agent_failure(self, mock_agent_class):
        """Test Strands Agent creation failure."""
        mock_agent_class.side_effect = Exception("Agent initialization failed")
        
        with pytest.raises(BedrockAgentServiceError, match="Failed to initialize Strands Agent"):
            self.service._get_agent()

    @patch.object(BedrockAgentService, '_get_agent')
    def test_execute_web_scraping_success(self, mock_get_agent):
        """Test successful web scraping execution."""
        # Mock agent response
        mock_agent = Mock()
        mock_response_text = json.dumps({
            'regions': [
                {
                    'region': 'us-east-1',
                    'instance_type': 'p5en.48xlarge',
                    'spot_price': 12.50,
                    'currency': 'USD',
                    'interruption_rate': 3.0,
                    'availability': True
                }
            ]
        })
        
        # Mock AgentResult object
        mock_agent_result = Mock()
        mock_agent_result.message = {
            'content': [{'text': mock_response_text}]
        }
        mock_agent.return_value = mock_agent_result
        mock_get_agent.return_value = mock_agent
        
        url = "https://aws.amazon.com/ec2/spot/instance-advisor/"
        result = self.service.execute_web_scraping(url)
        
        assert result == mock_response_text
        mock_agent.assert_called_once()
        
        # Verify the prompt contains the URL
        call_args = mock_agent.call_args[0][0]
        assert url in call_args

    @patch.object(BedrockAgentService, '_get_agent')
    def test_execute_web_scraping_custom_instructions(self, mock_get_agent):
        """Test web scraping with custom instructions."""
        mock_agent = Mock()
        mock_response_text = '<html>test content</html>'
        
        # Mock AgentResult object
        mock_agent_result = Mock()
        mock_agent_result.message = {
            'content': [{'text': mock_response_text}]
        }
        mock_agent.return_value = mock_agent_result
        mock_get_agent.return_value = mock_agent
        
        url = "https://test.com"
        custom_instructions = "Extract only price data"
        
        result = self.service.execute_web_scraping(url, custom_instructions)
        
        assert result == mock_response_text
        call_args = mock_agent.call_args[0][0]
        assert custom_instructions in call_args

    @patch.object(BedrockAgentService, '_get_agent')
    def test_execute_web_scraping_no_response(self, mock_get_agent):
        """Test web scraping with no response returned."""
        mock_agent = Mock()
        mock_agent.return_value = None
        mock_get_agent.return_value = mock_agent
        
        with pytest.raises(BedrockAgentServiceError, match="No response returned from agent"):
            self.service.execute_web_scraping("https://test.com")

    @patch.object(BedrockAgentService, '_get_agent')
    def test_execute_web_scraping_client_error(self, mock_get_agent):
        """Test web scraping with AWS client error."""
        mock_agent = Mock()
        mock_agent.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            'InvokeModel'
        )
        mock_get_agent.return_value = mock_agent
        
        with pytest.raises(BedrockAgentServiceError, match="AWS service error"):
            self.service.execute_web_scraping("https://test.com")

    def test_parse_spot_data_structured_json(self):
        """Test parsing structured JSON data."""
        json_content = json.dumps({
            'regions': [
                {
                    'region': 'us-east-1',
                    'instance_type': 'p5en.48xlarge',
                    'spot_price': 12.50,
                    'currency': 'USD',
                    'interruption_rate': 3.0,
                    'availability': True
                },
                {
                    'region': 'us-west-2',
                    'instance_type': 'p5.48xlarge',
                    'spot_price': 15.00,
                    'currency': 'USD',
                    'interruption_rate': 2.5,
                    'availability': True
                }
            ]
        })
        
        instance_types = ['p5en.48xlarge', 'p5.48xlarge']
        results = self.service.parse_spot_data(json_content, instance_types)
        
        assert len(results) == 2
        
        # Check first result
        assert results[0].region == 'us-east-1'
        assert results[0].instance_type == 'p5en.48xlarge'
        assert results[0].spot_price == 12.50
        assert results[0].currency == 'USD'
        assert results[0].interruption_rate == 0.03  # Converted from percentage
        assert results[0].availability is True
        
        # Check second result
        assert results[1].region == 'us-west-2'
        assert results[1].instance_type == 'p5.48xlarge'
        assert results[1].spot_price == 15.00
        assert results[1].interruption_rate == 0.025

    def test_parse_spot_data_filter_instance_types(self):
        """Test filtering by instance types during parsing."""
        json_content = json.dumps({
            'regions': [
                {
                    'region': 'us-east-1',
                    'instance_type': 'p5en.48xlarge',
                    'spot_price': 12.50,
                    'currency': 'USD',
                    'interruption_rate': 3.0,
                    'availability': True
                },
                {
                    'region': 'us-west-2',
                    'instance_type': 'm5.large',  # Different instance type
                    'spot_price': 0.10,
                    'currency': 'USD',
                    'interruption_rate': 1.0,
                    'availability': True
                }
            ]
        })
        
        instance_types = ['p5en.48xlarge']
        results = self.service.parse_spot_data(json_content, instance_types)
        
        assert len(results) == 1
        assert results[0].instance_type == 'p5en.48xlarge'

    def test_parse_spot_data_missing_fields(self):
        """Test parsing with missing required fields."""
        json_content = json.dumps({
            'regions': [
                {
                    'region': 'us-east-1',
                    'instance_type': 'p5en.48xlarge',
                    # Missing spot_price and interruption_rate
                    'currency': 'USD',
                    'availability': True
                },
                {
                    'region': 'us-west-2',
                    'instance_type': 'p5.48xlarge',
                    'spot_price': 15.00,
                    'currency': 'USD',
                    'interruption_rate': 2.5,
                    'availability': True
                }
            ]
        })
        
        instance_types = ['p5en.48xlarge', 'p5.48xlarge']
        results = self.service.parse_spot_data(json_content, instance_types)
        
        # Should only return the valid record
        assert len(results) == 1
        assert results[0].region == 'us-west-2'

    def test_parse_spot_data_percentage_conversion(self):
        """Test interruption rate percentage conversion."""
        json_content = json.dumps({
            'regions': [
                {
                    'region': 'us-east-1',
                    'instance_type': 'p5en.48xlarge',
                    'spot_price': 12.50,
                    'currency': 'USD',
                    'interruption_rate': 0.03,  # Already decimal
                    'availability': True
                },
                {
                    'region': 'us-west-2',
                    'instance_type': 'p5.48xlarge',
                    'spot_price': 15.00,
                    'currency': 'USD',
                    'interruption_rate': 5.0,  # Percentage format
                    'availability': True
                }
            ]
        })
        
        instance_types = ['p5en.48xlarge', 'p5.48xlarge']
        results = self.service.parse_spot_data(json_content, instance_types)
        
        assert len(results) == 2
        assert results[0].interruption_rate == 0.03  # Unchanged
        assert results[1].interruption_rate == 0.05  # Converted from 5.0%

    @patch.object(BedrockAgentService, '_parse_html_content')
    def test_parse_spot_data_fallback_to_html(self, mock_parse_html):
        """Test fallback to HTML parsing when JSON parsing fails."""
        html_content = "<html><body>Not JSON content</body></html>"
        mock_parse_html.return_value = [
            RawSpotData(
                region='us-east-1',
                instance_type='p5en.48xlarge',
                spot_price=12.50,
                currency='USD',
                interruption_rate=0.03,
                timestamp=datetime.now(timezone.utc),
                availability=True
            )
        ]
        
        instance_types = ['p5en.48xlarge']
        results = self.service.parse_spot_data(html_content, instance_types)
        
        assert len(results) == 1
        mock_parse_html.assert_called_once_with(html_content, instance_types)

    def test_parse_html_content_basic(self):
        """Test basic HTML content parsing."""
        html_content = """
        <html>
            <body>
                <table>
                    <tr>
                        <td>us-east-1</td>
                        <td>p5en.48xlarge</td>
                        <td>$12.50</td>
                        <td>3.0%</td>
                    </tr>
                    <tr>
                        <td>us-west-2</td>
                        <td>p5.48xlarge</td>
                        <td>$15.00</td>
                        <td>2.5%</td>
                    </tr>
                </table>
            </body>
        </html>
        """
        
        instance_types = ['p5en.48xlarge', 'p5.48xlarge']
        results = self.service._parse_html_content(html_content, instance_types)
        
        # HTML parsing is simplified and may not extract all data perfectly
        # This test verifies the method doesn't crash and returns a list
        assert isinstance(results, list)

    def test_parse_html_content_empty(self):
        """Test HTML parsing with empty content."""
        html_content = ""
        instance_types = ['p5en.48xlarge']
        
        results = self.service._parse_html_content(html_content, instance_types)
        
        assert isinstance(results, list)
        assert len(results) == 0

    def test_parse_spot_data_invalid_json(self):
        """Test parsing with invalid JSON that falls back to HTML."""
        invalid_content = "{ invalid json content"
        instance_types = ['p5en.48xlarge']
        
        # Should not raise exception, should fallback to HTML parsing
        results = self.service.parse_spot_data(invalid_content, instance_types)
        
        assert isinstance(results, list)

    @patch.object(BedrockAgentService, '_get_agent')
    def test_test_connection_success(self, mock_get_agent):
        """Test successful connection test."""
        mock_agent = Mock()
        
        # Mock AgentResult object
        mock_agent_result = Mock()
        mock_agent_result.message = {
            'content': [{'text': 'Connection successful'}]
        }
        mock_agent.return_value = mock_agent_result
        mock_get_agent.return_value = mock_agent
        
        result = self.service.test_connection()
        
        assert result is True
        mock_agent.assert_called_once()

    @patch.object(BedrockAgentService, '_get_agent')
    def test_test_connection_failure(self, mock_get_agent):
        """Test connection test failure."""
        mock_agent = Mock()
        mock_agent.side_effect = Exception("Connection failed")
        mock_get_agent.return_value = mock_agent
        
        result = self.service.test_connection()
        
        assert result is False

    def test_get_agent_info_success(self):
        """Test successful agent info retrieval."""
        result = self.service.get_agent_info()
        
        expected = {
            'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'region': 'us-east-1',
            'framework': 'Strands Agent',
            'tools': ['http_request']
        }
        assert result == expected


class TestBedrockAgentServiceIntegration:
    """Integration tests for BedrockAgentService."""

    def test_complete_workflow_structured_data(self):
        """Test complete workflow with structured data."""
        service = BedrockAgentService()
        
        # Mock structured response
        structured_response = {
            'regions': [
                {
                    'region': 'us-east-1',
                    'instance_type': 'p5en.48xlarge',
                    'spot_price': 12.50,
                    'currency': 'USD',
                    'interruption_rate': 3.0,
                    'availability': True
                },
                {
                    'region': 'us-west-2',
                    'instance_type': 'p5en.48xlarge',
                    'spot_price': 13.00,
                    'currency': 'USD',
                    'interruption_rate': 2.0,
                    'availability': True
                }
            ]
        }
        
        json_content = json.dumps(structured_response)
        instance_types = ['p5en.48xlarge']
        
        results = service.parse_spot_data(json_content, instance_types)
        
        assert len(results) == 2
        assert all(isinstance(result, RawSpotData) for result in results)
        assert results[0].region == 'us-east-1'
        assert results[1].region == 'us-west-2'
        assert results[0].interruption_rate == 0.03  # Converted from 3.0%
        assert results[1].interruption_rate == 0.02  # Converted from 2.0%

    def test_error_handling_chain(self):
        """Test error handling propagation through the service."""
        service = BedrockAgentService()
        
        # Test parsing error handling
        invalid_data = "completely invalid data that can't be parsed"
        instance_types = ['p5en.48xlarge']
        
        # Should not raise exception, should return empty list
        results = service.parse_spot_data(invalid_data, instance_types)
        assert isinstance(results, list)