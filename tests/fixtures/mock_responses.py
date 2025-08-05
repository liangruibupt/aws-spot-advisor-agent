"""
Mock responses and test data for integration tests.

This module provides mock responses and test data that simulate
real AWS Bedrock AgentCore responses and web page content.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock

from src.models.spot_data import RawSpotData


class MockResponses:
    """Collection of mock responses for testing."""

    @staticmethod
    def get_successful_spot_data_response() -> str:
        """Get mock successful spot data response."""
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
                    "interruption_rate": 6.2,  # High interruption rate
                    "availability": True
                }
            ]
        })

    @staticmethod
    def get_high_interruption_response() -> str:
        """Get mock response with all high interruption rates."""
        return json.dumps({
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 15.20,
                    "currency": "USD",
                    "interruption_rate": 8.5,
                    "availability": True
                },
                {
                    "region": "us-west-2",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.80,
                    "currency": "USD",
                    "interruption_rate": 7.2,
                    "availability": True
                }
            ]
        })

    @staticmethod
    def get_limited_regions_response() -> str:
        """Get mock response with limited valid regions."""
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
                    "interruption_rate": 8.2,  # High interruption rate
                    "availability": True
                }
            ]
        })

    @staticmethod
    def get_corrupted_data_response() -> str:
        """Get mock response with corrupted data."""
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
                    "spot_price": "invalid_price",  # Corrupted price
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
                }
            ]
        })

    @staticmethod
    def get_empty_response() -> str:
        """Get mock empty response."""
        return json.dumps({"regions": []})

    @staticmethod
    def get_malformed_json_response() -> str:
        """Get malformed JSON response."""
        return '{ "regions": [ invalid json content'

    @staticmethod
    def get_html_fallback_content() -> str:
        """Get mock HTML content for fallback parsing."""
        return """
        <html>
        <head><title>AWS EC2 Spot Instance Advisor</title></head>
        <body>
            <div class="spot-advisor-table">
                <table>
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Instance Type</th>
                            <th>Spot Price</th>
                            <th>Interruption Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>us-east-1</td>
                            <td>p5en.48xlarge</td>
                            <td>$15.20</td>
                            <td>2.1%</td>
                        </tr>
                        <tr>
                            <td>us-west-2</td>
                            <td>p5en.48xlarge</td>
                            <td>$12.80</td>
                            <td>3.5%</td>
                        </tr>
                        <tr>
                            <td>eu-west-1</td>
                            <td>p5.48xlarge</td>
                            <td>$13.90</td>
                            <td>1.9%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """

    @staticmethod
    def create_mock_agent_response(content: str) -> Mock:
        """Create mock agent response with given content."""
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'text': content}
            ]
        }
        return mock_response

    @staticmethod
    def create_empty_mock_agent_response() -> Mock:
        """Create mock agent response with no content."""
        mock_response = Mock()
        mock_response.message = None
        return mock_response

    @staticmethod
    def create_malformed_mock_agent_response() -> Mock:
        """Create mock agent response with malformed content."""
        mock_response = Mock()
        mock_response.message = {
            'content': [
                {'invalid_key': 'no text field'}
            ]
        }
        return mock_response


class TestDataFactory:
    """Factory for creating test data objects."""

    @staticmethod
    def create_raw_spot_data_list() -> List[RawSpotData]:
        """Create list of RawSpotData objects for testing."""
        current_time = datetime.now(timezone.utc)
        
        return [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=15.20,
                currency="USD",
                interruption_rate=0.021,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=12.80,
                currency="USD",
                interruption_rate=0.035,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=14.50,
                currency="USD",
                interruption_rate=0.018,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="ap-southeast-1",
                instance_type="p5en.48xlarge",
                spot_price=16.90,
                currency="USD",
                interruption_rate=0.042,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="us-east-1",
                instance_type="p5.48xlarge",
                spot_price=14.80,
                currency="USD",
                interruption_rate=0.023,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=12.20,
                currency="USD",
                interruption_rate=0.038,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5.48xlarge",
                spot_price=13.90,
                currency="USD",
                interruption_rate=0.019,
                timestamp=current_time,
                availability=True
            )
        ]

    @staticmethod
    def create_high_interruption_data_list() -> List[RawSpotData]:
        """Create list of RawSpotData with high interruption rates."""
        current_time = datetime.now(timezone.utc)
        
        return [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=15.20,
                currency="USD",
                interruption_rate=0.085,  # 8.5%
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=12.80,
                currency="USD",
                interruption_rate=0.072,  # 7.2%
                timestamp=current_time,
                availability=True
            )
        ]

    @staticmethod
    def create_mixed_validity_data_list() -> List[RawSpotData]:
        """Create list of RawSpotData with mixed validity."""
        current_time = datetime.now(timezone.utc)
        
        return [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=15.20,
                currency="USD",
                interruption_rate=0.021,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=-5.0,  # Invalid negative price
                currency="USD",
                interruption_rate=0.035,
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=14.50,
                currency="EUR",  # Invalid currency
                interruption_rate=0.018,
                timestamp=current_time,
                availability=True
            )
        ]


class ErrorScenarios:
    """Collection of error scenarios for testing."""

    @staticmethod
    def get_throttling_error():
        """Get throttling error for testing."""
        from botocore.exceptions import ClientError
        return ClientError(
            {
                'Error': {
                    'Code': 'ThrottlingException',
                    'Message': 'Rate exceeded'
                }
            },
            'InvokeModel'
        )

    @staticmethod
    def get_service_unavailable_error():
        """Get service unavailable error for testing."""
        from botocore.exceptions import ClientError
        return ClientError(
            {
                'Error': {
                    'Code': 'ServiceUnavailable',
                    'Message': 'Service temporarily unavailable'
                },
                'ResponseMetadata': {
                    'HTTPStatusCode': 503
                }
            },
            'InvokeModel'
        )

    @staticmethod
    def get_unauthorized_error():
        """Get unauthorized error for testing."""
        from botocore.exceptions import ClientError
        return ClientError(
            {
                'Error': {
                    'Code': 'UnauthorizedOperation',
                    'Message': 'You are not authorized to perform this operation'
                }
            },
            'CreateAgent'
        )

    @staticmethod
    def get_network_error():
        """Get network error for testing."""
        from botocore.exceptions import EndpointConnectionError
        return EndpointConnectionError(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        )

    @staticmethod
    def get_timeout_error():
        """Get timeout error for testing."""
        from botocore.exceptions import ReadTimeoutError
        return ReadTimeoutError(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        )


class MockWebScrapingScenarios:
    """Collection of web scraping scenarios for testing."""

    @staticmethod
    def successful_scraping_scenario():
        """Scenario for successful web scraping."""
        return {
            'response_content': MockResponses.get_successful_spot_data_response(),
            'expected_regions': 7,  # Excluding high interruption rate region
            'expected_instance_types': ['p5en.48xlarge', 'p5.48xlarge']
        }

    @staticmethod
    def insufficient_regions_scenario():
        """Scenario with insufficient regions meeting criteria."""
        return {
            'response_content': MockResponses.get_high_interruption_response(),
            'expected_error': 'InsufficientDataError',
            'max_interruption_rate': 0.05
        }

    @staticmethod
    def partial_results_scenario():
        """Scenario with partial results."""
        return {
            'response_content': MockResponses.get_limited_regions_response(),
            'expected_regions': 2,
            'expected_warnings': True
        }

    @staticmethod
    def data_corruption_scenario():
        """Scenario with corrupted data."""
        return {
            'response_content': MockResponses.get_corrupted_data_response(),
            'expected_regions': 1,  # Only valid record
            'expected_filtering': True
        }

    @staticmethod
    def empty_response_scenario():
        """Scenario with empty response."""
        return {
            'response_content': MockResponses.get_empty_response(),
            'expected_error': 'ServiceFailureError'
        }

    @staticmethod
    def malformed_json_scenario():
        """Scenario with malformed JSON."""
        return {
            'response_content': MockResponses.get_malformed_json_response(),
            'fallback_to_html': True,
            'expected_regions': 0  # HTML parsing may not extract data
        }