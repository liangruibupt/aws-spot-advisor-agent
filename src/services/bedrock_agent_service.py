"""
AWS Bedrock AgentCore service wrapper for web scraping functionality.

This module provides a wrapper around AWS Bedrock AgentCore to handle
web scraping operations for spot price data extraction using the Strands framework.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import asdict

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands_tools.http_request import http_request, extract_content_from_html

from src.models.spot_data import RawSpotData


logger = logging.getLogger(__name__)


class BedrockAgentServiceError(Exception):
    """Base exception for BedrockAgentService errors."""
    pass


class BedrockAgentService:
    """
    Service wrapper for AWS Bedrock AgentCore interactions.
    
    This service handles web scraping operations using the Strands Agent framework
    and provides methods for parsing spot price data from HTML content.
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    ):
        """
        Initialize the Bedrock Agent Service.
        
        Args:
            region_name: AWS region for Bedrock service
            model_name: Bedrock model to use for content analysis
        """
        self.region_name = region_name
        self.model_name = model_name
        
        # Initialize Strands agent
        self._agent: Optional[Agent] = None
        
        # Web scraping instructions template
        self._scraping_instructions = """
        Please analyze the AWS EC2 Spot Instance Advisor web page content and extract spot pricing data.
        
        For each region and instance type combination, extract:
        1. Region name (e.g., us-east-1, eu-west-1)
        2. Instance type (p5en.48xlarge or p5.48xlarge)
        3. Current spot price in USD
        4. Interruption rate as a percentage
        5. Availability status
        
        Focus specifically on p5en.48xlarge and p5.48xlarge instance types.
        Return the data in a structured JSON format with the following schema:
        {
            "regions": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": 12.50,
                    "currency": "USD",
                    "interruption_rate": 0.03,
                    "availability": true
                }
            ]
        }
        """

    def _get_agent(self) -> Agent:
        """Get or create Strands Agent instance."""
        if self._agent is None:
            try:
                # Create agent with Bedrock model
                self._agent = Agent(
                    model=f"bedrock:{self.model_name}",
                    tools=[http_request]
                )
            except Exception as e:
                logger.error(f"Failed to create Strands Agent: {e}")
                raise BedrockAgentServiceError(f"Failed to initialize Strands Agent: {e}")
        
        # At this point, self._agent is guaranteed to not be None
        assert self._agent is not None
        return self._agent

    def execute_web_scraping(self, url: str, custom_instructions: Optional[str] = None) -> str:
        """
        Execute web scraping using Strands Agent with HTTP request tool.
        
        Args:
            url: URL to scrape
            custom_instructions: Optional custom scraping instructions
            
        Returns:
            Raw HTML content or structured data from the agent
            
        Raises:
            BedrockAgentServiceError: If scraping fails
        """
        instructions = custom_instructions or self._scraping_instructions
        
        try:
            logger.info(f"Starting web scraping for URL: {url}")
            
            # Get the Strands agent
            agent = self._get_agent()
            
            # Create the prompt for the agent
            prompt = f"""
            Please fetch the content from this URL: {url}
            
            Then analyze the content according to these instructions:
            {instructions}
            
            Use the http_request tool to fetch the URL content first, then analyze it.
            """
            
            # Execute the agent with the prompt
            response = agent(prompt)
            
            if not response or not response.message:
                raise BedrockAgentServiceError("No response returned from agent")
            
            # Extract text content from the response
            content = ""
            if isinstance(response.message, dict) and 'content' in response.message:
                for item in response.message['content']:
                    if isinstance(item, dict) and 'text' in item:
                        content += item['text']
            
            if not content:
                raise BedrockAgentServiceError("No content extracted from agent response")
            
            logger.info("Web scraping completed successfully")
            return content
            
        except ClientError as e:
            logger.error(f"AWS client error during scraping: {e}")
            raise BedrockAgentServiceError(f"AWS service error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during web scraping: {e}")
            raise BedrockAgentServiceError(f"Web scraping failed: {e}")

    def parse_spot_data(self, content: str, instance_types: List[str]) -> List[RawSpotData]:
        """
        Parse spot price data from scraped content.
        
        Args:
            content: Raw content from web scraping
            instance_types: List of instance types to filter for
            
        Returns:
            List of RawSpotData objects
            
        Raises:
            BedrockAgentServiceError: If parsing fails
        """
        try:
            logger.info("Parsing spot price data from scraped content")
            
            # Try to parse as JSON first (structured response from agent)
            spot_data_list = []
            
            try:
                data = json.loads(content)
                if 'regions' in data:
                    spot_data_list = self._parse_structured_data(data['regions'], instance_types)
                else:
                    # Fallback to HTML parsing if not structured
                    spot_data_list = self._parse_html_content(content, instance_types)
            except json.JSONDecodeError:
                # Content is not JSON, try HTML parsing
                spot_data_list = self._parse_html_content(content, instance_types)
            
            logger.info(f"Successfully parsed {len(spot_data_list)} spot price records")
            return spot_data_list
            
        except Exception as e:
            logger.error(f"Error parsing spot data: {e}")
            raise BedrockAgentServiceError(f"Failed to parse spot data: {e}")

    def _parse_structured_data(self, regions_data: List[Dict[str, Any]], instance_types: List[str]) -> List[RawSpotData]:
        """
        Parse structured JSON data from agent response.
        
        Args:
            regions_data: List of region data dictionaries
            instance_types: List of instance types to filter for
            
        Returns:
            List of RawSpotData objects
        """
        spot_data_list = []
        current_time = datetime.now(timezone.utc)
        
        for region_data in regions_data:
            try:
                # Filter by instance type
                if region_data.get('instance_type') not in instance_types:
                    continue
                
                # Validate required fields
                required_fields = ['region', 'instance_type', 'spot_price', 'interruption_rate']
                if not all(field in region_data for field in required_fields):
                    logger.warning(f"Missing required fields in region data: {region_data}")
                    continue
                
                # Convert interruption rate from percentage to decimal if needed
                interruption_rate = region_data['interruption_rate']
                if interruption_rate > 1.0:
                    interruption_rate = interruption_rate / 100.0
                
                spot_data = RawSpotData(
                    region=region_data['region'],
                    instance_type=region_data['instance_type'],
                    spot_price=float(region_data['spot_price']),
                    currency=region_data.get('currency', 'USD'),
                    interruption_rate=float(interruption_rate),
                    timestamp=current_time,
                    availability=region_data.get('availability', True)
                )
                
                spot_data_list.append(spot_data)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing region data {region_data}: {e}")
                continue
        
        return spot_data_list

    def _parse_html_content(self, html_content: str, instance_types: List[str]) -> List[RawSpotData]:
        """
        Parse HTML content to extract spot price data.
        
        This is a fallback method when structured data is not available.
        
        Args:
            html_content: Raw HTML content
            instance_types: List of instance types to filter for
            
        Returns:
            List of RawSpotData objects
        """
        spot_data_list = []
        current_time = datetime.now(timezone.utc)
        
        try:
            # Use regex patterns to extract data from HTML
            # This is a simplified implementation - in practice, you might use BeautifulSoup
            
            # Pattern to match region data in HTML tables or structured content
            region_pattern = r'(?i)([a-z]{2}-[a-z]+-\d+)'  # AWS region format
            price_pattern = r'\$?(\d+\.?\d*)'  # Price pattern
            interruption_pattern = r'(\d+\.?\d*)%'  # Interruption rate pattern
            
            # Extract regions
            regions = re.findall(region_pattern, html_content)
            
            # For each instance type, try to find corresponding data
            for instance_type in instance_types:
                instance_pattern = rf'(?i){re.escape(instance_type)}'
                
                # Find sections containing the instance type
                instance_matches = list(re.finditer(instance_pattern, html_content))
                
                for match in instance_matches:
                    # Look for price and interruption rate near the instance type
                    start_pos = max(0, match.start() - 500)
                    end_pos = min(len(html_content), match.end() + 500)
                    context = html_content[start_pos:end_pos]
                    
                    # Extract price and interruption rate from context
                    prices = re.findall(price_pattern, context)
                    interruption_rates = re.findall(interruption_pattern, context)
                    
                    if prices and interruption_rates:
                        # Try to match with regions (simplified approach)
                        for i, region in enumerate(regions[:len(prices)]):
                            try:
                                price = float(prices[i] if i < len(prices) else prices[0])
                                interruption_rate = float(interruption_rates[i] if i < len(interruption_rates) else interruption_rates[0]) / 100.0
                                
                                spot_data = RawSpotData(
                                    region=region,
                                    instance_type=instance_type,
                                    spot_price=price,
                                    currency='USD',
                                    interruption_rate=interruption_rate,
                                    timestamp=current_time,
                                    availability=True
                                )
                                
                                spot_data_list.append(spot_data)
                                
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error parsing HTML data for region {region}: {e}")
                                continue
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            # Return empty list rather than raising exception for HTML parsing failures
        
        return spot_data_list

    def test_connection(self) -> bool:
        """
        Test connection to AWS Bedrock via Strands Agent.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            agent = self._get_agent()
            # Try a simple operation to test connectivity
            test_response = agent("Hello, can you respond with 'Connection successful'?")
            logger.info("Bedrock connection test successful")
            
            # Extract text from response
            response_text = ""
            if test_response and test_response.message and isinstance(test_response.message, dict):
                for item in test_response.message.get('content', []):
                    if isinstance(item, dict) and 'text' in item:
                        response_text += item['text']
            
            return "successful" in response_text.lower()
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {e}")
            return False

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the configured agent.
        
        Returns:
            Dictionary containing agent information
        """
        try:
            return {
                "model_name": self.model_name,
                "region": self.region_name,
                "framework": "Strands Agent",
                "tools": ["http_request"]
            }
            
        except Exception as e:
            logger.error(f"Error getting agent info: {e}")
            return {"error": str(e)}