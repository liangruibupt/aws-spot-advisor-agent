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
from strands.models.bedrock import BedrockModel
from strands.agent.agent_result import AgentResult
from strands_tools.http_request import http_request, extract_content_from_html

from src.models.spot_data import RawSpotData
from src.utils.exceptions import (
    BedrockServiceError,
    NetworkError,
    ServiceUnavailableError,
    RateLimitError,
    DataValidationError
)
from src.utils.retry_utils import aws_service_retry, web_scraping_retry
from src.utils.config import load_config


logger = logging.getLogger(__name__)


# Keep the old exception for backward compatibility
class BedrockAgentServiceError(BedrockServiceError):
    """Legacy exception for BedrockAgentService errors."""
    pass


class BedrockAgentService:
    """
    Service wrapper for AWS Bedrock AgentCore interactions.
    
    This service handles web scraping operations using the Strands Agent framework
    and provides methods for parsing spot price data from HTML content.
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the Bedrock Agent Service.
        
        Args:
            region_name: AWS region for Bedrock service (uses config default if None)
            model_name: Bedrock model to use for content analysis (uses config default if None)
        """
        # Load configuration
        config = load_config()
        
        self.region_name = region_name or config.get('bedrock_region', 'us-east-1')
        self.model_name = model_name or config.get('bedrock_model_id', 'anthropic.claude-sonnet-4-20250514-v1:0')
        
        # Fallback models from configuration and additional backups
        primary_fallback = config.get('bedrock_fallback_model_id', 'anthropic.claude-3-sonnet-20240229-v1:0')
        self.fallback_models = [
            primary_fallback,  # Configured fallback model
            'anthropic.claude-3-5-sonnet-20241022-v2:0',  # Claude 3.5 Sonnet v2
            'anthropic.claude-3-5-sonnet-20240620-v1:0',  # Claude 3.5 Sonnet
            'anthropic.claude-3-sonnet-20240229-v1:0',    # Claude 3 Sonnet (stable fallback)
            'anthropic.claude-3-haiku-20240307-v1:0',     # Claude 3 Haiku (fast fallback)
            'anthropic.claude-v2:1',                      # Claude v2.1 (legacy fallback)
            'anthropic.claude-v2'                         # Claude v2 (legacy fallback)
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        self.fallback_models = [x for x in self.fallback_models if not (x in seen or seen.add(x))]
        
        # Initialize Strands agent
        self._agent: Optional[Agent] = None
        
        # Web scraping instructions template
        self._scraping_instructions = """
        Please analyze the AWS EC2 Spot Instance Advisor web page content and extract spot pricing data.
        
        IMPORTANT: If the page uses dynamic JavaScript and doesn't contain static pricing data, 
        try to find API endpoints or data URLs that might be called by the JavaScript.
        
        Look for:
        1. API endpoints in script tags or network requests
        2. JSON data embedded in the HTML
        3. Data attributes or hidden elements containing pricing info
        
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
        
        If no pricing data is found, return an empty regions array and explain why.
        """

    def _get_agent(self) -> Agent:
        """Get or create Strands Agent instance with fallback models and inference profiles."""
        if self._agent is None:
            # Try primary model first
            models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
            
            last_error = None
            for model_id in models_to_try:
                try:
                    logger.info(f"Attempting to create Strands Agent with model: {model_id}")
                    
                    # Create Bedrock model
                    session = boto3.Session()
                    bedrock_model = BedrockModel(
                        model_id=model_id,
                        boto_session=session,
                        temperature=0.3,
                    )
                    
                    # Create agent with Bedrock model
                    self._agent = Agent(
                        model=bedrock_model,
                        tools=[http_request],
                        callback_handler=None  # Disable default callback handler
                    )
                    
                    # If successful, update the model name
                    if model_id != self.model_name:
                        logger.info(f"Successfully initialized with fallback model: {model_id}")
                        self.model_name = model_id
                    else:
                        logger.info(f"Successfully initialized with primary model: {model_id}")
                    break
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    logger.warning(f"Failed to initialize with model {model_id}: {e}")
                    last_error = e
                    
                    # If it's a validation error, try with inference profile
                    if error_code == 'ValidationException':
                        inference_profile_id = self._try_inference_profile(model_id)
                        if inference_profile_id:
                            try:
                                logger.info(f"Attempting to create Strands Agent with inference profile: {inference_profile_id}")
                                
                                # Create Bedrock model with inference profile
                                session = boto3.Session()
                                bedrock_model = BedrockModel(
                                    model_id=inference_profile_id,
                                    boto_session=session,
                                    temperature=0.3,
                                )
                                
                                self._agent = Agent(
                                    model=bedrock_model,
                                    tools=[http_request],
                                    callback_handler=None  # Disable default callback handler
                                )
                                logger.info(f"Successfully initialized with inference profile: {inference_profile_id}")
                                self.model_name = inference_profile_id
                                break
                            except Exception as profile_error:
                                logger.warning(f"Failed to initialize with inference profile {inference_profile_id}: {profile_error}")
                                continue
                    
                    # If it's a validation error (invalid model), try next model
                    if error_code in ['ValidationException', 'ResourceNotFoundException']:
                        continue
                    else:
                        # For other errors, don't try fallbacks
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to create Strands Agent with model {model_id}: {e}")
                    last_error = e
                    continue
            
            # If no model worked, raise the last error
            if self._agent is None:
                if isinstance(last_error, ClientError):
                    error_code = last_error.response.get('Error', {}).get('Code', 'Unknown')
                    raise BedrockServiceError(
                        message=f"Failed to initialize Strands Agent with any available model. Last error: {last_error}",
                        service_error_code=error_code,
                        region=self.region_name,
                        model_name=self.model_name,
                        original_error=last_error
                    )
                else:
                    raise BedrockServiceError(
                        message=f"Failed to initialize Strands Agent with any available model. Last error: {last_error}",
                        region=self.region_name,
                        model_name=self.model_name,
                        original_error=last_error
                    )
        
        # At this point, self._agent is guaranteed to not be None
        assert self._agent is not None
        return self._agent

    def _try_inference_profile(self, model_id: str) -> Optional[str]:
        """
        Try to find and test an inference profile for the given model ID.
        
        Args:
            model_id: The original model ID that failed
            
        Returns:
            The working inference profile ID, or None if none found
        """
        try:
            bedrock = boto3.client('bedrock', region_name=self.region_name)
            
            # Get available inference profiles
            response = bedrock.list_inference_profiles(typeEquals='SYSTEM_DEFINED')
            inference_profiles = response.get('inferenceProfileSummaries', [])
            
            logger.info(f"Found {len(inference_profiles)} inference profiles to try")
            
            # Look for matching inference profile
            for profile in inference_profiles:
                profile_id = profile.get('inferenceProfileId', '')
                
                # Check if this profile matches our model
                if model_id in profile_id or any(model_id in str(model.get('modelId', '')) for model in profile.get('models', [])):
                    logger.info(f"Testing inference profile: {profile_id}")
                    
                    # Test the inference profile with a simple call
                    if self._test_inference_profile(profile_id):
                        logger.info(f"Inference profile {profile_id} works!")
                        return profile_id
                    else:
                        logger.warning(f"Inference profile {profile_id} failed test")
            
            logger.warning(f"No working inference profile found for model {model_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error trying inference profiles: {e}")
            return None

    def _test_inference_profile(self, profile_id: str) -> bool:
        """
        Test if an inference profile works by making a simple call.
        
        Args:
            profile_id: The inference profile ID to test
            
        Returns:
            True if the profile works, False otherwise
        """
        try:
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region_name)
            
            user_message = "Hello, can you respond with 'Connection successful'?"
            conversation = [{
                "role": "user",
                "content": [{"text": user_message}],
            }]
            
            response = bedrock_runtime.converse(
                modelId=profile_id,
                messages=conversation,
                inferenceConfig={"maxTokens": 50, "temperature": 0.5, "topP": 0.9},
            )
            
            # Extract response text
            response_text = response["output"]["message"]["content"][0]["text"]
            return "successful" in response_text.lower()
            
        except Exception as e:
            logger.warning(f"Inference profile test failed for {profile_id}: {e}")
            return False

    @aws_service_retry(max_attempts=3)
    def execute_web_scraping(self, url: str, custom_instructions: Optional[str] = None) -> str:
        """
        Execute web scraping using Strands Agent with HTTP request tool.
        
        Args:
            url: URL to scrape
            custom_instructions: Optional custom scraping instructions
            
        Returns:
            Raw HTML content or structured data from the agent
            
        Raises:
            BedrockServiceError: If scraping fails
            NetworkError: If network issues occur
            ServiceUnavailableError: If Bedrock service is unavailable
            RateLimitError: If rate limits are exceeded
        """
        instructions = custom_instructions or self._scraping_instructions
        
        try:
            logger.info(f"Starting web scraping for URL: {url}")
            
            # Get the Strands agent
            agent = self._get_agent()
            
            # Create the prompt for the agent
            prompt = f"""
            Use the http_request tool to fetch the raw HTML content from: {url}

            After you get the HTML content, extract spot pricing data for p5en.48xlarge and p5.48xlarge instances.

            Look for:
            - Region names (like us-east-1, eu-west-1, etc.)
            - Instance types p5en.48xlarge and p5.48xlarge
            - Spot prices in USD
            - Interruption rates as percentages

            Return ONLY a JSON response with this format:
            {{
                "regions": [
                    {{
                        "region": "us-east-1",
                        "instance_type": "p5en.48xlarge",
                        "spot_price": 12.50,
                        "currency": "USD",
                        "interruption_rate": 0.03,
                        "availability": true
                    }}
                ]
            }}

            Do not provide explanations - just the JSON data.
            """
            
            # Execute the agent with the prompt
            response = agent(prompt)
            
            if not response or not response.message:
                raise BedrockServiceError(
                    message="No response returned from agent",
                    region=self.region_name,
                    model_name=self.model_name
                )
            
            # Debug: Log the full response structure
            logger.info(f"Agent response type: {type(response)}")
            logger.info(f"Agent response message: {response.message}")
            
            # Check if there are tool results in the response
            content = ""
            if hasattr(response, 'tool_results') and response.tool_results:
                logger.info(f"Found {len(response.tool_results)} tool results")
                for tool_result in response.tool_results:
                    logger.info(f"Tool result: {tool_result}")
                    if hasattr(tool_result, 'content'):
                        content += str(tool_result.content)
                    elif isinstance(tool_result, dict) and 'content' in tool_result:
                        content += str(tool_result['content'])
                    else:
                        content += str(tool_result)
            
            # If no tool results, extract from message
            if not content and isinstance(response.message, dict) and 'content' in response.message:
                for item in response.message['content']:
                    if isinstance(item, dict) and 'text' in item:
                        content += item['text']
            
            # If still no content, try the raw message
            if not content:
                content = str(response.message)
            
            if not content:
                raise BedrockServiceError(
                    message="No content extracted from agent response",
                    region=self.region_name,
                    model_name=self.model_name
                )
            
            logger.info("Web scraping completed successfully")
            return content
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            
            logger.error(f"AWS client error during scraping: {e}")
            
            # Handle specific AWS error types
            if error_code in ['Throttling', 'ThrottlingException', 'TooManyRequestsException']:
                raise RateLimitError(
                    message=f"Rate limit exceeded for Bedrock service: {e}",
                    service_name="AWS Bedrock",
                    original_error=e
                )
            elif error_code in ['ServiceUnavailable', 'InternalError']:
                raise ServiceUnavailableError(
                    message=f"Bedrock service unavailable: {e}",
                    service_name="AWS Bedrock",
                    status_code=status_code,
                    original_error=e
                )
            else:
                raise BedrockServiceError(
                    message=f"AWS service error: {e}",
                    service_error_code=error_code,
                    region=self.region_name,
                    model_name=self.model_name,
                    original_error=e
                )
        except BotoCoreError as e:
            logger.error(f"Boto core error during scraping: {e}")
            raise NetworkError(
                message=f"Network error during web scraping: {e}",
                url=url,
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unexpected error during web scraping: {e}")
            raise BedrockServiceError(
                message=f"Web scraping failed: {e}",
                region=self.region_name,
                model_name=self.model_name,
                original_error=e
            )

    def parse_spot_data(self, content: str, instance_types: List[str]) -> List[RawSpotData]:
        """
        Parse spot price data from scraped content.
        
        Args:
            content: Raw content from web scraping
            instance_types: List of instance types to filter for
            
        Returns:
            List of RawSpotData objects
            
        Raises:
            BedrockServiceError: If parsing fails
            DataValidationError: If data validation fails
        """
        try:
            logger.info("Parsing spot price data from scraped content")
            
            if not content or not content.strip():
                raise DataValidationError(
                    message="Empty content provided for parsing",
                    field_name="content",
                    field_value=content
                )
            
            if not instance_types:
                raise DataValidationError(
                    message="No instance types provided for filtering",
                    field_name="instance_types",
                    field_value=instance_types
                )
            
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
            
        except DataValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Error parsing spot data: {e}")
            raise BedrockServiceError(
                message=f"Failed to parse spot data: {e}",
                region=self.region_name,
                model_name=self.model_name,
                original_error=e
            )

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
                missing_fields = [field for field in required_fields if field not in region_data]
                if missing_fields:
                    logger.warning(f"Missing required fields {missing_fields} in region data: {region_data}")
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