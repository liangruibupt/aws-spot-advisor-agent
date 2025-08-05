"""
Result formatter service for AWS Spot Price Analyzer.

This module provides JSON response formatting functionality that structures
output according to API requirements with proper timestamp formatting,
currency denomination, and percentage formatting.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.models.spot_data import SpotPriceResult, AnalysisResponse
from src.utils.exceptions import FormattingError, DataValidationError
from src.utils.error_middleware import handle_service_errors, create_error_mapping
from src.utils.retry_utils import data_processing_retry


logger = logging.getLogger(__name__)


# Legacy exceptions for backward compatibility
class ResultFormatterError(FormattingError):
    """Base exception for ResultFormatter errors."""
    pass


class InvalidDataError(DataValidationError):
    """Raised when input data is invalid for formatting."""
    pass


class ResultFormatter:
    """
    Service for formatting analysis results into structured JSON responses.
    
    This service handles:
    - Timestamp formatting in ISO 8601 format
    - Currency denomination (USD) formatting
    - Interruption rate formatting as percentages with 2 decimal places
    - Structured JSON output according to API requirements
    - Enhanced error handling and validation
    """

    def __init__(self):
        """Initialize the Result Formatter with error handling."""
        self.error_mappings = create_error_mapping()
        self.error_mappings.update({
            json.JSONDecodeError: FormattingError,
            UnicodeDecodeError: FormattingError,
        })
        logger.info("ResultFormatter initialized with enhanced error handling")

    @handle_service_errors("result_formatter", "format_analysis_response")
    @data_processing_retry(max_attempts=2)
    def format_analysis_response(self, response: AnalysisResponse) -> Dict[str, Any]:
        """
        Format an AnalysisResponse into a structured JSON-ready dictionary.
        
        Args:
            response: AnalysisResponse object to format
            
        Returns:
            Dictionary ready for JSON serialization
            
        Raises:
            FormattingError: If response data is invalid or formatting fails
        """
        if not isinstance(response, AnalysisResponse):
            raise DataValidationError(
                message="Response must be an AnalysisResponse instance",
                field_name="response",
                field_value=type(response).__name__,
                validation_rule="must be AnalysisResponse"
            )
        
        logger.debug(f"Formatting analysis response with {len(response.results)} results")
        
        formatted_response = {
            "results": [self._format_spot_price_result(result) for result in response.results],
            "metadata": {
                "total_regions_analyzed": response.total_regions_analyzed,
                "filtered_regions_count": response.filtered_regions_count,
                "data_collection_timestamp": self._format_timestamp(response.data_collection_timestamp),
                "result_count": len(response.results)
            }
        }
        
        # Add warnings if present
        if response.warnings:
            formatted_response["warnings"] = response.warnings
        
        logger.debug("Analysis response formatted successfully")
        return formatted_response

    @handle_service_errors("result_formatter", "format_spot_price_results")
    @data_processing_retry(max_attempts=2)
    def format_spot_price_results(self, results: List[SpotPriceResult]) -> List[Dict[str, Any]]:
        """
        Format a list of SpotPriceResult objects into JSON-ready dictionaries.
        
        Args:
            results: List of SpotPriceResult objects to format
            
        Returns:
            List of dictionaries ready for JSON serialization
            
        Raises:
            FormattingError: If results data is invalid or formatting fails
        """
        if not isinstance(results, list):
            raise DataValidationError(
                message="Results must be a list",
                field_name="results",
                field_value=type(results).__name__,
                validation_rule="must be list"
            )
        
        logger.debug(f"Formatting {len(results)} spot price results")
        
        formatted_results = []
        for i, result in enumerate(results):
            try:
                formatted_result = self._format_spot_price_result(result)
                formatted_results.append(formatted_result)
            except Exception as e:
                raise FormattingError(
                    message=f"Failed to format result at index {i}: {e}",
                    format_type="spot_price_result",
                    data_type="SpotPriceResult",
                    original_error=e
                )
        
        logger.debug("Spot price results formatted successfully")
        return formatted_results

    def _format_spot_price_result(self, result: SpotPriceResult) -> Dict[str, Any]:
        """
        Format a single SpotPriceResult into a JSON-ready dictionary.
        
        Args:
            result: SpotPriceResult object to format
            
        Returns:
            Dictionary ready for JSON serialization
            
        Raises:
            InvalidDataError: If result data is invalid
        """
        if not isinstance(result, SpotPriceResult):
            raise DataValidationError(
                message="Result must be a SpotPriceResult instance",
                field_name="result",
                field_value=type(result).__name__,
                validation_rule="must be SpotPriceResult"
            )
        
        return {
            "region": result.region,
            "instance_type": result.instance_type,
            "spot_price": self._format_price(result.spot_price, result.currency),
            "interruption_rate": self._format_interruption_rate(result.interruption_rate),
            "rank": result.rank,
            "data_timestamp": self._format_timestamp(result.data_timestamp)
        }

    def _format_price(self, price: float, currency: str) -> Dict[str, Any]:
        """
        Format price with currency denomination.
        
        Args:
            price: Price value
            currency: Currency code (should be 'USD')
            
        Returns:
            Dictionary with formatted price and currency
            
        Raises:
            InvalidDataError: If price or currency is invalid
        """
        if not isinstance(price, (int, float)) or price < 0:
            raise DataValidationError(
                message="Price must be a non-negative number",
                field_name="price",
                field_value=price,
                validation_rule="must be non-negative number"
            )
        
        if currency != "USD":
            raise DataValidationError(
                message="Currency must be 'USD'",
                field_name="currency",
                field_value=currency,
                validation_rule="must be 'USD'"
            )
        
        return {
            "amount": round(price, 4),  # Round to 4 decimal places for precision
            "currency": currency
        }

    def _format_interruption_rate(self, rate: float) -> str:
        """
        Format interruption rate as percentage with 2 decimal places.
        
        Args:
            rate: Interruption rate as decimal (0.05 = 5%)
            
        Returns:
            Formatted percentage string (e.g., "5.00%")
            
        Raises:
            InvalidDataError: If rate is invalid
        """
        if not isinstance(rate, (int, float)) or not (0 <= rate <= 1):
            raise DataValidationError(
                message="Interruption rate must be between 0 and 1",
                field_name="interruption_rate",
                field_value=rate,
                validation_rule="must be between 0 and 1"
            )
        
        percentage = rate * 100
        return f"{percentage:.2f}%"

    def _format_timestamp(self, timestamp: datetime) -> str:
        """
        Format timestamp in ISO 8601 format with UTC timezone.
        
        Args:
            timestamp: Datetime object to format
            
        Returns:
            ISO 8601 formatted timestamp string
            
        Raises:
            InvalidDataError: If timestamp is invalid
        """
        if not isinstance(timestamp, datetime):
            raise DataValidationError(
                message="Timestamp must be a datetime object",
                field_name="timestamp",
                field_value=type(timestamp).__name__,
                validation_rule="must be datetime"
            )
        
        # Ensure timezone is UTC for consistency
        if timestamp.tzinfo is None:
            # Assume naive datetime is UTC
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            # Convert to UTC
            timestamp = timestamp.astimezone(timezone.utc)
        
        return timestamp.isoformat()

    def format_error_response(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format an error response into a structured JSON-ready dictionary.
        
        Args:
            error_message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional error details
            
        Returns:
            Dictionary ready for JSON serialization
        """
        error_response = {
            "error": {
                "message": error_message,
                "timestamp": self._format_timestamp(datetime.now(timezone.utc))
            }
        }
        
        if error_code:
            error_response["error"]["code"] = error_code
        
        if details:
            error_response["error"]["details"] = details
        
        logger.debug(f"Formatted error response: {error_message}")
        return error_response

    def to_json_string(
        self,
        data: Dict[str, Any],
        indent: Optional[int] = None,
        ensure_ascii: bool = False
    ) -> str:
        """
        Convert formatted data to JSON string.
        
        Args:
            data: Dictionary to convert to JSON
            indent: Number of spaces for indentation (None for compact)
            ensure_ascii: Whether to escape non-ASCII characters
            
        Returns:
            JSON string representation
            
        Raises:
            InvalidDataError: If data cannot be serialized to JSON
        """
        try:
            return json.dumps(
                data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                separators=(',', ':') if indent is None else (',', ': ')
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data to JSON: {e}")
            raise FormattingError(
                message=f"Failed to serialize to JSON: {e}",
                format_type="JSON",
                original_error=e
            )

    def format_summary_statistics(self, response: AnalysisResponse) -> Dict[str, Any]:
        """
        Format summary statistics from analysis response.
        
        Args:
            response: AnalysisResponse object
            
        Returns:
            Dictionary with formatted summary statistics
        """
        if not isinstance(response, AnalysisResponse):
            raise DataValidationError(
                message="Response must be an AnalysisResponse instance",
                field_name="response",
                field_value=type(response).__name__,
                validation_rule="must be AnalysisResponse"
            )
        
        # Calculate statistics
        prices = [result.spot_price for result in response.results]
        interruption_rates = [result.interruption_rate for result in response.results]
        
        statistics = {
            "analysis_summary": {
                "total_regions_analyzed": response.total_regions_analyzed,
                "regions_meeting_criteria": response.filtered_regions_count,
                "results_returned": len(response.results),
                "data_collection_time": self._format_timestamp(response.data_collection_timestamp)
            }
        }
        
        if prices:
            statistics["price_statistics"] = {
                "lowest_price": {
                    "amount": round(min(prices), 4),
                    "currency": "USD"
                },
                "highest_price": {
                    "amount": round(max(prices), 4),
                    "currency": "USD"
                },
                "average_price": {
                    "amount": round(sum(prices) / len(prices), 4),
                    "currency": "USD"
                }
            }
        
        if interruption_rates:
            statistics["interruption_rate_statistics"] = {
                "lowest_rate": self._format_interruption_rate(min(interruption_rates)),
                "highest_rate": self._format_interruption_rate(max(interruption_rates)),
                "average_rate": self._format_interruption_rate(sum(interruption_rates) / len(interruption_rates))
            }
        
        if response.warnings:
            statistics["warnings_count"] = len(response.warnings)
        
        return statistics

    def validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the formatted data has the expected JSON structure.
        
        Args:
            data: Dictionary to validate
            
        Returns:
            True if structure is valid
            
        Raises:
            InvalidDataError: If structure is invalid
        """
        required_fields = ["results", "metadata"]
        
        for field in required_fields:
            if field not in data:
                raise DataValidationError(
                    message=f"Missing required field: {field}",
                    field_name=field,
                    validation_rule="required field"
                )
        
        # Validate results structure
        if not isinstance(data["results"], list):
            raise DataValidationError(
                message="Results must be a list",
                field_name="results",
                field_value=type(data["results"]).__name__,
                validation_rule="must be list"
            )
        
        for i, result in enumerate(data["results"]):
            self._validate_result_structure(result, i)
        
        # Validate metadata structure
        self._validate_metadata_structure(data["metadata"])
        
        logger.debug("JSON structure validation passed")
        return True

    def _validate_result_structure(self, result: Dict[str, Any], index: int) -> None:
        """Validate individual result structure."""
        required_result_fields = [
            "region", "instance_type", "spot_price", 
            "interruption_rate", "rank", "data_timestamp"
        ]
        
        for field in required_result_fields:
            if field not in result:
                raise DataValidationError(
                    message=f"Result at index {index} missing field: {field}",
                    field_name=field,
                    validation_rule="required field"
                )
        
        # Validate spot_price structure
        if not isinstance(result["spot_price"], dict):
            raise DataValidationError(
                message=f"Result at index {index}: spot_price must be a dictionary",
                field_name="spot_price",
                field_value=type(result["spot_price"]).__name__,
                validation_rule="must be dictionary"
            )
        
        if "amount" not in result["spot_price"] or "currency" not in result["spot_price"]:
            raise DataValidationError(
                message=f"Result at index {index}: spot_price missing amount or currency",
                field_name="spot_price",
                validation_rule="must contain amount and currency"
            )

    def _validate_metadata_structure(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata structure."""
        required_metadata_fields = [
            "total_regions_analyzed", "filtered_regions_count", 
            "data_collection_timestamp", "result_count"
        ]
        
        for field in required_metadata_fields:
            if field not in metadata:
                raise DataValidationError(
                    message=f"Metadata missing field: {field}",
                    field_name=field,
                    validation_rule="required metadata field"
                )