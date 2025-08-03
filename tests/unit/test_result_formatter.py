"""
Unit tests for ResultFormatter service.

Tests JSON response formatting functionality including timestamp formatting,
currency denomination, and percentage formatting.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from src.services.result_formatter import (
    ResultFormatter,
    ResultFormatterError,
    InvalidDataError
)
from src.models.spot_data import SpotPriceResult, AnalysisResponse


class TestResultFormatter:
    """Test cases for ResultFormatter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResultFormatter()
        
        # Create test data
        self.test_timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        
        self.test_result = SpotPriceResult(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.5678,
            currency="USD",
            interruption_rate=0.0234,
            rank=1,
            data_timestamp=self.test_timestamp
        )
        
        self.test_results = [
            self.test_result,
            SpotPriceResult(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=13.1234,
                currency="USD",
                interruption_rate=0.0456,
                rank=2,
                data_timestamp=self.test_timestamp
            )
        ]
        
        self.test_response = AnalysisResponse(
            results=self.test_results,
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=self.test_timestamp,
            warnings=["Test warning"]
        )

    def test_init(self):
        """Test ResultFormatter initialization."""
        formatter = ResultFormatter()
        assert formatter is not None

    def test_format_analysis_response_success(self):
        """Test successful formatting of analysis response."""
        result = self.formatter.format_analysis_response(self.test_response)
        
        # Check structure
        assert "results" in result
        assert "metadata" in result
        assert "warnings" in result
        
        # Check metadata
        metadata = result["metadata"]
        assert metadata["total_regions_analyzed"] == 10
        assert metadata["filtered_regions_count"] == 5
        assert metadata["result_count"] == 2
        assert metadata["data_collection_timestamp"] == "2024-01-15T10:30:45+00:00"
        
        # Check results
        assert len(result["results"]) == 2
        
        # Check warnings
        assert result["warnings"] == ["Test warning"]

    def test_format_analysis_response_no_warnings(self):
        """Test formatting analysis response without warnings."""
        response_no_warnings = AnalysisResponse(
            results=self.test_results,
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=self.test_timestamp
        )
        
        result = self.formatter.format_analysis_response(response_no_warnings)
        
        # Should not include warnings key when no warnings
        assert "warnings" not in result

    def test_format_analysis_response_invalid_input(self):
        """Test formatting analysis response with invalid input."""
        with pytest.raises(InvalidDataError, match="Response must be an AnalysisResponse instance"):
            self.formatter.format_analysis_response("invalid")

    def test_format_spot_price_results_success(self):
        """Test successful formatting of spot price results."""
        result = self.formatter.format_spot_price_results(self.test_results)
        
        assert len(result) == 2
        
        # Check first result
        first_result = result[0]
        assert first_result["region"] == "us-east-1"
        assert first_result["instance_type"] == "p5en.48xlarge"
        assert first_result["spot_price"]["amount"] == 12.5678
        assert first_result["spot_price"]["currency"] == "USD"
        assert first_result["interruption_rate"] == "2.34%"
        assert first_result["rank"] == 1
        assert first_result["data_timestamp"] == "2024-01-15T10:30:45+00:00"

    def test_format_spot_price_results_invalid_input(self):
        """Test formatting spot price results with invalid input."""
        with pytest.raises(InvalidDataError, match="Results must be a list"):
            self.formatter.format_spot_price_results("invalid")

    def test_format_spot_price_results_invalid_result_in_list(self):
        """Test formatting spot price results with invalid result in list."""
        invalid_results = [self.test_result, "invalid_result"]
        
        with pytest.raises(InvalidDataError, match="Failed to format result at index 1"):
            self.formatter.format_spot_price_results(invalid_results)

    def test_format_spot_price_result_success(self):
        """Test successful formatting of single spot price result."""
        result = self.formatter._format_spot_price_result(self.test_result)
        
        expected = {
            "region": "us-east-1",
            "instance_type": "p5en.48xlarge",
            "spot_price": {
                "amount": 12.5678,
                "currency": "USD"
            },
            "interruption_rate": "2.34%",
            "rank": 1,
            "data_timestamp": "2024-01-15T10:30:45+00:00"
        }
        
        assert result == expected

    def test_format_spot_price_result_invalid_input(self):
        """Test formatting single spot price result with invalid input."""
        with pytest.raises(InvalidDataError, match="Result must be a SpotPriceResult instance"):
            self.formatter._format_spot_price_result("invalid")

    def test_format_price_success(self):
        """Test successful price formatting."""
        result = self.formatter._format_price(12.5678, "USD")
        
        expected = {
            "amount": 12.5678,
            "currency": "USD"
        }
        
        assert result == expected

    def test_format_price_rounding(self):
        """Test price formatting with rounding to 4 decimal places."""
        result = self.formatter._format_price(12.56789123, "USD")
        
        expected = {
            "amount": 12.5679,  # Rounded to 4 decimal places
            "currency": "USD"
        }
        
        assert result == expected

    def test_format_price_invalid_price(self):
        """Test price formatting with invalid price."""
        with pytest.raises(InvalidDataError, match="Price must be a non-negative number"):
            self.formatter._format_price(-1.0, "USD")
        
        with pytest.raises(InvalidDataError, match="Price must be a non-negative number"):
            self.formatter._format_price("invalid", "USD")

    def test_format_price_invalid_currency(self):
        """Test price formatting with invalid currency."""
        with pytest.raises(InvalidDataError, match="Currency must be 'USD'"):
            self.formatter._format_price(12.34, "EUR")

    def test_format_interruption_rate_success(self):
        """Test successful interruption rate formatting."""
        test_cases = [
            (0.0234, "2.34%"),
            (0.05, "5.00%"),
            (0.1, "10.00%"),
            (0.0001, "0.01%"),
            (0.9999, "99.99%"),
            (0, "0.00%"),
            (1, "100.00%")
        ]
        
        for rate, expected in test_cases:
            result = self.formatter._format_interruption_rate(rate)
            assert result == expected

    def test_format_interruption_rate_invalid_input(self):
        """Test interruption rate formatting with invalid input."""
        with pytest.raises(InvalidDataError, match="Interruption rate must be between 0 and 1"):
            self.formatter._format_interruption_rate(-0.1)
        
        with pytest.raises(InvalidDataError, match="Interruption rate must be between 0 and 1"):
            self.formatter._format_interruption_rate(1.1)
        
        with pytest.raises(InvalidDataError, match="Interruption rate must be between 0 and 1"):
            self.formatter._format_interruption_rate("invalid")

    def test_format_timestamp_success(self):
        """Test successful timestamp formatting."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        result = self.formatter._format_timestamp(timestamp)
        
        assert result == "2024-01-15T10:30:45+00:00"

    def test_format_timestamp_naive_datetime(self):
        """Test timestamp formatting with naive datetime (assumes UTC)."""
        naive_timestamp = datetime(2024, 1, 15, 10, 30, 45)
        result = self.formatter._format_timestamp(naive_timestamp)
        
        assert result == "2024-01-15T10:30:45+00:00"

    def test_format_timestamp_non_utc_timezone(self):
        """Test timestamp formatting with non-UTC timezone (converts to UTC)."""
        from datetime import timedelta
        
        # Create a timezone offset of +5 hours
        tz_offset = timezone(timedelta(hours=5))
        timestamp = datetime(2024, 1, 15, 15, 30, 45, tzinfo=tz_offset)
        
        result = self.formatter._format_timestamp(timestamp)
        
        # Should be converted to UTC (15:30 +5 = 10:30 UTC)
        assert result == "2024-01-15T10:30:45+00:00"

    def test_format_timestamp_invalid_input(self):
        """Test timestamp formatting with invalid input."""
        with pytest.raises(InvalidDataError, match="Timestamp must be a datetime object"):
            self.formatter._format_timestamp("invalid")

    def test_format_error_response_basic(self):
        """Test basic error response formatting."""
        result = self.formatter.format_error_response("Test error message")
        
        assert "error" in result
        assert result["error"]["message"] == "Test error message"
        assert "timestamp" in result["error"]
        
        # Verify timestamp format
        timestamp_str = result["error"]["timestamp"]
        datetime.fromisoformat(timestamp_str.replace('+00:00', '+0000'))

    def test_format_error_response_with_code_and_details(self):
        """Test error response formatting with code and details."""
        details = {"field": "instance_type", "value": "invalid"}
        result = self.formatter.format_error_response(
            "Validation error",
            error_code="VALIDATION_ERROR",
            details=details
        )
        
        assert result["error"]["message"] == "Validation error"
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert result["error"]["details"] == details

    def test_to_json_string_success(self):
        """Test successful JSON string conversion."""
        data = {"test": "value", "number": 123}
        result = self.formatter.to_json_string(data)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == data

    def test_to_json_string_with_indent(self):
        """Test JSON string conversion with indentation."""
        data = {"test": "value", "nested": {"key": "value"}}
        result = self.formatter.to_json_string(data, indent=2)
        
        # Should contain newlines and spaces for formatting
        assert "\n" in result
        assert "  " in result
        
        # Should still be valid JSON
        parsed = json.loads(result)
        assert parsed == data

    def test_to_json_string_invalid_data(self):
        """Test JSON string conversion with non-serializable data."""
        # Create non-serializable data
        data = {"datetime": datetime.now()}
        
        with pytest.raises(InvalidDataError, match="Failed to serialize to JSON"):
            self.formatter.to_json_string(data)

    def test_format_summary_statistics_success(self):
        """Test successful summary statistics formatting."""
        result = self.formatter.format_summary_statistics(self.test_response)
        
        # Check analysis summary
        assert "analysis_summary" in result
        summary = result["analysis_summary"]
        assert summary["total_regions_analyzed"] == 10
        assert summary["regions_meeting_criteria"] == 5
        assert summary["results_returned"] == 2
        
        # Check price statistics
        assert "price_statistics" in result
        price_stats = result["price_statistics"]
        assert price_stats["lowest_price"]["amount"] == 12.5678
        assert price_stats["highest_price"]["amount"] == 13.1234
        assert price_stats["average_price"]["amount"] == 12.8456  # (12.5678 + 13.1234) / 2
        
        # Check interruption rate statistics
        assert "interruption_rate_statistics" in result
        rate_stats = result["interruption_rate_statistics"]
        assert rate_stats["lowest_rate"] == "2.34%"
        assert rate_stats["highest_rate"] == "4.56%"
        assert rate_stats["average_rate"] == "3.45%"  # (2.34 + 4.56) / 2
        
        # Check warnings count
        assert result["warnings_count"] == 1

    def test_format_summary_statistics_empty_results(self):
        """Test summary statistics formatting with empty results."""
        empty_response = AnalysisResponse(
            results=[],
            total_regions_analyzed=10,
            filtered_regions_count=0,
            data_collection_timestamp=self.test_timestamp
        )
        
        result = self.formatter.format_summary_statistics(empty_response)
        
        # Should have analysis summary but no price/rate statistics
        assert "analysis_summary" in result
        assert "price_statistics" not in result
        assert "interruption_rate_statistics" not in result
        assert "warnings_count" not in result

    def test_format_summary_statistics_invalid_input(self):
        """Test summary statistics formatting with invalid input."""
        with pytest.raises(InvalidDataError, match="Response must be an AnalysisResponse instance"):
            self.formatter.format_summary_statistics("invalid")

    def test_validate_json_structure_success(self):
        """Test successful JSON structure validation."""
        valid_data = {
            "results": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": {"amount": 12.34, "currency": "USD"},
                    "interruption_rate": "2.34%",
                    "rank": 1,
                    "data_timestamp": "2024-01-15T10:30:45+00:00"
                }
            ],
            "metadata": {
                "total_regions_analyzed": 10,
                "filtered_regions_count": 5,
                "data_collection_timestamp": "2024-01-15T10:30:45+00:00",
                "result_count": 1
            }
        }
        
        result = self.formatter.validate_json_structure(valid_data)
        assert result is True

    def test_validate_json_structure_missing_required_field(self):
        """Test JSON structure validation with missing required field."""
        invalid_data = {
            "results": [],
            # Missing metadata
        }
        
        with pytest.raises(InvalidDataError, match="Missing required field: metadata"):
            self.formatter.validate_json_structure(invalid_data)

    def test_validate_json_structure_invalid_results_type(self):
        """Test JSON structure validation with invalid results type."""
        invalid_data = {
            "results": "not_a_list",
            "metadata": {}
        }
        
        with pytest.raises(InvalidDataError, match="Results must be a list"):
            self.formatter.validate_json_structure(invalid_data)

    def test_validate_json_structure_invalid_result_structure(self):
        """Test JSON structure validation with invalid result structure."""
        invalid_data = {
            "results": [
                {
                    "region": "us-east-1",
                    # Missing required fields
                }
            ],
            "metadata": {
                "total_regions_analyzed": 10,
                "filtered_regions_count": 5,
                "data_collection_timestamp": "2024-01-15T10:30:45+00:00",
                "result_count": 1
            }
        }
        
        with pytest.raises(InvalidDataError, match="Result at index 0 missing field"):
            self.formatter.validate_json_structure(invalid_data)

    def test_validate_json_structure_invalid_spot_price_structure(self):
        """Test JSON structure validation with invalid spot price structure."""
        invalid_data = {
            "results": [
                {
                    "region": "us-east-1",
                    "instance_type": "p5en.48xlarge",
                    "spot_price": "not_a_dict",  # Should be dict with amount and currency
                    "interruption_rate": "2.34%",
                    "rank": 1,
                    "data_timestamp": "2024-01-15T10:30:45+00:00"
                }
            ],
            "metadata": {
                "total_regions_analyzed": 10,
                "filtered_regions_count": 5,
                "data_collection_timestamp": "2024-01-15T10:30:45+00:00",
                "result_count": 1
            }
        }
        
        with pytest.raises(InvalidDataError, match="spot_price must be a dictionary"):
            self.formatter.validate_json_structure(invalid_data)

    def test_integration_with_real_data(self):
        """Test integration with real analysis response data."""
        # Create a complete analysis response
        response = self.formatter.format_analysis_response(self.test_response)
        
        # Convert to JSON string and back to verify serialization
        json_string = self.formatter.to_json_string(response, indent=2)
        parsed_back = json.loads(json_string)
        
        # Verify structure is maintained
        assert parsed_back == response
        
        # Validate the structure
        assert self.formatter.validate_json_structure(response) is True
        
        # Verify specific formatting requirements
        first_result = response["results"][0]
        assert first_result["spot_price"]["currency"] == "USD"
        assert first_result["interruption_rate"].endswith("%")
        assert "." in first_result["interruption_rate"]  # Should have decimal places
        assert first_result["data_timestamp"].endswith("+00:00")  # UTC timezone