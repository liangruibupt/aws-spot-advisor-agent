"""
Unit tests for data models in the AWS Spot Price Analyzer.

Tests cover validation, serialization, and business logic for all data models.
"""

import pytest
from datetime import datetime, timezone
from typing import List

from src.models.spot_data import (
    RawSpotData,
    SpotPriceResult,
    AnalysisResponse,
    RawSpotDataValidator,
    validate_raw_spot_data,
    validate_spot_price_results,
)
from pydantic import ValidationError


class TestRawSpotData:
    """Test cases for RawSpotData dataclass."""

    def test_valid_raw_spot_data_creation(self):
        """Test creating valid RawSpotData instance."""
        timestamp = datetime.now(timezone.utc)
        data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp,
            availability=True
        )
        
        assert data.region == "us-east-1"
        assert data.instance_type == "p5en.48xlarge"
        assert data.spot_price == 12.50
        assert data.currency == "USD"
        assert data.interruption_rate == 0.03
        assert data.timestamp == timestamp
        assert data.availability is True

    def test_raw_spot_data_default_availability(self):
        """Test that availability defaults to True."""
        timestamp = datetime.now(timezone.utc)
        data = RawSpotData(
            region="us-west-2",
            instance_type="p5.48xlarge",
            spot_price=15.00,
            currency="USD",
            interruption_rate=0.02,
            timestamp=timestamp
        )
        
        assert data.availability is True

    def test_interruption_rate_percentage_property(self):
        """Test interruption rate percentage calculation."""
        timestamp = datetime.now(timezone.utc)
        data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.035,
            timestamp=timestamp
        )
        
        assert abs(data.interruption_rate_percentage - 3.5) < 0.0001

    def test_is_low_interruption_method(self):
        """Test low interruption rate checking."""
        timestamp = datetime.now(timezone.utc)
        
        # Low interruption rate
        low_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp
        )
        assert low_data.is_low_interruption() is True
        assert low_data.is_low_interruption(0.02) is False
        
        # High interruption rate
        high_data = RawSpotData(
            region="us-west-1",
            instance_type="p5en.48xlarge",
            spot_price=10.00,
            currency="USD",
            interruption_rate=0.08,
            timestamp=timestamp
        )
        assert high_data.is_low_interruption() is False

    def test_raw_spot_data_validation_empty_region(self):
        """Test validation fails for empty region."""
        timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValueError, match="Region must be a non-empty string"):
            RawSpotData(
                region="",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp
            )

    def test_raw_spot_data_validation_invalid_price(self):
        """Test validation fails for negative price."""
        timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValueError, match="Spot price must be a non-negative number"):
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=-5.00,
                currency="USD",
                interruption_rate=0.03,
                timestamp=timestamp
            )

    def test_raw_spot_data_validation_invalid_currency(self):
        """Test validation fails for non-USD currency."""
        timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValueError, match="Currency must be 'USD'"):
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="EUR",
                interruption_rate=0.03,
                timestamp=timestamp
            )

    def test_raw_spot_data_validation_invalid_interruption_rate(self):
        """Test validation fails for invalid interruption rate."""
        timestamp = datetime.now(timezone.utc)
        
        # Test negative rate
        with pytest.raises(ValueError, match="Interruption rate must be between 0 and 1"):
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=-0.1,
                timestamp=timestamp
            )
        
        # Test rate > 1
        with pytest.raises(ValueError, match="Interruption rate must be between 0 and 1"):
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=1.5,
                timestamp=timestamp
            )


class TestSpotPriceResult:
    """Test cases for SpotPriceResult dataclass."""

    def test_valid_spot_price_result_creation(self):
        """Test creating valid SpotPriceResult instance."""
        timestamp = datetime.now(timezone.utc)
        result = SpotPriceResult(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            rank=1,
            data_timestamp=timestamp
        )
        
        assert result.region == "us-east-1"
        assert result.instance_type == "p5en.48xlarge"
        assert result.spot_price == 12.50
        assert result.currency == "USD"
        assert result.interruption_rate == 0.03
        assert result.rank == 1
        assert result.data_timestamp == timestamp

    def test_interruption_rate_percentage_property(self):
        """Test formatted interruption rate percentage."""
        timestamp = datetime.now(timezone.utc)
        result = SpotPriceResult(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.0345,
            rank=1,
            data_timestamp=timestamp
        )
        
        assert result.interruption_rate_percentage == "3.45%"

    def test_to_dict_method(self):
        """Test dictionary conversion for JSON serialization."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = SpotPriceResult(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            rank=1,
            data_timestamp=timestamp
        )
        
        expected_dict = {
            "region": "us-east-1",
            "instance_type": "p5en.48xlarge",
            "spot_price": 12.50,
            "currency": "USD",
            "interruption_rate": "3.00%",
            "rank": 1,
            "data_timestamp": "2024-01-15T10:30:00+00:00"
        }
        
        assert result.to_dict() == expected_dict

    def test_spot_price_result_validation_invalid_rank(self):
        """Test validation fails for invalid rank."""
        timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValueError, match="Rank must be a positive integer"):
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                rank=0,
                data_timestamp=timestamp
            )


class TestAnalysisResponse:
    """Test cases for AnalysisResponse dataclass."""

    def test_valid_analysis_response_creation(self):
        """Test creating valid AnalysisResponse instance."""
        timestamp = datetime.now(timezone.utc)
        results = [
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                rank=1,
                data_timestamp=timestamp
            )
        ]
        
        response = AnalysisResponse(
            results=results,
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=timestamp
        )
        
        assert response.results == results
        assert response.total_regions_analyzed == 10
        assert response.filtered_regions_count == 5
        assert response.data_collection_timestamp == timestamp
        assert response.warnings == []

    def test_analysis_response_with_warnings(self):
        """Test AnalysisResponse with warning messages."""
        timestamp = datetime.now(timezone.utc)
        results = []
        warnings = ["No regions met the criteria", "Data may be stale"]
        
        response = AnalysisResponse(
            results=results,
            total_regions_analyzed=10,
            filtered_regions_count=0,
            data_collection_timestamp=timestamp,
            warnings=warnings
        )
        
        assert response.warnings == warnings
        assert response.has_warnings is True

    def test_add_warning_method(self):
        """Test adding warning messages."""
        timestamp = datetime.now(timezone.utc)
        response = AnalysisResponse(
            results=[],
            total_regions_analyzed=10,
            filtered_regions_count=0,
            data_collection_timestamp=timestamp
        )
        
        assert response.has_warnings is False
        
        response.add_warning("Test warning")
        assert response.has_warnings is True
        assert "Test warning" in response.warnings
        
        response.add_warning("Another warning")
        assert len(response.warnings) == 2

    def test_result_count_property(self):
        """Test result count property."""
        timestamp = datetime.now(timezone.utc)
        results = [
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                rank=1,
                data_timestamp=timestamp
            ),
            SpotPriceResult(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=13.00,
                currency="USD",
                interruption_rate=0.02,
                rank=2,
                data_timestamp=timestamp
            )
        ]
        
        response = AnalysisResponse(
            results=results,
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=timestamp
        )
        
        assert response.result_count == 2

    def test_to_dict_method(self):
        """Test dictionary conversion for JSON serialization."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        results = [
            SpotPriceResult(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                rank=1,
                data_timestamp=timestamp
            )
        ]
        warnings = ["Test warning"]
        
        response = AnalysisResponse(
            results=results,
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=timestamp,
            warnings=warnings
        )
        
        result_dict = response.to_dict()
        
        assert result_dict["total_regions_analyzed"] == 10
        assert result_dict["filtered_regions_count"] == 5
        assert result_dict["data_collection_timestamp"] == "2024-01-15T10:30:00+00:00"
        assert result_dict["warnings"] == ["Test warning"]
        assert len(result_dict["results"]) == 1

    def test_analysis_response_validation_invalid_filtered_count(self):
        """Test validation fails when filtered count exceeds total."""
        timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValueError, match="Filtered regions count cannot exceed total regions analyzed"):
            AnalysisResponse(
                results=[],
                total_regions_analyzed=5,
                filtered_regions_count=10,
                data_collection_timestamp=timestamp
            )


class TestRawSpotDataValidator:
    """Test cases for Pydantic RawSpotDataValidator."""

    def test_valid_data_validation(self):
        """Test validation of valid data."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "region": "us-east-1",
            "instance_type": "p5en.48xlarge",
            "spot_price": 12.50,
            "currency": "USD",
            "interruption_rate": 0.03,
            "timestamp": timestamp,
            "availability": True
        }
        
        validator = RawSpotDataValidator(**data)
        assert validator.region == "us-east-1"
        assert validator.spot_price == 12.50

    def test_invalid_currency_validation(self):
        """Test validation fails for invalid currency."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "region": "us-east-1",
            "instance_type": "p5en.48xlarge",
            "spot_price": 12.50,
            "currency": "EUR",
            "interruption_rate": 0.03,
            "timestamp": timestamp,
            "availability": True
        }
        
        with pytest.raises(ValidationError):
            RawSpotDataValidator(**data)

    def test_invalid_instance_type_validation(self):
        """Test validation fails for invalid instance type format."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "region": "us-east-1",
            "instance_type": "invalid",
            "spot_price": 12.50,
            "currency": "USD",
            "interruption_rate": 0.03,
            "timestamp": timestamp,
            "availability": True
        }
        
        with pytest.raises(ValidationError):
            RawSpotDataValidator(**data)


class TestValidationFunctions:
    """Test cases for validation helper functions."""

    def test_validate_raw_spot_data_success(self):
        """Test successful validation of raw spot data."""
        timestamp = datetime.now(timezone.utc)
        data = {
            "region": "us-east-1",
            "instance_type": "p5en.48xlarge",
            "spot_price": 12.50,
            "currency": "USD",
            "interruption_rate": 0.03,
            "timestamp": timestamp,
            "availability": True
        }
        
        result = validate_raw_spot_data(data)
        assert isinstance(result, RawSpotData)
        assert result.region == "us-east-1"

    def test_validate_raw_spot_data_failure(self):
        """Test validation failure for invalid raw spot data."""
        data = {
            "region": "",
            "instance_type": "p5en.48xlarge",
            "spot_price": 12.50,
            "currency": "USD",
            "interruption_rate": 0.03,
            "timestamp": datetime.now(timezone.utc),
            "availability": True
        }
        
        with pytest.raises(ValueError, match="Invalid spot data"):
            validate_raw_spot_data(data)

    def test_validate_spot_price_results_success(self):
        """Test successful validation of spot price results."""
        timestamp = datetime.now(timezone.utc)
        results_data = [
            {
                "region": "us-east-1",
                "instance_type": "p5en.48xlarge",
                "spot_price": 12.50,
                "currency": "USD",
                "interruption_rate": 0.03,
                "rank": 1,
                "data_timestamp": timestamp
            }
        ]
        
        results = validate_spot_price_results(results_data)
        assert len(results) == 1
        assert isinstance(results[0], SpotPriceResult)
        assert results[0].region == "us-east-1"

    def test_validate_spot_price_results_failure(self):
        """Test validation failure for invalid spot price results."""
        results_data = [
            {
                "region": "us-east-1",
                "instance_type": "p5en.48xlarge",
                "spot_price": -12.50,  # Invalid negative price
                "currency": "USD",
                "interruption_rate": 0.03,
                "rank": 1,
                "data_timestamp": datetime.now(timezone.utc)
            }
        ]
        
        with pytest.raises(ValueError, match="Invalid result data at index 0"):
            validate_spot_price_results(results_data)


# Integration test for complete workflow
class TestDataModelIntegration:
    """Integration tests for data model workflow."""

    def test_complete_data_flow(self):
        """Test complete data flow from raw data to analysis response."""
        timestamp = datetime.now(timezone.utc)
        
        # Create raw data
        raw_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=timestamp
        )
        
        # Convert to result
        result = SpotPriceResult(
            region=raw_data.region,
            instance_type=raw_data.instance_type,
            spot_price=raw_data.spot_price,
            currency=raw_data.currency,
            interruption_rate=raw_data.interruption_rate,
            rank=1,
            data_timestamp=raw_data.timestamp
        )
        
        # Create analysis response
        response = AnalysisResponse(
            results=[result],
            total_regions_analyzed=10,
            filtered_regions_count=5,
            data_collection_timestamp=timestamp
        )
        
        # Verify complete flow
        assert response.result_count == 1
        assert response.results[0].region == "us-east-1"
        assert response.to_dict()["results"][0]["interruption_rate"] == "3.00%"