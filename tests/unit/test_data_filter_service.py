"""
Unit tests for DataFilterService.

Tests cover data filtering functionality, interruption rate filtering,
data validation, and various edge cases with comprehensive scenarios.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from src.services.data_filter_service import DataFilterService, DataFilterServiceError
from src.models.spot_data import RawSpotData


class TestDataFilterService:
    """Test cases for DataFilterService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataFilterService()
        
        # Create test data with various scenarios
        self.current_time = datetime.now(timezone.utc)
        
        self.valid_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,  # 3% - below threshold
                timestamp=self.current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.02,  # 2% - below threshold
                timestamp=self.current_time,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=14.00,
                currency="USD",
                interruption_rate=0.04,  # 4% - below threshold
                timestamp=self.current_time,
                availability=True
            )
        ]
        
        self.high_interruption_data = [
            RawSpotData(
                region="ap-south-1",
                instance_type="p5en.48xlarge",
                spot_price=10.00,  # Cheap but high interruption
                currency="USD",
                interruption_rate=0.08,  # 8% - above threshold
                timestamp=self.current_time,
                availability=True
            ),
            RawSpotData(
                region="sa-east-1",
                instance_type="p5.48xlarge",
                spot_price=11.00,
                currency="USD",
                interruption_rate=0.06,  # 6% - above threshold
                timestamp=self.current_time,
                availability=True
            )
        ]

    def test_initialization_default_values(self):
        """Test service initialization with default values."""
        service = DataFilterService()
        
        assert service.max_interruption_rate == DataFilterService.DEFAULT_MAX_INTERRUPTION_RATE
        assert service.max_interruption_rate == 0.05  # 5%
        assert service._last_filter_stats == {}

    def test_initialization_custom_max_rate(self):
        """Test service initialization with custom max interruption rate."""
        custom_rate = 0.03  # 3%
        service = DataFilterService(max_interruption_rate=custom_rate)
        
        assert service.max_interruption_rate == custom_rate

    def test_filter_by_interruption_rate_success(self):
        """Test successful interruption rate filtering."""
        # Combine valid and high interruption data
        all_data = self.valid_data + self.high_interruption_data
        
        result = self.service.filter_by_interruption_rate(all_data)
        
        # Should only return the 3 valid records (below 5% threshold)
        assert len(result) == 3
        assert all(data.interruption_rate < 0.05 for data in result)
        
        # Check that high interruption regions are excluded
        result_regions = [data.region for data in result]
        assert "ap-south-1" not in result_regions
        assert "sa-east-1" not in result_regions

    def test_filter_by_interruption_rate_custom_threshold(self):
        """Test interruption rate filtering with custom threshold."""
        # Use stricter threshold of 3%
        custom_threshold = 0.03
        all_data = self.valid_data + self.high_interruption_data
        
        result = self.service.filter_by_interruption_rate(all_data, max_rate=custom_threshold)
        
        # Should only return records with interruption rate < 3%
        # us-east-1 has exactly 3% which should be excluded (< 3%, not <= 3%)
        # Only us-west-2 (2%) should pass
        assert len(result) == 1  # Only us-west-2 (2%)
        assert all(data.interruption_rate < custom_threshold for data in result)
        
        # us-east-1 has 3% which equals the threshold, eu-west-1 has 4%
        result_regions = [data.region for data in result]
        assert "us-east-1" not in result_regions  # 3% equals threshold
        assert "eu-west-1" not in result_regions  # 4% above threshold
        assert "us-west-2" in result_regions      # 2% below threshold

    def test_filter_by_interruption_rate_empty_data(self):
        """Test interruption rate filtering with empty data."""
        result = self.service.filter_by_interruption_rate([])
        
        assert result == []

    def test_filter_by_interruption_rate_all_excluded(self):
        """Test interruption rate filtering when all data is excluded."""
        # Use very strict threshold
        strict_threshold = 0.01  # 1%
        
        result = self.service.filter_by_interruption_rate(self.valid_data, max_rate=strict_threshold)
        
        assert result == []

    def test_validate_price_data_success(self):
        """Test successful price data validation."""
        result = self.service.validate_price_data(self.valid_data)
        
        assert len(result) == 3
        assert result == self.valid_data

    def test_validate_price_data_missing_price(self):
        """Test price data validation with missing price."""
        # Create valid data first, then modify to have invalid price
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=self.current_time,
            availability=True
        )
        
        # Bypass validation by directly setting the attribute
        valid_data.spot_price = None
        invalid_data = [valid_data]
        
        result = self.service.validate_price_data(invalid_data)
        
        assert len(result) == 0

    def test_validate_price_data_invalid_price(self):
        """Test price data validation with invalid price."""
        # Create valid data first, then modify to have invalid prices
        data1 = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=self.current_time,
            availability=True
        )
        data2 = RawSpotData(
            region="us-west-2",
            instance_type="p5.48xlarge",
            spot_price=15.00,
            currency="USD",
            interruption_rate=0.02,
            timestamp=self.current_time,
            availability=True
        )
        
        # Bypass validation by directly setting invalid prices
        data1.spot_price = -5.00  # Negative price
        data2.spot_price = 0.00   # Zero price
        
        invalid_data = [data1, data2]
        
        result = self.service.validate_price_data(invalid_data)
        
        assert len(result) == 0

    def test_validate_price_data_missing_region(self):
        """Test price data validation with missing region."""
        # Create valid data first, then modify to have empty region
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=self.current_time,
            availability=True
        )
        
        # Bypass validation by directly setting empty region
        valid_data.region = ""
        invalid_data = [valid_data]
        
        result = self.service.validate_price_data(invalid_data)
        
        assert len(result) == 0

    def test_validate_price_data_invalid_currency(self):
        """Test price data validation with invalid currency."""
        # Create valid data first, then modify to have invalid currency
        valid_data = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=self.current_time,
            availability=True
        )
        
        # Bypass validation by directly setting invalid currency
        valid_data.currency = "EUR"
        invalid_data = [valid_data]
        
        result = self.service.validate_price_data(invalid_data)
        
        assert len(result) == 0

    def test_validate_price_data_stale_data(self):
        """Test price data validation with stale timestamp."""
        old_timestamp = self.current_time - timedelta(hours=25)  # 25 hours old
        
        stale_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=old_timestamp,  # Too old
                availability=True
            )
        ]
        
        result = self.service.validate_price_data(stale_data)
        
        assert len(result) == 0

    def test_validate_price_data_mixed_valid_invalid(self):
        """Test price data validation with mix of valid and invalid data."""
        # Create all valid data first
        data1 = RawSpotData(
            region="us-east-1",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=self.current_time,
            availability=True
        )
        data2 = RawSpotData(
            region="us-west-2",
            instance_type="p5.48xlarge",
            spot_price=15.00,
            currency="USD",
            interruption_rate=0.02,
            timestamp=self.current_time,
            availability=True
        )
        data3 = RawSpotData(
            region="eu-west-1",
            instance_type="p5en.48xlarge",
            spot_price=14.00,
            currency="USD",
            interruption_rate=0.04,
            timestamp=self.current_time,
            availability=True
        )
        
        # Modify to create invalid data
        data2.spot_price = -5.00  # Invalid price
        data3.currency = "EUR"    # Invalid currency
        
        mixed_data = [data1, data2, data3]
        
        result = self.service.validate_price_data(mixed_data)
        
        assert len(result) == 1
        assert result[0].region == "us-east-1"

    def test_filter_and_validate_success(self):
        """Test combined filtering and validation."""
        # Create invalid data by modifying valid data
        invalid_data = RawSpotData(
            region="invalid-region",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.02,
            timestamp=self.current_time,
            availability=True
        )
        # Modify to make it invalid
        invalid_data.spot_price = -1.00
        
        # Mix valid data with high interruption and invalid data
        all_data = self.valid_data + self.high_interruption_data + [invalid_data]
        
        result = self.service.filter_and_validate(all_data)
        
        # Should only return valid data with low interruption rates
        assert len(result) == 3
        assert all(data.interruption_rate < 0.05 for data in result)
        assert all(data.spot_price > 0 for data in result)

    def test_get_filter_statistics(self):
        """Test filter statistics tracking."""
        # Perform filtering to generate statistics
        all_data = self.valid_data + self.high_interruption_data
        self.service.filter_by_interruption_rate(all_data)
        
        stats = self.service.get_filter_statistics()
        
        assert "input_count" in stats
        assert "filtered_count" in stats
        assert "excluded_by_interruption_rate" in stats
        assert stats["input_count"] == 5
        assert stats["filtered_count"] == 3
        assert stats["excluded_by_interruption_rate"] == 2

    def test_set_max_interruption_rate_valid(self):
        """Test setting valid max interruption rate."""
        new_rate = 0.03  # 3%
        
        self.service.set_max_interruption_rate(new_rate)
        
        assert self.service.max_interruption_rate == new_rate

    def test_set_max_interruption_rate_invalid(self):
        """Test setting invalid max interruption rate."""
        with pytest.raises(DataFilterServiceError, match="must be between 0.0 and 1.0"):
            self.service.set_max_interruption_rate(-0.1)
        
        with pytest.raises(DataFilterServiceError, match="must be between 0.0 and 1.0"):
            self.service.set_max_interruption_rate(1.5)
        
        with pytest.raises(DataFilterServiceError, match="must be between 0.0 and 1.0"):
            self.service.set_max_interruption_rate("invalid")

    def test_filter_by_instance_type(self):
        """Test filtering by specific instance types."""
        # Filter for only p5en.48xlarge
        result = self.service.filter_by_instance_type(
            self.valid_data,
            ["p5en.48xlarge"]
        )
        
        assert len(result) == 2  # us-east-1 and eu-west-1
        assert all(data.instance_type == "p5en.48xlarge" for data in result)

    def test_filter_by_instance_type_empty_types(self):
        """Test filtering with empty instance types list."""
        result = self.service.filter_by_instance_type(self.valid_data, [])
        
        # Should return all data when no types specified
        assert result == self.valid_data

    def test_filter_by_availability(self):
        """Test filtering by availability."""
        # Create data with mixed availability
        mixed_availability_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=self.current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.02,
                timestamp=self.current_time,
                availability=False  # Not available
            )
        ]
        
        result = self.service.filter_by_availability(mixed_availability_data)
        
        assert len(result) == 1
        assert result[0].region == "us-east-1"
        assert result[0].availability is True

    def test_get_supported_instance_types(self):
        """Test getting supported instance types."""
        types = self.service.get_supported_instance_types()
        
        assert types == ["p5en.48xlarge", "p5.48xlarge"]

    def test_clear_statistics(self):
        """Test clearing filter statistics."""
        # Generate some statistics
        self.service.filter_by_interruption_rate(self.valid_data)
        assert self.service.get_filter_statistics()  # Should have data
        
        # Clear statistics
        self.service.clear_statistics()
        assert self.service.get_filter_statistics() == {}

    def test_validate_price_data_empty_data(self):
        """Test price data validation with empty data."""
        result = self.service.validate_price_data([])
        
        assert result == []

    def test_filter_by_interruption_rate_statistics_update(self):
        """Test that statistics are properly updated during filtering."""
        all_data = self.valid_data + self.high_interruption_data
        
        self.service.filter_by_interruption_rate(all_data)
        
        stats = self.service.get_filter_statistics()
        assert stats["input_count"] == 5
        assert stats["filtered_count"] == 3
        assert stats["excluded_by_interruption_rate"] == 2
        assert stats["max_interruption_rate"] == 0.05

    def test_validate_price_data_statistics_update(self):
        """Test that validation statistics are properly updated."""
        # Create valid data first, then modify to create invalid data
        data1 = self.valid_data[0]  # Valid data
        
        data2 = RawSpotData(
            region="invalid-price",
            instance_type="p5en.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.03,
            timestamp=self.current_time,
            availability=True
        )
        data2.spot_price = -1.00  # Make invalid
        
        data3 = RawSpotData(
            region="invalid-currency",
            instance_type="p5.48xlarge",
            spot_price=12.50,
            currency="USD",
            interruption_rate=0.02,
            timestamp=self.current_time,
            availability=True
        )
        data3.currency = "EUR"  # Make invalid
        
        mixed_data = [data1, data2, data3]
        
        self.service.validate_price_data(mixed_data)
        
        stats = self.service.get_filter_statistics()
        assert stats["validation_input_count"] == 3
        assert stats["validation_valid_count"] == 1
        assert stats["validation_excluded_count"] == 2
        assert "validation_failures" in stats
        assert stats["validation_failures"]["invalid_price"] == 1
        assert stats["validation_failures"]["invalid_currency"] == 1


class TestDataFilterServiceIntegration:
    """Integration tests for DataFilterService."""

    def test_complete_filtering_workflow(self):
        """Test complete filtering workflow with realistic data."""
        service = DataFilterService()
        current_time = datetime.now(timezone.utc)
        
        # Create comprehensive test dataset
        test_data = [
            # Valid, low interruption
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=current_time,
                availability=True
            ),
            # Valid, low interruption
            RawSpotData(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.02,
                timestamp=current_time,
                availability=True
            ),
            # High interruption (should be filtered out)
            RawSpotData(
                region="ap-south-1",
                instance_type="p5en.48xlarge",
                spot_price=10.00,
                currency="USD",
                interruption_rate=0.08,
                timestamp=current_time,
                availability=True
            ),
            # Invalid price (should be filtered out) - create valid first then modify
            RawSpotData(
                region="eu-west-1",
                instance_type="p5.48xlarge",
                spot_price=15.00,  # Will be modified to -5.00 after creation
                currency="USD",
                interruption_rate=0.04,
                timestamp=current_time,
                availability=True
            ),
            # Unavailable (should pass other filters but can be filtered by availability)
            RawSpotData(
                region="ca-central-1",
                instance_type="p5en.48xlarge",
                spot_price=13.00,
                currency="USD",
                interruption_rate=0.03,
                timestamp=current_time,
                availability=False
            )
        ]
        
        # Modify the invalid price data after creation
        test_data[3].spot_price = -5.00  # Make eu-west-1 have invalid price
        
        # Apply complete filtering workflow
        result = service.filter_and_validate(test_data)
        
        # Should return 3 valid records with low interruption (availability is not filtered by filter_and_validate)
        assert len(result) == 3
        result_regions = [data.region for data in result]
        assert "us-east-1" in result_regions
        assert "us-west-2" in result_regions
        assert "ca-central-1" in result_regions    # Unavailable but passes validation and interruption filtering
        
        # Verify excluded regions
        assert "ap-south-1" not in result_regions  # High interruption
        assert "eu-west-1" not in result_regions   # Invalid price
        
        # Check statistics
        stats = service.get_filter_statistics()
        assert stats["validation_input_count"] == 5
        assert stats["validation_valid_count"] == 4  # 4 valid after price validation
        assert stats["filtered_count"] == 3         # 3 after interruption filtering (includes unavailable but valid data)

    def test_edge_case_handling(self):
        """Test handling of various edge cases."""
        service = DataFilterService()
        
        # Test with None values and edge cases
        edge_case_data = []
        
        # Empty data
        result = service.filter_and_validate(edge_case_data)
        assert result == []
        
        # Single valid record
        single_valid = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=datetime.now(timezone.utc),
                availability=True
            )
        ]
        
        result = service.filter_and_validate(single_valid)
        assert len(result) == 1
        assert result[0].region == "us-east-1"

    def test_custom_thresholds_workflow(self):
        """Test workflow with custom filtering thresholds."""
        # Use stricter threshold
        service = DataFilterService(max_interruption_rate=0.025)  # 2.5%
        current_time = datetime.now(timezone.utc)
        
        test_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=12.50,
                currency="USD",
                interruption_rate=0.02,  # 2% - below 2.5%
                timestamp=current_time,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.03,  # 3% - above 2.5%
                timestamp=current_time,
                availability=True
            )
        ]
        
        result = service.filter_and_validate(test_data)
        
        # Only the first record should pass the stricter threshold
        assert len(result) == 1
        assert result[0].region == "us-east-1"
        assert result[0].interruption_rate == 0.02