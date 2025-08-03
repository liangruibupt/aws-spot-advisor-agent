"""
Unit tests for RankingEngine class.

Tests the ranking and sorting functionality for spot pricing data,
including edge cases and error handling scenarios.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.services.ranking_engine import RankingEngine
from src.models.spot_data import RawSpotData, SpotPriceResult


class TestRankingEngine:
    """Test cases for RankingEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RankingEngine()
        self.timestamp = datetime.now(timezone.utc)
        
        # Create sample data for testing
        self.sample_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=15.50,
                currency="USD",
                interruption_rate=0.03,
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=12.25,
                currency="USD",
                interruption_rate=0.04,
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=12.25,  # Same price as us-west-2
                currency="USD",
                interruption_rate=0.02,  # Lower interruption rate
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="ap-southeast-1",
                instance_type="p5en.48xlarge",
                spot_price=18.75,
                currency="USD",
                interruption_rate=0.01,
                timestamp=self.timestamp,
                availability=True
            )
        ]

    def test_init(self):
        """Test RankingEngine initialization."""
        engine = RankingEngine()
        assert engine is not None
        assert hasattr(engine, '_logger')

    def test_rank_by_price_basic_sorting(self):
        """Test basic price-based sorting functionality."""
        result = self.engine.rank_by_price(self.sample_data)
        
        assert len(result) == 4
        assert isinstance(result, list)
        
        # Check that results are sorted by price
        prices = [item.spot_price for item in result]
        assert prices == sorted(prices)
        
        # First should be lowest price
        assert result[0].spot_price == 12.25
        assert result[-1].spot_price == 18.75

    def test_rank_by_price_secondary_sort_by_interruption_rate(self):
        """Test secondary sorting by interruption rate when prices are equal."""
        result = self.engine.rank_by_price(self.sample_data)
        
        # Find the two regions with same price (12.25)
        same_price_items = [item for item in result if item.spot_price == 12.25]
        assert len(same_price_items) == 2
        
        # eu-west-1 should come first due to lower interruption rate (0.02 vs 0.04)
        assert same_price_items[0].region == "eu-west-1"
        assert same_price_items[0].interruption_rate == 0.02
        assert same_price_items[1].region == "us-west-2"
        assert same_price_items[1].interruption_rate == 0.04

    def test_rank_by_price_empty_data(self):
        """Test ranking with empty data."""
        result = self.engine.rank_by_price([])
        assert result == []

    def test_rank_by_price_single_item(self):
        """Test ranking with single item."""
        single_item = [self.sample_data[0]]
        result = self.engine.rank_by_price(single_item)
        
        assert len(result) == 1
        assert result[0] == single_item[0]

    def test_rank_by_price_invalid_input_type(self):
        """Test ranking with invalid input type."""
        with pytest.raises(ValueError, match="Data must be a list of RawSpotData"):
            self.engine.rank_by_price("not a list")

    def test_rank_by_price_invalid_data_items(self):
        """Test ranking with invalid data items."""
        invalid_data = [self.sample_data[0], "not a RawSpotData", self.sample_data[1]]
        
        with pytest.raises(ValueError, match="Item at index 1 is not RawSpotData instance"):
            self.engine.rank_by_price(invalid_data)

    def test_get_top_regions_basic_functionality(self):
        """Test basic top regions selection."""
        ranked_data = self.engine.rank_by_price(self.sample_data)
        result = self.engine.get_top_regions(ranked_data, 3)
        
        assert len(result) == 3
        assert all(isinstance(item, SpotPriceResult) for item in result)
        
        # Check ranking order
        assert result[0].rank == 1
        assert result[1].rank == 2
        assert result[2].rank == 3
        
        # Check that lowest price is rank 1
        assert result[0].spot_price <= result[1].spot_price <= result[2].spot_price

    def test_get_top_regions_default_count(self):
        """Test get_top_regions with default count of 3."""
        ranked_data = self.engine.rank_by_price(self.sample_data)
        result = self.engine.get_top_regions(ranked_data)
        
        assert len(result) == 3

    def test_get_top_regions_fewer_than_requested(self):
        """Test get_top_regions when fewer items available than requested."""
        single_item = [self.sample_data[0]]
        result = self.engine.get_top_regions(single_item, 3)
        
        assert len(result) == 1
        assert result[0].rank == 1

    def test_get_top_regions_exact_count(self):
        """Test get_top_regions with exact count match."""
        result = self.engine.get_top_regions(self.sample_data, 4)
        
        assert len(result) == 4
        assert result[0].rank == 1
        assert result[3].rank == 4

    def test_get_top_regions_empty_data(self):
        """Test get_top_regions with empty data."""
        result = self.engine.get_top_regions([], 3)
        assert result == []

    def test_get_top_regions_invalid_count(self):
        """Test get_top_regions with invalid count values."""
        with pytest.raises(ValueError, match="Count must be a positive integer"):
            self.engine.get_top_regions(self.sample_data, 0)
        
        with pytest.raises(ValueError, match="Count must be a positive integer"):
            self.engine.get_top_regions(self.sample_data, -1)

    def test_get_top_regions_invalid_input_type(self):
        """Test get_top_regions with invalid input type."""
        with pytest.raises(ValueError, match="Data must be a list of RawSpotData"):
            self.engine.get_top_regions("not a list", 3)

    def test_get_top_regions_invalid_data_items(self):
        """Test get_top_regions with invalid data items."""
        invalid_data = [self.sample_data[0], "not a RawSpotData"]
        
        with pytest.raises(ValueError, match="Item at index 1 is not RawSpotData instance"):
            self.engine.get_top_regions(invalid_data, 2)

    def test_get_top_regions_result_structure(self):
        """Test that SpotPriceResult objects are properly constructed."""
        ranked_data = self.engine.rank_by_price(self.sample_data)
        result = self.engine.get_top_regions(ranked_data, 2)
        
        first_result = result[0]
        original_data = ranked_data[0]
        
        assert first_result.region == original_data.region
        assert first_result.instance_type == original_data.instance_type
        assert first_result.spot_price == original_data.spot_price
        assert first_result.currency == original_data.currency
        assert first_result.interruption_rate == original_data.interruption_rate
        assert first_result.rank == 1
        assert first_result.data_timestamp == original_data.timestamp

    def test_rank_and_get_top_convenience_method(self):
        """Test the convenience method that combines ranking and top selection."""
        result = self.engine.rank_and_get_top(self.sample_data, 2)
        
        assert len(result) == 2
        assert all(isinstance(item, SpotPriceResult) for item in result)
        assert result[0].rank == 1
        assert result[1].rank == 2
        
        # Should be same as calling methods separately
        ranked_data = self.engine.rank_by_price(self.sample_data)
        separate_result = self.engine.get_top_regions(ranked_data, 2)
        
        assert len(result) == len(separate_result)
        for i in range(len(result)):
            assert result[i].region == separate_result[i].region
            assert result[i].spot_price == separate_result[i].spot_price
            assert result[i].rank == separate_result[i].rank

    def test_rank_and_get_top_empty_data(self):
        """Test convenience method with empty data."""
        result = self.engine.rank_and_get_top([], 3)
        assert result == []

    def test_get_ranking_summary_basic_stats(self):
        """Test ranking summary statistics."""
        summary = self.engine.get_ranking_summary(self.sample_data)
        
        assert summary["total_regions"] == 4
        assert summary["price_range"]["min"] == 12.25
        assert summary["price_range"]["max"] == 18.75
        assert summary["interruption_rate_range"]["min"] == 0.01
        assert summary["interruption_rate_range"]["max"] == 0.04
        
        # Check averages
        expected_avg_price = (15.50 + 12.25 + 12.25 + 18.75) / 4
        expected_avg_interruption = (0.03 + 0.04 + 0.02 + 0.01) / 4
        
        assert abs(summary["average_price"] - expected_avg_price) < 0.001
        assert abs(summary["average_interruption_rate"] - expected_avg_interruption) < 0.001

    def test_get_ranking_summary_empty_data(self):
        """Test ranking summary with empty data."""
        summary = self.engine.get_ranking_summary([])
        
        assert summary["total_regions"] == 0
        assert summary["price_range"]["min"] is None
        assert summary["price_range"]["max"] is None
        assert summary["interruption_rate_range"]["min"] is None
        assert summary["interruption_rate_range"]["max"] is None
        assert summary["average_price"] is None
        assert summary["average_interruption_rate"] is None

    def test_get_ranking_summary_single_item(self):
        """Test ranking summary with single item."""
        single_item = [self.sample_data[0]]
        summary = self.engine.get_ranking_summary(single_item)
        
        assert summary["total_regions"] == 1
        assert summary["price_range"]["min"] == 15.50
        assert summary["price_range"]["max"] == 15.50
        assert summary["interruption_rate_range"]["min"] == 0.03
        assert summary["interruption_rate_range"]["max"] == 0.03
        assert summary["average_price"] == 15.50
        assert summary["average_interruption_rate"] == 0.03

    def test_edge_case_identical_prices_and_interruption_rates(self):
        """Test ranking when multiple regions have identical prices and interruption rates."""
        identical_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=10.00,
                currency="USD",
                interruption_rate=0.03,
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=10.00,
                currency="USD",
                interruption_rate=0.03,
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="eu-west-1",
                instance_type="p5en.48xlarge",
                spot_price=10.00,
                currency="USD",
                interruption_rate=0.03,
                timestamp=self.timestamp,
                availability=True
            )
        ]
        
        result = self.engine.rank_by_price(identical_data)
        assert len(result) == 3
        
        # All should have same price and interruption rate
        for item in result:
            assert item.spot_price == 10.00
            assert item.interruption_rate == 0.03
        
        # Test top regions selection
        top_results = self.engine.get_top_regions(result, 2)
        assert len(top_results) == 2
        assert top_results[0].rank == 1
        assert top_results[1].rank == 2

    def test_edge_case_zero_prices(self):
        """Test ranking with zero spot prices."""
        zero_price_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=0.0,
                currency="USD",
                interruption_rate=0.03,
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=5.00,
                currency="USD",
                interruption_rate=0.02,
                timestamp=self.timestamp,
                availability=True
            )
        ]
        
        result = self.engine.rank_by_price(zero_price_data)
        assert result[0].spot_price == 0.0
        assert result[0].region == "us-east-1"

    def test_edge_case_very_high_interruption_rates(self):
        """Test ranking with high interruption rates (edge of valid range)."""
        high_interruption_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=10.00,
                currency="USD",
                interruption_rate=0.99,  # Very high but valid
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=10.00,
                currency="USD",
                interruption_rate=1.0,  # Maximum valid value
                timestamp=self.timestamp,
                availability=True
            )
        ]
        
        result = self.engine.rank_by_price(high_interruption_data)
        # us-east-1 should come first due to lower interruption rate
        assert result[0].region == "us-east-1"
        assert result[0].interruption_rate == 0.99

    def test_logging_behavior(self):
        """Test that appropriate logging occurs during ranking operations."""
        with patch.object(self.engine, '_logger') as mock_logger:
            # Test ranking logging
            self.engine.rank_by_price(self.sample_data)
            
            # Test top regions logging
            ranked_data = self.engine.rank_by_price(self.sample_data)
            self.engine.get_top_regions(ranked_data, 2)
            
            # Verify logger was used (exact calls depend on implementation)
            assert mock_logger.info.called or mock_logger.debug.called or mock_logger.warning.called

    def test_requirements_compliance_1_3(self):
        """Test compliance with requirement 1.3: return exactly 3 regions ranked by lowest spot price."""
        # Create data with more than 3 regions
        extended_data = self.sample_data + [
            RawSpotData(
                region="ca-central-1",
                instance_type="p5en.48xlarge",
                spot_price=20.00,
                currency="USD",
                interruption_rate=0.02,
                timestamp=self.timestamp,
                availability=True
            )
        ]
        
        result = self.engine.rank_and_get_top(extended_data, 3)
        
        # Should return exactly 3 results
        assert len(result) == 3
        
        # Should be ranked by price (lowest first)
        assert result[0].spot_price <= result[1].spot_price <= result[2].spot_price
        
        # Should have proper ranking numbers
        assert result[0].rank == 1
        assert result[1].rank == 2
        assert result[2].rank == 3

    def test_requirements_compliance_3_4(self):
        """Test compliance with requirement 3.4: use interruption rate as secondary sort criterion."""
        # Create data where multiple regions have same price
        same_price_data = [
            RawSpotData(
                region="us-east-1",
                instance_type="p5en.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.04,  # Higher interruption rate
                timestamp=self.timestamp,
                availability=True
            ),
            RawSpotData(
                region="us-west-2",
                instance_type="p5en.48xlarge",
                spot_price=15.00,
                currency="USD",
                interruption_rate=0.02,  # Lower interruption rate
                timestamp=self.timestamp,
                availability=True
            )
        ]
        
        result = self.engine.rank_by_price(same_price_data)
        
        # us-west-2 should come first due to lower interruption rate
        assert result[0].region == "us-west-2"
        assert result[0].interruption_rate == 0.02
        assert result[1].region == "us-east-1"
        assert result[1].interruption_rate == 0.04