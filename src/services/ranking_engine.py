"""
Ranking and sorting engine for AWS spot price analysis.

This module provides functionality to rank and sort spot pricing data
based on price and secondary criteria like interruption rates.
"""

from typing import List, Optional
from datetime import datetime
import logging

from src.models.spot_data import RawSpotData, SpotPriceResult
from src.utils.exceptions import (
    RankingError,
    DataValidationError,
    InsufficientDataError
)
from src.utils.retry_utils import data_processing_retry

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    Engine for ranking and sorting spot pricing data.
    
    This class provides methods to sort regions by spot price with
    secondary sorting by interruption rate, and to extract the top
    N results for analysis.
    """

    def __init__(self):
        """Initialize the ranking engine."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @data_processing_retry(max_attempts=2)
    def rank_by_price(self, data: List[RawSpotData]) -> List[RawSpotData]:
        """
        Rank spot data by price with secondary sorting by interruption rate.
        
        Primary sort: Ascending by spot_price (lowest price first)
        Secondary sort: Ascending by interruption_rate (lowest interruption first)
        
        Args:
            data: List of RawSpotData to rank
            
        Returns:
            List of RawSpotData sorted by price and interruption rate
            
        Raises:
            RankingError: If ranking fails
            DataValidationError: If data validation fails
        """
        if not data:
            self._logger.warning("No data provided for ranking")
            return []
        
        try:
            if not isinstance(data, list):
                raise DataValidationError(
                    message="Data must be a list of RawSpotData",
                    field_name="data",
                    field_value=type(data).__name__,
                    validation_rule="must be list"
                )
            
            # Validate all entries are RawSpotData
            for i, item in enumerate(data):
                if not isinstance(item, RawSpotData):
                    raise DataValidationError(
                        message=f"Item at index {i} is not RawSpotData instance",
                        field_name=f"data[{i}]",
                        field_value=type(item).__name__,
                        validation_rule="must be RawSpotData"
                    )
            
            self._logger.info(f"Ranking {len(data)} regions by price and interruption rate")
            
            # Sort by price (ascending), then by interruption rate (ascending)
            sorted_data = sorted(
                data,
                key=lambda x: (x.spot_price, x.interruption_rate)
            )
            
            self._logger.debug(f"Ranking complete. Best price: ${sorted_data[0].spot_price:.4f} "
                              f"in {sorted_data[0].region}")
            
            return sorted_data
            
        except (DataValidationError, RankingError):
            # Re-raise validation and ranking errors as-is
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error during ranking: {e}")
            raise RankingError(
                message=f"Failed to rank data by price: {e}",
                ranking_criteria="price_and_interruption_rate",
                data_count=len(data) if isinstance(data, list) else 0,
                original_error=e
            )

    def get_top_regions(self, data: List[RawSpotData], count: int = 3) -> List[SpotPriceResult]:
        """
        Get the top N regions from ranked data.
        
        Args:
            data: List of RawSpotData (should be pre-sorted by rank_by_price)
            count: Number of top regions to return (default: 3)
            
        Returns:
            List of SpotPriceResult with exactly 'count' results (or fewer if insufficient data)
            
        Raises:
            RankingError: If ranking fails
            DataValidationError: If data validation fails
        """
        try:
            if count <= 0:
                raise DataValidationError(
                    message="Count must be a positive integer",
                    field_name="count",
                    field_value=count,
                    validation_rule="must be > 0"
                )
            
            if not data:
                self._logger.warning("No data provided for top regions selection")
                return []
            
            if not isinstance(data, list):
                raise DataValidationError(
                    message="Data must be a list of RawSpotData",
                    field_name="data",
                    field_value=type(data).__name__,
                    validation_rule="must be list"
                )
            
            # Validate all entries are RawSpotData
            for i, item in enumerate(data):
                if not isinstance(item, RawSpotData):
                    raise DataValidationError(
                        message=f"Item at index {i} is not RawSpotData instance",
                        field_name=f"data[{i}]",
                        field_value=type(item).__name__,
                        validation_rule="must be RawSpotData"
                    )
            
            # Take only the requested number of top results
            top_data = data[:count]
            
            self._logger.info(f"Selecting top {len(top_data)} regions from {len(data)} available")
            
            # Convert to SpotPriceResult with ranking information
            results = []
            for rank, spot_data in enumerate(top_data, start=1):
                try:
                    result = SpotPriceResult(
                        region=spot_data.region,
                        instance_type=spot_data.instance_type,
                        spot_price=spot_data.spot_price,
                        currency=spot_data.currency,
                        interruption_rate=spot_data.interruption_rate,
                        rank=rank,
                        data_timestamp=spot_data.timestamp
                    )
                    results.append(result)
                except Exception as e:
                    self._logger.warning(f"Failed to create SpotPriceResult for {spot_data.region}: {e}")
                    continue
            
            if results:
                self._logger.info(f"Top region: {results[0].region} "
                                 f"(${results[0].spot_price:.4f}, "
                                 f"{results[0].interruption_rate_percentage})")
            
            return results
            
        except (DataValidationError, RankingError):
            # Re-raise validation and ranking errors as-is
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error getting top regions: {e}")
            raise RankingError(
                message=f"Failed to get top regions: {e}",
                ranking_criteria="top_selection",
                data_count=len(data) if isinstance(data, list) else 0,
                original_error=e
            )

    def rank_and_get_top(self, data: List[RawSpotData], count: int = 3) -> List[SpotPriceResult]:
        """
        Convenience method to rank data and get top regions in one call.
        
        Args:
            data: List of RawSpotData to rank
            count: Number of top regions to return (default: 3)
            
        Returns:
            List of SpotPriceResult with top regions
            
        Raises:
            RankingError: If ranking fails
            DataValidationError: If data validation fails
        """
        try:
            if not data:
                return []
            
            # First rank the data
            ranked_data = self.rank_by_price(data)
            
            # Then get the top results
            return self.get_top_regions(ranked_data, count)
            
        except (DataValidationError, RankingError):
            # Re-raise validation and ranking errors as-is
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error in rank_and_get_top: {e}")
            raise RankingError(
                message=f"Failed to rank and get top regions: {e}",
                ranking_criteria="rank_and_get_top",
                data_count=len(data) if isinstance(data, list) else 0,
                original_error=e
            )

    def get_ranking_summary(self, data: List[RawSpotData]) -> dict:
        """
        Get summary statistics about the ranking data.
        
        Args:
            data: List of RawSpotData to analyze
            
        Returns:
            Dictionary with ranking statistics
            
        Raises:
            RankingError: If summary calculation fails
        """
        try:
            if not data:
                return {
                    "total_regions": 0,
                    "price_range": {"min": None, "max": None},
                    "interruption_rate_range": {"min": None, "max": None},
                    "average_price": None,
                    "average_interruption_rate": None
                }
            
            # Validate data and extract values
            prices = []
            interruption_rates = []
            
            for item in data:
                if not isinstance(item, RawSpotData):
                    continue  # Skip invalid items
                
                if isinstance(item.spot_price, (int, float)):
                    prices.append(item.spot_price)
                
                if isinstance(item.interruption_rate, (int, float)):
                    interruption_rates.append(item.interruption_rate)
            
            if not prices or not interruption_rates:
                self._logger.warning("No valid price or interruption rate data found")
                return {
                    "total_regions": len(data),
                    "price_range": {"min": None, "max": None},
                    "interruption_rate_range": {"min": None, "max": None},
                    "average_price": None,
                    "average_interruption_rate": None
                }
            
            return {
                "total_regions": len(data),
                "price_range": {
                    "min": min(prices),
                    "max": max(prices)
                },
                "interruption_rate_range": {
                    "min": min(interruption_rates),
                    "max": max(interruption_rates)
                },
                "average_price": sum(prices) / len(prices),
                "average_interruption_rate": sum(interruption_rates) / len(interruption_rates)
            }
            
        except Exception as e:
            self._logger.error(f"Error calculating ranking summary: {e}")
            raise RankingError(
                message=f"Failed to calculate ranking summary: {e}",
                ranking_criteria="summary_statistics",
                data_count=len(data) if isinstance(data, list) else 0,
                original_error=e
            )