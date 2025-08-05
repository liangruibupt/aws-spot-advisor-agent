"""
Data filtering service for AWS Spot Price Analyzer.

This module provides filtering functionality to process raw spot pricing data
and apply business rules such as interruption rate thresholds and data validation.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from src.models.spot_data import RawSpotData
from src.utils.exceptions import (
    DataFilteringError,
    DataValidationError,
    InsufficientDataError
)


logger = logging.getLogger(__name__)


# Keep the old exception for backward compatibility
class DataFilterServiceError(DataFilteringError):
    """Legacy exception for DataFilterService errors."""
    pass


class DataFilterService:
    """
    Service for filtering and validating spot pricing data.
    
    This service applies business rules to filter raw spot data based on
    interruption rates, data completeness, and other quality criteria.
    """

    # Default maximum interruption rate threshold (5%)
    DEFAULT_MAX_INTERRUPTION_RATE = 0.05

    def __init__(self, max_interruption_rate: float = DEFAULT_MAX_INTERRUPTION_RATE):
        """
        Initialize the Data Filter Service.
        
        Args:
            max_interruption_rate: Maximum acceptable interruption rate (default: 0.05 = 5%)
        """
        self.max_interruption_rate = max_interruption_rate
        
        # Validation statistics
        self._last_filter_stats: Dict[str, int] = {}

    def filter_by_interruption_rate(
        self,
        data: List[RawSpotData],
        max_rate: Optional[float] = None
    ) -> List[RawSpotData]:
        """
        Filter spot data by interruption rate threshold.
        
        Args:
            data: List of RawSpotData objects to filter
            max_rate: Maximum acceptable interruption rate (optional, uses instance default)
            
        Returns:
            List of RawSpotData objects with interruption rate below threshold
            
        Raises:
            DataFilterServiceError: If filtering fails
        """
        if not data:
            logger.warning("No data provided for interruption rate filtering")
            return []
        
        if max_rate is None:
            max_rate = self.max_interruption_rate
        
        try:
            logger.info(f"Filtering {len(data)} records by interruption rate <= {max_rate * 100:.1f}%")
            
            filtered_data = []
            excluded_count = 0
            
            for spot_data in data:
                if self._validate_interruption_rate(spot_data, max_rate):
                    filtered_data.append(spot_data)
                else:
                    excluded_count += 1
                    logger.debug(
                        f"Excluded {spot_data.region} ({spot_data.instance_type}): "
                        f"interruption rate {spot_data.interruption_rate * 100:.2f}% > {max_rate * 100:.1f}%"
                    )
            
            # Update statistics
            self._last_filter_stats.update({
                "input_count": len(data),
                "filtered_count": len(filtered_data),
                "excluded_by_interruption_rate": excluded_count,
                "max_interruption_rate": max_rate
            })
            
            logger.info(
                f"Interruption rate filtering complete: {len(filtered_data)} regions passed, "
                f"{excluded_count} excluded"
            )
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error during interruption rate filtering: {e}")
            raise DataFilterServiceError(f"Interruption rate filtering failed: {e}")

    def validate_price_data(self, data: List[RawSpotData]) -> List[RawSpotData]:
        """
        Validate spot data for completeness and data quality.
        
        Args:
            data: List of RawSpotData objects to validate
            
        Returns:
            List of valid RawSpotData objects
            
        Raises:
            DataFilterServiceError: If validation fails
        """
        if not data:
            logger.warning("No data provided for price validation")
            return []
        
        try:
            logger.info(f"Validating {len(data)} records for data completeness")
            
            valid_data = []
            validation_failures = {
                "missing_price": 0,
                "invalid_price": 0,
                "missing_region": 0,
                "missing_instance_type": 0,
                "invalid_currency": 0,
                "missing_timestamp": 0,
                "stale_data": 0
            }
            
            for spot_data in data:
                validation_result = self._validate_single_record(spot_data)
                
                if validation_result["is_valid"]:
                    valid_data.append(spot_data)
                else:
                    # Track validation failure reasons
                    for failure_type in validation_result["failures"]:
                        if failure_type in validation_failures:
                            validation_failures[failure_type] += 1
                    
                    logger.debug(
                        f"Invalid data for {spot_data.region} ({spot_data.instance_type}): "
                        f"{', '.join(validation_result['failures'])}"
                    )
            
            # Update statistics
            total_excluded = sum(validation_failures.values())
            self._last_filter_stats.update({
                "validation_input_count": len(data),
                "validation_valid_count": len(valid_data),
                "validation_excluded_count": total_excluded,
                "validation_failures": validation_failures
            })
            
            logger.info(
                f"Price data validation complete: {len(valid_data)} valid records, "
                f"{total_excluded} excluded"
            )
            
            if validation_failures:
                failure_summary = [
                    f"{count} {failure_type.replace('_', ' ')}"
                    for failure_type, count in validation_failures.items()
                    if count > 0
                ]
                logger.warning(f"Validation failures: {', '.join(failure_summary)}")
            
            return valid_data
            
        except Exception as e:
            logger.error(f"Error during price data validation: {e}")
            raise DataFilterServiceError(f"Price data validation failed: {e}")

    def filter_and_validate(
        self,
        data: List[RawSpotData],
        max_interruption_rate: Optional[float] = None
    ) -> List[RawSpotData]:
        """
        Apply both interruption rate filtering and data validation.
        
        Args:
            data: List of RawSpotData objects to process
            max_interruption_rate: Maximum acceptable interruption rate (optional)
            
        Returns:
            List of filtered and validated RawSpotData objects
            
        Raises:
            DataFilterServiceError: If filtering or validation fails
        """
        try:
            logger.info(f"Starting combined filtering and validation for {len(data)} records")
            
            # First validate data quality
            validated_data = self.validate_price_data(data)
            
            # Then filter by interruption rate
            filtered_data = self.filter_by_interruption_rate(validated_data, max_interruption_rate)
            
            logger.info(
                f"Combined filtering complete: {len(data)} → {len(validated_data)} → {len(filtered_data)} records"
            )
            
            return filtered_data
            
        except DataFilterServiceError:
            raise
        except Exception as e:
            logger.error(f"Error during combined filtering and validation: {e}")
            raise DataFilterServiceError(f"Combined filtering failed: {e}")

    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the last filtering operation.
        
        Returns:
            Dictionary containing filtering statistics
        """
        return self._last_filter_stats.copy()

    def set_max_interruption_rate(self, max_rate: float) -> None:
        """
        Update the maximum interruption rate threshold.
        
        Args:
            max_rate: New maximum interruption rate (0.0 to 1.0)
            
        Raises:
            DataFilterServiceError: If rate is invalid
        """
        if not isinstance(max_rate, (int, float)) or not (0 <= max_rate <= 1):
            raise DataFilterServiceError("Max interruption rate must be between 0.0 and 1.0")
        
        old_rate = self.max_interruption_rate
        self.max_interruption_rate = max_rate
        
        logger.info(f"Updated max interruption rate: {old_rate * 100:.1f}% → {max_rate * 100:.1f}%")

    def _validate_interruption_rate(self, data: RawSpotData, max_rate: float) -> bool:
        """
        Validate a single record's interruption rate.
        
        Args:
            data: RawSpotData object to validate
            max_rate: Maximum acceptable interruption rate
            
        Returns:
            True if interruption rate is acceptable, False otherwise
        """
        try:
            # Check if interruption rate is within acceptable range
            if not isinstance(data.interruption_rate, (int, float)):
                return False
            
            # Check if rate is below threshold
            return data.interruption_rate < max_rate
            
        except Exception as e:
            logger.warning(f"Error validating interruption rate for {data.region}: {e}")
            return False

    def _validate_single_record(self, data: RawSpotData) -> Dict[str, Any]:
        """
        Validate a single spot data record for completeness.
        
        Args:
            data: RawSpotData object to validate
            
        Returns:
            Dictionary with validation results
        """
        failures = []
        
        try:
            # Check required fields
            if not data.region or not isinstance(data.region, str):
                failures.append("missing_region")
            
            if not data.instance_type or not isinstance(data.instance_type, str):
                failures.append("missing_instance_type")
            
            # Check price data
            if data.spot_price is None:
                failures.append("missing_price")
            elif not isinstance(data.spot_price, (int, float)) or data.spot_price <= 0:
                failures.append("invalid_price")
            
            # Check currency
            if data.currency != "USD":
                failures.append("invalid_currency")
            
            # Check timestamp
            if not data.timestamp or not isinstance(data.timestamp, datetime):
                failures.append("missing_timestamp")
            else:
                # Check if data is too old (more than 24 hours)
                current_time = datetime.now(timezone.utc)
                if data.timestamp.tzinfo is None:
                    data_timestamp = data.timestamp.replace(tzinfo=timezone.utc)
                else:
                    data_timestamp = data.timestamp
                
                age_hours = (current_time - data_timestamp).total_seconds() / 3600
                if age_hours > 24:
                    failures.append("stale_data")
            
            return {
                "is_valid": len(failures) == 0,
                "failures": failures
            }
            
        except Exception as e:
            logger.warning(f"Error validating record for {getattr(data, 'region', 'unknown')}: {e}")
            return {
                "is_valid": False,
                "failures": ["validation_error"]
            }

    def get_supported_instance_types(self) -> List[str]:
        """
        Get list of supported instance types for filtering.
        
        Returns:
            List of supported instance type strings
        """
        return ["p5en.48xlarge", "p5.48xlarge"]

    def filter_by_instance_type(
        self,
        data: List[RawSpotData],
        instance_types: List[str]
    ) -> List[RawSpotData]:
        """
        Filter data by specific instance types.
        
        Args:
            data: List of RawSpotData objects to filter
            instance_types: List of instance types to include
            
        Returns:
            List of RawSpotData objects matching the instance types
        """
        if not data:
            return []
        
        if not instance_types:
            logger.warning("No instance types specified for filtering")
            return data
        
        try:
            logger.info(f"Filtering {len(data)} records by instance types: {instance_types}")
            
            filtered_data = [
                spot_data for spot_data in data
                if spot_data.instance_type in instance_types
            ]
            
            excluded_count = len(data) - len(filtered_data)
            
            logger.info(
                f"Instance type filtering complete: {len(filtered_data)} records matched, "
                f"{excluded_count} excluded"
            )
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error during instance type filtering: {e}")
            raise DataFilterServiceError(f"Instance type filtering failed: {e}")

    def filter_by_availability(self, data: List[RawSpotData]) -> List[RawSpotData]:
        """
        Filter data to include only available instances.
        
        Args:
            data: List of RawSpotData objects to filter
            
        Returns:
            List of RawSpotData objects where availability is True
        """
        if not data:
            return []
        
        try:
            logger.info(f"Filtering {len(data)} records by availability")
            
            available_data = [
                spot_data for spot_data in data
                if spot_data.availability
            ]
            
            excluded_count = len(data) - len(available_data)
            
            logger.info(
                f"Availability filtering complete: {len(available_data)} available, "
                f"{excluded_count} unavailable"
            )
            
            return available_data
            
        except Exception as e:
            logger.error(f"Error during availability filtering: {e}")
            raise DataFilterServiceError(f"Availability filtering failed: {e}")

    def clear_statistics(self) -> None:
        """Clear filtering statistics."""
        self._last_filter_stats.clear()
        logger.debug("Filter statistics cleared")