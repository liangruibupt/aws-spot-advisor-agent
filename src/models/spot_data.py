"""
Data models for AWS Spot Price Analyzer.

This module contains dataclasses and validation functions for handling
spot pricing data throughout the analysis pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator, ValidationError


@dataclass
class RawSpotData:
    """
    Raw spot pricing data scraped from AWS EC2 Spot Instance Advisor.
    
    Attributes:
        region: AWS region identifier (e.g., 'us-east-1')
        instance_type: EC2 instance type (e.g., 'p5en.48xlarge')
        spot_price: Current spot price in USD
        currency: Currency denomination (should be 'USD')
        interruption_rate: Interruption rate as a decimal (0.05 = 5%)
        timestamp: When this data was collected
        availability: Whether the instance type is available in this region
    """
    region: str
    instance_type: str
    spot_price: float
    currency: str
    interruption_rate: float
    timestamp: datetime
    availability: bool = True

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate the raw spot data fields.
        
        Raises:
            ValueError: If any field contains invalid data
        """
        if not self.region or not isinstance(self.region, str):
            raise ValueError("Region must be a non-empty string")
        
        if not self.instance_type or not isinstance(self.instance_type, str):
            raise ValueError("Instance type must be a non-empty string")
        
        if not isinstance(self.spot_price, (int, float)) or self.spot_price < 0:
            raise ValueError("Spot price must be a non-negative number")
        
        if self.currency != "USD":
            raise ValueError("Currency must be 'USD'")
        
        if not isinstance(self.interruption_rate, (int, float)) or not (0 <= self.interruption_rate <= 1):
            raise ValueError("Interruption rate must be between 0 and 1")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object")
        
        if not isinstance(self.availability, bool):
            raise ValueError("Availability must be a boolean")

    @property
    def interruption_rate_percentage(self) -> float:
        """Return interruption rate as a percentage."""
        return self.interruption_rate * 100

    def is_low_interruption(self, threshold: float = 0.05) -> bool:
        """
        Check if interruption rate is below threshold.
        
        Args:
            threshold: Maximum acceptable interruption rate (default 5%)
            
        Returns:
            True if interruption rate is below threshold
        """
        return self.interruption_rate < threshold


@dataclass
class SpotPriceResult:
    """
    Processed spot pricing result for a specific region.
    
    Attributes:
        region: AWS region identifier
        instance_type: EC2 instance type
        spot_price: Current spot price in USD
        currency: Currency denomination
        interruption_rate: Interruption rate as a decimal
        rank: Ranking position (1 = best price)
        data_timestamp: When the underlying data was collected
    """
    region: str
    instance_type: str
    spot_price: float
    currency: str
    interruption_rate: float
    rank: int
    data_timestamp: datetime

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate the spot price result fields.
        
        Raises:
            ValueError: If any field contains invalid data
        """
        if not self.region or not isinstance(self.region, str):
            raise ValueError("Region must be a non-empty string")
        
        if not self.instance_type or not isinstance(self.instance_type, str):
            raise ValueError("Instance type must be a non-empty string")
        
        if not isinstance(self.spot_price, (int, float)) or self.spot_price < 0:
            raise ValueError("Spot price must be a non-negative number")
        
        if self.currency != "USD":
            raise ValueError("Currency must be 'USD'")
        
        if not isinstance(self.interruption_rate, (int, float)) or not (0 <= self.interruption_rate <= 1):
            raise ValueError("Interruption rate must be between 0 and 1")
        
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("Rank must be a positive integer")
        
        if not isinstance(self.data_timestamp, datetime):
            raise ValueError("Data timestamp must be a datetime object")

    @property
    def interruption_rate_percentage(self) -> str:
        """Return formatted interruption rate as percentage string."""
        return f"{self.interruption_rate * 100:.2f}%"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "region": self.region,
            "instance_type": self.instance_type,
            "spot_price": self.spot_price,
            "currency": self.currency,
            "interruption_rate": self.interruption_rate_percentage,
            "rank": self.rank,
            "data_timestamp": self.data_timestamp.isoformat()
        }


@dataclass
class AnalysisResponse:
    """
    Complete analysis response containing results and metadata.
    
    Attributes:
        results: List of top spot price results
        total_regions_analyzed: Total number of regions processed
        filtered_regions_count: Number of regions that met criteria
        data_collection_timestamp: When the analysis was performed
        warnings: Optional list of warning messages
    """
    results: List[SpotPriceResult]
    total_regions_analyzed: int
    filtered_regions_count: int
    data_collection_timestamp: datetime
    warnings: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate the analysis response fields.
        
        Raises:
            ValueError: If any field contains invalid data
        """
        if not isinstance(self.results, list):
            raise ValueError("Results must be a list")
        
        for result in self.results:
            if not isinstance(result, SpotPriceResult):
                raise ValueError("All results must be SpotPriceResult instances")
        
        if not isinstance(self.total_regions_analyzed, int) or self.total_regions_analyzed < 0:
            raise ValueError("Total regions analyzed must be a non-negative integer")
        
        if not isinstance(self.filtered_regions_count, int) or self.filtered_regions_count < 0:
            raise ValueError("Filtered regions count must be a non-negative integer")
        
        if self.filtered_regions_count > self.total_regions_analyzed:
            raise ValueError("Filtered regions count cannot exceed total regions analyzed")
        
        if not isinstance(self.data_collection_timestamp, datetime):
            raise ValueError("Data collection timestamp must be a datetime object")
        
        if self.warnings is not None and not isinstance(self.warnings, list):
            raise ValueError("Warnings must be a list or None")
        
        if self.warnings:
            for warning in self.warnings:
                if not isinstance(warning, str):
                    raise ValueError("All warnings must be strings")

    def add_warning(self, message: str) -> None:
        """
        Add a warning message to the response.
        
        Args:
            message: Warning message to add
        """
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(message)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [result.to_dict() for result in self.results],
            "total_regions_analyzed": self.total_regions_analyzed,
            "filtered_regions_count": self.filtered_regions_count,
            "data_collection_timestamp": self.data_collection_timestamp.isoformat(),
            "warnings": self.warnings or []
        }

    @property
    def has_warnings(self) -> bool:
        """Check if response contains any warnings."""
        return bool(self.warnings)

    @property
    def result_count(self) -> int:
        """Get the number of results returned."""
        return len(self.results)


# Pydantic models for additional validation if needed
class RawSpotDataValidator(BaseModel):
    """Pydantic validator for RawSpotData."""
    region: str = Field(..., min_length=1, description="AWS region identifier")
    instance_type: str = Field(..., min_length=1, description="EC2 instance type")
    spot_price: float = Field(..., ge=0, description="Spot price in USD")
    currency: str = Field(..., pattern="^USD$", description="Currency code")
    interruption_rate: float = Field(..., ge=0, le=1, description="Interruption rate as decimal")
    timestamp: datetime = Field(..., description="Data collection timestamp")
    availability: bool = Field(True, description="Instance availability")

    @field_validator('region')
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate AWS region format."""
        if not v or len(v) < 3:
            raise ValueError('Region must be a valid AWS region identifier')
        return v

    @field_validator('instance_type')
    @classmethod
    def validate_instance_type(cls, v: str) -> str:
        """Validate instance type format."""
        if not v or '.' not in v:
            raise ValueError('Instance type must be in format like p5en.48xlarge')
        return v


def validate_raw_spot_data(data: dict) -> RawSpotData:
    """
    Validate and create RawSpotData from dictionary.
    
    Args:
        data: Dictionary containing spot data fields
        
    Returns:
        Validated RawSpotData instance
        
    Raises:
        ValidationError: If data is invalid
    """
    try:
        validated = RawSpotDataValidator(**data)
        return RawSpotData(
            region=validated.region,
            instance_type=validated.instance_type,
            spot_price=validated.spot_price,
            currency=validated.currency,
            interruption_rate=validated.interruption_rate,
            timestamp=validated.timestamp,
            availability=validated.availability
        )
    except ValidationError as e:
        raise ValueError(f"Invalid spot data: {e}")


def validate_spot_price_results(results: List[dict]) -> List[SpotPriceResult]:
    """
    Validate and create list of SpotPriceResult from dictionaries.
    
    Args:
        results: List of dictionaries containing result data
        
    Returns:
        List of validated SpotPriceResult instances
        
    Raises:
        ValidationError: If any result data is invalid
    """
    validated_results = []
    for i, result_data in enumerate(results):
        try:
            result = SpotPriceResult(**result_data)
            validated_results.append(result)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid result data at index {i}: {e}")
    
    return validated_results