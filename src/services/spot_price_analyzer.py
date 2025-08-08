"""
Main orchestration service for AWS Spot Price Analyzer.

This module provides the main SpotPriceAnalyzer class that coordinates
all services to perform complete spot price analysis workflow.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from src.models.spot_data import RawSpotData, SpotPriceResult, AnalysisResponse
from src.services.aws_spot_price_service import AwsSpotPriceService, AwsSpotPriceServiceError
from src.services.data_filter_service import DataFilterService, DataFilterServiceError
from src.services.ranking_engine import RankingEngine
# Keep old imports for backward compatibility
from src.services.web_scraper_service import WebScraperService, WebScraperServiceError
from src.services.bedrock_agent_service import BedrockAgentService, BedrockAgentServiceError
from src.services.result_formatter import ResultFormatter
from src.utils.exceptions import (
    SpotAnalyzerBaseError,
    InsufficientDataError,
    ConfigurationError,
    RankingError,
    FormattingError,
    RetryableError
)
from src.utils.error_response import ErrorResponseFormatter
from src.utils.retry_utils import data_processing_retry
from src.utils.cache import ttl_cache, analysis_cache, cache_warmer


logger = logging.getLogger(__name__)


# Legacy exceptions for backward compatibility
class SpotPriceAnalyzerError(SpotAnalyzerBaseError):
    """Base exception for SpotPriceAnalyzer errors."""
    pass


class InsufficientRegionsError(InsufficientDataError):
    """Raised when insufficient regions meet the criteria."""
    pass


class ServiceFailureError(SpotAnalyzerBaseError):
    """Raised when a dependent service fails."""
    pass


class SpotPriceAnalyzer:
    """
    Main orchestration service for AWS spot price analysis.
    
    This service coordinates web scraping, data filtering, and ranking
    to provide the top 3 most cost-effective AWS regions for spot instances.
    """

    # Default instance types to analyze
    DEFAULT_INSTANCE_TYPES = ["p5en.48xlarge", "p5.48xlarge"]
    
    # Default maximum interruption rate (5%)
    DEFAULT_MAX_INTERRUPTION_RATE = 0.05
    
    # Minimum number of regions required for analysis
    MIN_REGIONS_REQUIRED = 1

    def __init__(
        self,
        aws_spot_service: Optional[AwsSpotPriceService] = None,
        data_filter: Optional[DataFilterService] = None,
        ranking_engine: Optional[RankingEngine] = None,
        result_formatter: Optional[ResultFormatter] = None,
        # Keep old parameters for backward compatibility
        web_scraper: Optional[WebScraperService] = None,
        bedrock_service: Optional[BedrockAgentService] = None
    ):
        """
        Initialize the Spot Price Analyzer.
        
        Args:
            aws_spot_service: AwsSpotPriceService instance (optional, will create if None)
            data_filter: DataFilterService instance (optional, will create if None)
            ranking_engine: RankingEngine instance (optional, will create if None)
            result_formatter: ResultFormatter instance (optional, will create if None)
            web_scraper: WebScraperService instance (deprecated, for backward compatibility)
            bedrock_service: BedrockAgentService instance (deprecated, for backward compatibility)
        """
        # Initialize services with dependency injection support
        self.aws_spot_service = aws_spot_service or AwsSpotPriceService()
        self.data_filter = data_filter or DataFilterService()
        self.ranking_engine = ranking_engine or RankingEngine()
        self.result_formatter = result_formatter or ResultFormatter()
        
        # Keep old services for backward compatibility (but prefer new AWS API service)
        if web_scraper is not None or bedrock_service is not None:
            logger.warning("web_scraper and bedrock_service parameters are deprecated. Using AWS API service instead.")
        self.bedrock_service = bedrock_service or BedrockAgentService()
        self.web_scraper = web_scraper or WebScraperService(self.bedrock_service)
        
        # Initialize error response formatter
        self.error_formatter = ErrorResponseFormatter()
        
        # Analysis configuration
        self.max_interruption_rate = self.DEFAULT_MAX_INTERRUPTION_RATE
        self.instance_types = self.DEFAULT_INSTANCE_TYPES.copy()
        
        logger.info("SpotPriceAnalyzer initialized")

    @data_processing_retry(max_attempts=2)
    def analyze_spot_prices(
        self,
        instance_types: Optional[List[str]] = None,
        max_interruption_rate: Optional[float] = None,
        top_count: int = 3,
        force_refresh: bool = False
    ) -> AnalysisResponse:
        """
        Perform complete spot price analysis workflow.
        
        Args:
            instance_types: List of instance types to analyze (optional)
            max_interruption_rate: Maximum acceptable interruption rate (optional)
            top_count: Number of top regions to return (default: 3)
            force_refresh: Force refresh of cached data (default: False)
            
        Returns:
            AnalysisResponse containing top regions and analysis metadata
            
        Raises:
            SpotPriceAnalyzerError: If analysis fails
            InsufficientRegionsError: If insufficient regions meet criteria
            ServiceFailureError: If a dependent service fails
        """
        analysis_start_time = datetime.now(timezone.utc)
        
        # Use default instance types if none provided
        if instance_types is None:
            instance_types = self.instance_types.copy()
        
        # Use default interruption rate if none provided
        if max_interruption_rate is None:
            max_interruption_rate = self.max_interruption_rate
        
        logger.info(
            f"Starting spot price analysis for {instance_types} "
            f"with max interruption rate {max_interruption_rate * 100:.1f}%"
        )
        
        try:
            # Step 1: Get spot pricing data from AWS API
            raw_data = self._get_spot_data_from_aws(instance_types)
            
            # Step 2: Filter and validate data
            filtered_data = self._filter_data(raw_data, max_interruption_rate)
            
            # Step 3: Check if we have sufficient regions
            self._validate_sufficient_regions(filtered_data, top_count)
            
            # Step 4: Rank and get top regions
            top_results = self._rank_and_select_top(filtered_data, top_count)
            
            # Step 5: Create analysis response
            response = self._create_analysis_response(
                results=top_results,
                total_analyzed=len(raw_data),
                filtered_count=len(filtered_data),
                analysis_time=analysis_start_time
            )
            
            logger.info(
                f"Analysis complete: {len(top_results)} regions returned "
                f"from {len(raw_data)} analyzed"
            )
            
            return response
            
        except AwsSpotPriceServiceError as e:
            logger.error(f"AWS Spot Price Service failure during analysis: {e}")
            raise ServiceFailureError(
                message=f"AWS Spot Price Service failed: {e}",
                details={
                    "service": "aws_spot_price_service",
                    "instance_types": instance_types
                },
                original_error=e
            )
        
        except (WebScraperServiceError, BedrockAgentServiceError) as e:
            logger.error(f"Legacy service failure during analysis: {e}")
            raise ServiceFailureError(
                message=f"Legacy web scraping service failed: {e}",
                details={
                    "service": "web_scraper",
                    "instance_types": instance_types,
                    "force_refresh": force_refresh
                },
                original_error=e
            )
        
        except DataFilterServiceError as e:
            logger.error(f"Data filtering failure during analysis: {e}")
            raise ServiceFailureError(
                message=f"Data filtering service failed: {e}",
                details={
                    "service": "data_filter",
                    "max_interruption_rate": max_interruption_rate
                },
                original_error=e
            )
        
        except InsufficientDataError:
            # Re-raise insufficient data error as-is
            raise
        
        except ServiceFailureError:
            # Re-raise service failure errors as-is
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            raise SpotPriceAnalyzerError(
                message=f"Analysis failed: {e}",
                details={
                    "instance_types": instance_types,
                    "max_interruption_rate": max_interruption_rate,
                    "top_count": top_count,
                    "force_refresh": force_refresh
                },
                original_error=e
            )

    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get current analysis configuration and status.
        
        Returns:
            Dictionary containing analysis configuration and status
        """
        filter_stats = self.data_filter.get_filter_statistics()
        aws_service_info = self.aws_spot_service.get_service_info()
        
        return {
            "configuration": {
                "instance_types": self.instance_types,
                "max_interruption_rate": self.max_interruption_rate,
                "max_interruption_rate_percentage": f"{self.max_interruption_rate * 100:.1f}%"
            },
            "data_source": aws_service_info,
            "last_filter_statistics": filter_stats,
            "services_initialized": {
                "aws_spot_service": self.aws_spot_service is not None,
                "data_filter": self.data_filter is not None,
                "ranking_engine": self.ranking_engine is not None,
                # Legacy services (deprecated)
                "web_scraper": self.web_scraper is not None,
                "bedrock_service": self.bedrock_service is not None
            }
        }

    def update_configuration(
        self,
        instance_types: Optional[List[str]] = None,
        max_interruption_rate: Optional[float] = None
    ) -> None:
        """
        Update analysis configuration.
        
        Args:
            instance_types: New list of instance types (optional)
            max_interruption_rate: New maximum interruption rate (optional)
            
        Raises:
            SpotPriceAnalyzerError: If configuration is invalid
        """
        try:
            if instance_types is not None:
                if not isinstance(instance_types, list) or not instance_types:
                    raise ConfigurationError(
                        message="Instance types must be a non-empty list",
                        config_key="instance_types",
                        config_value=instance_types,
                        expected_type="list[str]"
                    )
                
                # Validate instance types are supported
                supported_types = self.aws_spot_service.get_supported_instance_types()
                invalid_types = [t for t in instance_types if t not in supported_types]
                if invalid_types:
                    # Check if they exist in AWS (more permissive than supported list)
                    existence_check = self.aws_spot_service.check_instance_types_exist(invalid_types)
                    truly_invalid = [t for t, exists in existence_check.items() if not exists]
                    
                    if truly_invalid:
                        raise ConfigurationError(
                            message=f"Instance types do not exist in AWS: {truly_invalid}",
                            config_key="instance_types",
                            config_value=truly_invalid,
                            expected_type="valid AWS instance types"
                        )
                
                self.instance_types = instance_types.copy()
                logger.info(f"Updated instance types: {self.instance_types}")
            
            if max_interruption_rate is not None:
                if not isinstance(max_interruption_rate, (int, float)) or not (0 <= max_interruption_rate <= 1):
                    raise ConfigurationError(
                        message="Max interruption rate must be between 0.0 and 1.0",
                        config_key="max_interruption_rate",
                        config_value=max_interruption_rate,
                        expected_type="float (0.0-1.0)"
                    )
                
                self.max_interruption_rate = max_interruption_rate
                self.data_filter.set_max_interruption_rate(max_interruption_rate)
                logger.info(f"Updated max interruption rate: {max_interruption_rate * 100:.1f}%")
            
        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            raise ConfigurationError(
                message=f"Configuration update failed: {e}",
                original_error=e
            )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        # Clear filter statistics (AWS service doesn't have cache to clear)
        self.data_filter.clear_statistics()
        # Clear legacy cache if using old services
        if hasattr(self, 'web_scraper') and self.web_scraper:
            self.web_scraper.clear_cache()
        logger.info("All caches cleared")

    def _get_spot_data_from_aws(self, instance_types: List[str]) -> List[RawSpotData]:
        """
        Get spot pricing data using the AWS API service.
        
        Args:
            instance_types: List of instance types to get data for
            
        Returns:
            List of RawSpotData objects
            
        Raises:
            ServiceFailureError: If AWS API call fails
        """
        try:
            logger.info(f"Getting spot data from AWS API for {len(instance_types)} instance types")
            
            raw_data = self.aws_spot_service.get_spot_prices(instance_types=instance_types)
            
            if not raw_data:
                raise ServiceFailureError(
                    message="No spot data retrieved from AWS API",
                    details={
                        "instance_types": instance_types,
                        "service": "aws_spot_price_service"
                    }
                )
            
            logger.info(f"Successfully retrieved {len(raw_data)} spot price records from AWS API")
            return raw_data
            
        except AwsSpotPriceServiceError as e:
            logger.error(f"AWS API call failed: {e}")
            raise ServiceFailureError(
                message=f"Failed to get spot data from AWS API: {e}",
                details={
                    "service": "aws_spot_price_service",
                    "instance_types": instance_types
                },
                original_error=e
            )

    def _scrape_spot_data(self, instance_types: List[str], force_refresh: bool) -> List[RawSpotData]:
        """
        Scrape spot pricing data using the web scraper service.
        
        Args:
            instance_types: List of instance types to scrape
            force_refresh: Whether to force refresh cached data
            
        Returns:
            List of RawSpotData objects
            
        Raises:
            ServiceFailureError: If scraping fails
        """
        try:
            logger.info(f"Scraping spot data for {len(instance_types)} instance types")
            
            raw_data = self.web_scraper.scrape_spot_data(
                instance_types=instance_types,
                force_refresh=force_refresh
            )
            
            if not raw_data:
                raise ServiceFailureError(
                    message="No spot data retrieved from web scraping",
                    details={
                        "instance_types": instance_types,
                        "force_refresh": force_refresh
                    }
                )
            
            logger.info(f"Successfully scraped {len(raw_data)} spot price records")
            return raw_data
            
        except WebScraperServiceError as e:
            logger.error(f"Web scraping failed: {e}")
            raise ServiceFailureError(
                message=f"Failed to scrape spot data: {e}",
                details={
                    "service": "web_scraper",
                    "instance_types": instance_types,
                    "force_refresh": force_refresh
                },
                original_error=e
            )

    def _filter_data(self, raw_data: List[RawSpotData], max_interruption_rate: float) -> List[RawSpotData]:
        """
        Filter and validate spot pricing data.
        
        Args:
            raw_data: List of raw spot data to filter
            max_interruption_rate: Maximum acceptable interruption rate
            
        Returns:
            List of filtered RawSpotData objects
            
        Raises:
            ServiceFailureError: If filtering fails
        """
        try:
            logger.info(f"Filtering {len(raw_data)} records")
            
            # Apply combined filtering and validation
            filtered_data = self.data_filter.filter_and_validate(
                data=raw_data,
                max_interruption_rate=max_interruption_rate
            )
            
            logger.info(f"Filtering complete: {len(filtered_data)} records passed")
            return filtered_data
            
        except DataFilterServiceError as e:
            logger.error(f"Data filtering failed: {e}")
            raise ServiceFailureError(
                message=f"Failed to filter data: {e}",
                details={
                    "service": "data_filter",
                    "input_count": len(raw_data),
                    "max_interruption_rate": max_interruption_rate
                },
                original_error=e
            )

    def _validate_sufficient_regions(self, filtered_data: List[RawSpotData], required_count: int) -> None:
        """
        Validate that we have sufficient regions for analysis.
        
        Args:
            filtered_data: List of filtered spot data
            required_count: Number of regions required
            
        Raises:
            InsufficientRegionsError: If insufficient regions available
        """
        available_count = len(filtered_data)
        
        if available_count < self.MIN_REGIONS_REQUIRED:
            error_msg = (
                f"No regions meet the criteria (interruption rate < "
                f"{self.max_interruption_rate * 100:.1f}%)"
            )
            logger.error(error_msg)
            raise InsufficientDataError(
                message=error_msg,
                required_count=self.MIN_REGIONS_REQUIRED,
                available_count=available_count,
                criteria={"max_interruption_rate": self.max_interruption_rate}
            )
        
        if available_count < required_count:
            warning_msg = (
                f"Only {available_count} regions available, requested {required_count}. "
                f"Returning all available regions."
            )
            logger.warning(warning_msg)
            # Don't raise error, just log warning - we'll return what we have

    def _rank_and_select_top(self, filtered_data: List[RawSpotData], top_count: int) -> List[SpotPriceResult]:
        """
        Rank filtered data and select top regions.
        
        Args:
            filtered_data: List of filtered spot data
            top_count: Number of top regions to select
            
        Returns:
            List of SpotPriceResult objects
        """
        logger.info(f"Ranking {len(filtered_data)} regions and selecting top {top_count}")
        
        # Use the ranking engine to rank and get top results
        top_results = self.ranking_engine.rank_and_get_top(filtered_data, top_count)
        
        logger.info(f"Selected {len(top_results)} top regions")
        return top_results

    def _create_analysis_response(
        self,
        results: List[SpotPriceResult],
        total_analyzed: int,
        filtered_count: int,
        analysis_time: datetime
    ) -> AnalysisResponse:
        """
        Create the final analysis response.
        
        Args:
            results: List of top spot price results
            total_analyzed: Total number of regions analyzed
            filtered_count: Number of regions that passed filtering
            analysis_time: When the analysis was performed
            
        Returns:
            AnalysisResponse object
        """
        response = AnalysisResponse(
            results=results,
            total_regions_analyzed=total_analyzed,
            filtered_regions_count=filtered_count,
            data_collection_timestamp=analysis_time
        )
        
        # Add warnings if applicable
        if filtered_count < 3:
            response.add_warning(
                f"Only {filtered_count} regions met the criteria "
                f"(interruption rate < {self.max_interruption_rate * 100:.1f}%)"
            )
        
        if len(results) < 3:
            response.add_warning(
                f"Returning {len(results)} regions instead of requested 3"
            )
        
        # Add filter statistics as warnings if there were significant exclusions
        filter_stats = self.data_filter.get_filter_statistics()
        if filter_stats.get("validation_excluded_count", 0) > 0:
            excluded_count = filter_stats["validation_excluded_count"]
            response.add_warning(
                f"{excluded_count} regions excluded due to data quality issues"
            )
        
        return response

    def get_supported_instance_types(self) -> List[str]:
        """
        Get list of supported instance types.
        
        Returns:
            List of supported instance type strings
        """
        return self.aws_spot_service.get_supported_instance_types()

    def validate_instance_types(self, instance_types: List[str]) -> Dict[str, List[str]]:
        """
        Validate a list of instance types.
        
        Args:
            instance_types: List of instance types to validate
            
        Returns:
            Dictionary with 'valid' and 'invalid' lists
        """
        supported_types = self.get_supported_instance_types()
        
        valid_types = [t for t in instance_types if t in supported_types]
        invalid_types = [t for t in instance_types if t not in supported_types]
        
        return {
            "valid": valid_types,
            "invalid": invalid_types
        }

    def analyze_spot_prices_cached(
        self,
        instance_types: Optional[List[str]] = None,
        max_interruption_rate: Optional[float] = None,
        top_count: int = 3,
        force_refresh: bool = False,
        cache_ttl_seconds: Optional[float] = None
    ) -> AnalysisResponse:
        """
        Perform cached spot price analysis workflow.
        
        This method provides caching for analysis results to improve performance
        for repeated requests with the same parameters.
        
        Args:
            instance_types: List of instance types to analyze (optional)
            max_interruption_rate: Maximum acceptable interruption rate (optional)
            top_count: Number of top regions to return (default: 3)
            force_refresh: Force refresh of cached data (default: False)
            cache_ttl_seconds: Cache TTL override (uses default if None)
            
        Returns:
            AnalysisResponse containing top regions and analysis metadata
            
        Raises:
            SpotPriceAnalyzerError: If analysis fails
            InsufficientRegionsError: If insufficient regions meet criteria
            ServiceFailureError: If a dependent service fails
        """
        # Use defaults if not provided
        if instance_types is None:
            instance_types = self.instance_types.copy()
        if max_interruption_rate is None:
            max_interruption_rate = self.max_interruption_rate
        
        # Create cache key
        cache_key = self._create_analysis_cache_key(
            instance_types, max_interruption_rate, top_count
        )
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_result = analysis_cache.get(cache_key)
            if cached_result is not None:
                logger.info("Returning cached analysis result")
                return cached_result
        
        # Perform analysis
        result = self.analyze_spot_prices(
            instance_types=instance_types,
            max_interruption_rate=max_interruption_rate,
            top_count=top_count,
            force_refresh=force_refresh
        )
        
        # Cache the result
        ttl = cache_ttl_seconds or analysis_cache.default_ttl_seconds
        analysis_cache.set(cache_key, result, ttl_seconds=ttl)
        
        logger.debug(f"Cached analysis result with key: {cache_key}")
        return result
    
    def warm_analysis_cache(
        self,
        instance_type_combinations: Optional[List[List[str]]] = None,
        max_interruption_rates: Optional[List[float]] = None,
        top_counts: Optional[List[int]] = None,
        force: bool = False
    ) -> List[Any]:
        """
        Warm the analysis cache with common parameter combinations.
        
        Args:
            instance_type_combinations: List of instance type combinations to warm
            max_interruption_rates: List of interruption rates to warm
            top_counts: List of top counts to warm
            force: Force warming even if entries exist
            
        Returns:
            List of futures for warming tasks
        """
        # Use defaults if not provided
        if instance_type_combinations is None:
            instance_type_combinations = [
                self.instance_types.copy(),  # All types
                [self.instance_types[0]],    # First type only
                [self.instance_types[1]] if len(self.instance_types) > 1 else []  # Second type only
            ]
        
        if max_interruption_rates is None:
            max_interruption_rates = [self.max_interruption_rate, 0.03, 0.10]  # 5%, 3%, 10%
        
        if top_counts is None:
            top_counts = [3, 5, 10]
        
        warming_entries = []
        
        for instance_types in instance_type_combinations:
            if not instance_types:  # Skip empty combinations
                continue
                
            for max_rate in max_interruption_rates:
                for top_count in top_counts:
                    cache_key = self._create_analysis_cache_key(
                        instance_types, max_rate, top_count
                    )
                    
                    def value_factory(it=instance_types, mr=max_rate, tc=top_count):
                        return self.analyze_spot_prices(
                            instance_types=it,
                            max_interruption_rate=mr,
                            top_count=tc,
                            force_refresh=False
                        )
                    
                    warming_entries.append((cache_key, value_factory, None))
        
        logger.info(f"Warming analysis cache with {len(warming_entries)} combinations")
        return cache_warmer.warm_multiple(analysis_cache, warming_entries, force=force)
    
    def get_analysis_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the analysis cache state.
        
        Returns:
            Dictionary containing cache information
        """
        cache_stats = analysis_cache.get_stats()
        entries_info = analysis_cache.get_entries_info()
        
        return {
            "cache_type": "TTLCache",
            "cache_purpose": "analysis_results",
            "entries": entries_info,
            **cache_stats
        }
    
    def clear_analysis_cache(self) -> None:
        """Clear the analysis cache."""
        analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def _create_analysis_cache_key(
        self,
        instance_types: List[str],
        max_interruption_rate: float,
        top_count: int
    ) -> str:
        """
        Create cache key for analysis parameters.
        
        Args:
            instance_types: List of instance types
            max_interruption_rate: Maximum interruption rate
            top_count: Number of top results
            
        Returns:
            Cache key string
        """
        sorted_types = sorted(instance_types)
        return f"analysis:{':'.join(sorted_types)}:{max_interruption_rate}:{top_count}"

    def analyze_spot_prices_json(
        self,
        instance_types: Optional[List[str]] = None,
        max_interruption_rate: Optional[float] = None,
        top_count: int = 3,
        force_refresh: bool = False,
        include_summary: bool = False
    ) -> Dict[str, Any]:
        """
        Perform complete spot price analysis and return formatted JSON response.
        
        Args:
            instance_types: List of instance types to analyze (optional)
            max_interruption_rate: Maximum acceptable interruption rate (optional)
            top_count: Number of top regions to return (default: 3)
            force_refresh: Force refresh of cached data (default: False)
            include_summary: Include summary statistics in response (default: False)
            
        Returns:
            Dictionary ready for JSON serialization
            
        Raises:
            SpotPriceAnalyzerError: If analysis fails
            InsufficientRegionsError: If insufficient regions meet criteria
            ServiceFailureError: If a dependent service fails
        """
        try:
            # Perform the analysis
            analysis_response = self.analyze_spot_prices(
                instance_types=instance_types,
                max_interruption_rate=max_interruption_rate,
                top_count=top_count,
                force_refresh=force_refresh
            )
            
            # Format the response for JSON
            formatted_response = self.result_formatter.format_analysis_response(analysis_response)
            
            # Add summary statistics if requested
            if include_summary:
                summary_stats = self.result_formatter.format_summary_statistics(analysis_response)
                formatted_response["summary_statistics"] = summary_stats
            
            logger.info("Analysis response formatted for JSON output")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to generate JSON response: {e}")
            # Return formatted error response using error formatter
            error_response = self.error_formatter.format_error_response(
                error=e,
                details={
                    "instance_types": instance_types or self.instance_types,
                    "max_interruption_rate": max_interruption_rate or self.max_interruption_rate,
                    "top_count": top_count
                }
            )
            return error_response

    def analyze_spot_prices_json_string(
        self,
        instance_types: Optional[List[str]] = None,
        max_interruption_rate: Optional[float] = None,
        top_count: int = 3,
        force_refresh: bool = False,
        include_summary: bool = False,
        indent: Optional[int] = None
    ) -> str:
        """
        Perform complete spot price analysis and return JSON string response.
        
        Args:
            instance_types: List of instance types to analyze (optional)
            max_interruption_rate: Maximum acceptable interruption rate (optional)
            top_count: Number of top regions to return (default: 3)
            force_refresh: Force refresh of cached data (default: False)
            include_summary: Include summary statistics in response (default: False)
            indent: Number of spaces for JSON indentation (None for compact)
            
        Returns:
            JSON string representation of analysis results
            
        Raises:
            SpotPriceAnalyzerError: If analysis or formatting fails
        """
        try:
            # Get the formatted response
            formatted_response = self.analyze_spot_prices_json(
                instance_types=instance_types,
                max_interruption_rate=max_interruption_rate,
                top_count=top_count,
                force_refresh=force_refresh,
                include_summary=include_summary
            )
            
            # Convert to JSON string
            json_string = self.result_formatter.to_json_string(formatted_response, indent=indent)
            
            logger.info("Analysis response converted to JSON string")
            return json_string
            
        except Exception as e:
            logger.error(f"Failed to generate JSON string response: {e}")
            # Return formatted error as JSON string using error formatter
            error_response = self.error_formatter.format_error_response(
                error=FormattingError(
                    message=f"Failed to generate JSON response: {e}",
                    format_type="JSON",
                    original_error=e
                )
            )
            return self.error_formatter.to_json_string(error_response, indent=indent)

    def format_results_only(self, results: List[SpotPriceResult]) -> Dict[str, Any]:
        """
        Format only the spot price results without metadata.
        
        Args:
            results: List of SpotPriceResult objects to format
            
        Returns:
            Dictionary with formatted results
        """
        try:
            formatted_results = self.result_formatter.format_spot_price_results(results)
            return {"results": formatted_results}
        except Exception as e:
            logger.error(f"Failed to format results: {e}")
            error_response = self.error_formatter.format_error_response(
                error=FormattingError(
                    message=f"Failed to format results: {e}",
                    format_type="spot_price_results",
                    data_type="SpotPriceResult",
                    original_error=e
                )
            )
            return error_response