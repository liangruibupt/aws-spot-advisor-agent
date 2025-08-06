"""
Integration tests for caching functionality.

This module tests the integration of caching across different services
and validates cache warming and performance improvements.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.services.spot_price_analyzer import SpotPriceAnalyzer
from src.services.web_scraper_service import WebScraperService
from src.models.spot_data import RawSpotData, SpotPriceResult
from src.utils.cache import spot_data_cache, analysis_cache, cache_warmer
from tests.fixtures.mock_responses import TestDataFactory


class TestCachingIntegration:
    """Integration tests for caching functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear all caches before each test
        spot_data_cache.clear()
        analysis_cache.clear()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Clear all caches after each test
        spot_data_cache.clear()
        analysis_cache.clear()
    
    @patch('src.services.bedrock_agent_service.BedrockAgentService')
    def test_web_scraper_caching_integration(self, mock_bedrock_service):
        """Test web scraper caching integration."""
        # Setup mock
        mock_bedrock_instance = Mock()
        mock_bedrock_service.return_value = mock_bedrock_instance
        
        # Mock web scraping responses
        mock_bedrock_instance.execute_web_scraping.return_value = "mock_html_content"
        mock_bedrock_instance.parse_spot_data.return_value = TestDataFactory.create_raw_spot_data_list()
        
        # Create web scraper service
        scraper = WebScraperService(use_global_cache=True)
        
        # First call - should hit the service
        start_time = time.time()
        result1 = scraper.scrape_spot_data(["p5en.48xlarge"])
        first_call_time = time.time() - start_time
        
        # Verify service was called
        assert mock_bedrock_instance.execute_web_scraping.call_count == 1
        assert mock_bedrock_instance.parse_spot_data.call_count == 1
        assert len(result1) > 0
        
        # Second call - should use cache
        start_time = time.time()
        result2 = scraper.scrape_spot_data(["p5en.48xlarge"])
        second_call_time = time.time() - start_time
        
        # Verify cache was used (no additional service calls)
        assert mock_bedrock_instance.execute_web_scraping.call_count == 1
        assert mock_bedrock_instance.parse_spot_data.call_count == 1
        
        # Results should be identical
        assert len(result1) == len(result2)
        assert result1[0].region == result2[0].region
        
        # Second call should be faster (cached)
        assert second_call_time < first_call_time
        
        # Verify cache info
        cache_info = scraper.get_cache_info()
        assert cache_info['size'] == 1
        assert cache_info['hits'] == 1
        assert cache_info['misses'] == 1
        
        scraper.shutdown()
    
    @patch('src.services.bedrock_agent_service.BedrockAgentService')
    def test_analysis_caching_integration(self, mock_bedrock_service):
        """Test analysis result caching integration."""
        # Setup mock
        mock_bedrock_instance = Mock()
        mock_bedrock_service.return_value = mock_bedrock_instance
        
        # Mock responses
        mock_bedrock_instance.execute_web_scraping.return_value = "mock_html_content"
        mock_bedrock_instance.parse_spot_data.return_value = TestDataFactory.create_raw_spot_data_list()
        
        # Create analyzer
        analyzer = SpotPriceAnalyzer()
        
        # First analysis - should perform full workflow
        start_time = time.time()
        result1 = analyzer.analyze_spot_prices_cached(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        first_analysis_time = time.time() - start_time
        
        # Verify result
        assert result1 is not None
        assert len(result1.results) > 0
        
        # Second analysis with same parameters - should use cache
        start_time = time.time()
        result2 = analyzer.analyze_spot_prices_cached(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        second_analysis_time = time.time() - start_time
        
        # Results should be identical
        assert result1.results[0].region == result2.results[0].region
        assert result1.results[0].spot_price == result2.results[0].spot_price
        
        # Second analysis should be much faster (cached)
        assert second_analysis_time < first_analysis_time * 0.5
        
        # Verify cache info
        cache_info = analyzer.get_analysis_cache_info()
        assert cache_info['size'] == 1
        assert cache_info['hits'] == 1
        assert cache_info['misses'] == 1
    
    @patch('src.services.bedrock_agent_service.BedrockAgentService')
    def test_cache_warming_integration(self, mock_bedrock_service):
        """Test cache warming integration."""
        # Setup mock
        mock_bedrock_instance = Mock()
        mock_bedrock_service.return_value = mock_bedrock_instance
        
        # Mock responses
        mock_bedrock_instance.execute_web_scraping.return_value = "mock_html_content"
        mock_bedrock_instance.parse_spot_data.return_value = TestDataFactory.create_raw_spot_data_list()
        
        # Create services
        scraper = WebScraperService(use_global_cache=True)
        analyzer = SpotPriceAnalyzer()
        
        # Warm web scraper cache
        warming_future = scraper.warm_cache(["p5en.48xlarge"])
        warming_result = warming_future.result(timeout=10)
        
        assert warming_result is True
        
        # Verify cache was warmed
        cache_info = scraper.get_cache_info()
        assert cache_info['size'] == 1
        
        # Now scraping should be instant (cached)
        start_time = time.time()
        result = scraper.scrape_spot_data(["p5en.48xlarge"])
        scrape_time = time.time() - start_time
        
        # Should be very fast since it's cached
        assert scrape_time < 0.1  # Less than 100ms
        assert len(result) > 0
        
        # Warm analysis cache
        warming_futures = analyzer.warm_analysis_cache(
            instance_type_combinations=[["p5en.48xlarge"]],
            max_interruption_rates=[0.05],
            top_counts=[3]
        )
        
        # Wait for warming to complete
        for future in warming_futures:
            assert future.result(timeout=10) is True
        
        # Analysis should now be cached
        start_time = time.time()
        analysis_result = analyzer.analyze_spot_prices_cached(
            instance_types=["p5en.48xlarge"],
            max_interruption_rate=0.05,
            top_count=3
        )
        analysis_time = time.time() - start_time
        
        # Should be very fast since it's cached
        assert analysis_time < 0.1  # Less than 100ms
        assert len(analysis_result.results) > 0
        
        scraper.shutdown()
    
    @patch('src.services.bedrock_agent_service.BedrockAgentService')
    def test_cache_invalidation_integration(self, mock_bedrock_service):
        """Test cache invalidation and force refresh."""
        # Setup mock
        mock_bedrock_instance = Mock()
        mock_bedrock_service.return_value = mock_bedrock_instance
        
        # Mock responses - first call returns different data than second
        first_data = TestDataFactory.create_raw_spot_data_list()
        second_data = TestDataFactory.create_raw_spot_data_list()
        second_data[0].spot_price = 999.99  # Different price
        
        mock_bedrock_instance.execute_web_scraping.return_value = "mock_html_content"
        mock_bedrock_instance.parse_spot_data.side_effect = [first_data, second_data]
        
        # Create scraper
        scraper = WebScraperService(use_global_cache=True)
        
        # First call - caches result
        result1 = scraper.scrape_spot_data(["p5en.48xlarge"])
        assert result1[0].spot_price != 999.99
        
        # Second call - uses cache (same result)
        result2 = scraper.scrape_spot_data(["p5en.48xlarge"])
        assert result1[0].spot_price == result2[0].spot_price
        
        # Force refresh - should get new data
        result3 = scraper.scrape_spot_data(["p5en.48xlarge"], force_refresh=True)
        assert result3[0].spot_price == 999.99  # New data
        
        # Verify service was called twice (initial + force refresh)
        assert mock_bedrock_instance.execute_web_scraping.call_count == 2
        assert mock_bedrock_instance.parse_spot_data.call_count == 2
        
        scraper.shutdown()
    
    @patch('src.services.bedrock_agent_service.BedrockAgentService')
    def test_cache_expiration_integration(self, mock_bedrock_service):
        """Test cache expiration behavior."""
        # Setup mock
        mock_bedrock_instance = Mock()
        mock_bedrock_service.return_value = mock_bedrock_instance
        
        # Mock responses
        mock_bedrock_instance.execute_web_scraping.return_value = "mock_html_content"
        mock_bedrock_instance.parse_spot_data.return_value = TestDataFactory.create_raw_spot_data_list()
        
        # Create scraper with very short cache TTL
        scraper = WebScraperService(cache_ttl_seconds=0.1, use_global_cache=False)
        
        # First call - caches result
        result1 = scraper.scrape_spot_data(["p5en.48xlarge"])
        assert len(result1) > 0
        
        # Immediate second call - uses cache
        result2 = scraper.scrape_spot_data(["p5en.48xlarge"])
        assert mock_bedrock_instance.execute_web_scraping.call_count == 1
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Third call - cache expired, should fetch new data
        result3 = scraper.scrape_spot_data(["p5en.48xlarge"])
        assert mock_bedrock_instance.execute_web_scraping.call_count == 2
        
        scraper.shutdown()
    
    def test_global_cache_sharing(self):
        """Test that global caches are shared between service instances."""
        # Clear global caches
        spot_data_cache.clear()
        analysis_cache.clear()
        
        # Create two scraper instances using global cache
        scraper1 = WebScraperService(use_global_cache=True)
        scraper2 = WebScraperService(use_global_cache=True)
        
        # Set data in first scraper's cache
        test_data = TestDataFactory.create_raw_spot_data_list()
        cache_key = scraper1._get_cache_key(["p5en.48xlarge"])
        scraper1._cache.set(cache_key, test_data)
        
        # Second scraper should see the same data
        cached_data = scraper2._cache.get(cache_key)
        assert cached_data is not None
        assert len(cached_data) == len(test_data)
        assert cached_data[0].region == test_data[0].region
        
        # Cache info should be consistent
        info1 = scraper1.get_cache_info()
        info2 = scraper2.get_cache_info()
        assert info1['size'] == info2['size']
        
        scraper1.shutdown()
        scraper2.shutdown()
    
    def test_cache_statistics_tracking(self):
        """Test cache statistics tracking across operations."""
        # Clear global caches
        spot_data_cache.clear()
        analysis_cache.clear()
        
        # Create scraper
        scraper = WebScraperService(use_global_cache=True)
        
        # Initial stats
        initial_stats = scraper.get_cache_info()
        assert initial_stats['hits'] == 0
        assert initial_stats['misses'] == 0
        assert initial_stats['size'] == 0
        
        # Set some test data
        test_data = TestDataFactory.create_raw_spot_data_list()
        cache_key = scraper._get_cache_key(["p5en.48xlarge"])
        scraper._cache.set(cache_key, test_data)
        
        # Get data (should be a hit)
        cached_data = scraper._cache.get(cache_key)
        assert cached_data is not None
        
        # Check stats
        stats = scraper.get_cache_info()
        assert stats['hits'] == 1
        assert stats['misses'] == 0
        assert stats['size'] == 1
        assert stats['sets'] == 1
        
        # Try to get non-existent data (should be a miss)
        missing_data = scraper._cache.get("nonexistent_key")
        assert missing_data is None
        
        # Check updated stats
        final_stats = scraper.get_cache_info()
        assert final_stats['hits'] == 1
        assert final_stats['misses'] == 1
        assert final_stats['hit_rate'] == 0.5  # 1 hit out of 2 requests
        
        scraper.shutdown()


class TestCachePerformance:
    """Performance tests for caching functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        spot_data_cache.clear()
        analysis_cache.clear()
    
    def teardown_method(self):
        """Clean up test environment."""
        spot_data_cache.clear()
        analysis_cache.clear()
    
    def test_cache_performance_improvement(self):
        """Test that caching provides significant performance improvement."""
        # Create test data
        test_data = TestDataFactory.create_raw_spot_data_list()
        cache_key = "performance_test_key"
        
        # Simulate expensive operation
        def expensive_operation():
            time.sleep(0.1)  # Simulate 100ms operation
            return test_data
        
        # Time uncached operation
        start_time = time.time()
        result1 = expensive_operation()
        uncached_time = time.time() - start_time
        
        # Cache the result
        spot_data_cache.set(cache_key, result1)
        
        # Time cached operation
        start_time = time.time()
        result2 = spot_data_cache.get(cache_key)
        cached_time = time.time() - start_time
        
        # Cached operation should be much faster
        assert cached_time < uncached_time * 0.1  # At least 10x faster
        assert result1 == result2  # Same result
    
    def test_cache_memory_efficiency(self):
        """Test cache memory efficiency with large datasets."""
        # Create cache with size limit
        from src.utils.cache import TTLCache
        limited_cache = TTLCache(max_size=5)
        
        # Add more entries than the limit
        for i in range(10):
            key = f"key_{i}"
            value = f"value_{i}" * 1000  # Large value
            limited_cache.set(key, value)
        
        # Cache should not exceed max size
        stats = limited_cache.get_stats()
        assert stats['size'] <= 5
        assert stats['evictions'] > 0  # Some entries were evicted
    
    def test_concurrent_cache_access(self):
        """Test cache performance under concurrent access."""
        import threading
        import concurrent.futures
        
        # Create test data
        test_data = TestDataFactory.create_raw_spot_data_list()
        
        def cache_worker(worker_id):
            """Worker function for concurrent cache access."""
            results = []
            for i in range(50):
                key = f"worker_{worker_id}_key_{i}"
                
                # Set data
                spot_data_cache.set(key, test_data)
                
                # Get data
                cached_data = spot_data_cache.get(key)
                results.append(cached_data is not None)
            
            return all(results)
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # All workers should succeed
        assert all(results)
        
        # Cache should have entries from all workers
        stats = spot_data_cache.get_stats()
        assert stats['size'] > 0
        assert stats['sets'] >= 250  # 5 workers * 50 operations each