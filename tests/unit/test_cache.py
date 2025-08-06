"""
Unit tests for caching utilities.

This module tests the advanced caching functionality including TTL cache,
cache warming, and caching decorators.
"""

import pytest
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
from concurrent.futures import Future

from src.utils.cache import (
    CacheEntry,
    TTLCache,
    CacheWarmer,
    ttl_cache,
    cached_property_ttl,
    spot_data_cache,
    analysis_cache,
    cache_warmer,
    _make_cache_key
)
from src.utils.exceptions import CacheError


class TestCacheEntry:
    """Test cases for CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        timestamp = datetime.now(timezone.utc)
        entry = CacheEntry(
            value="test_value",
            timestamp=timestamp,
            ttl_seconds=3600
        )
        
        assert entry.value == "test_value"
        assert entry.timestamp == timestamp
        assert entry.ttl_seconds == 3600
        assert entry.access_count == 0
        assert entry.last_access is None
    
    def test_cache_entry_not_expired(self):
        """Test cache entry that hasn't expired."""
        entry = CacheEntry(
            value="test_value",
            timestamp=datetime.now(timezone.utc),
            ttl_seconds=3600
        )
        
        assert not entry.is_expired()
    
    def test_cache_entry_expired(self):
        """Test cache entry that has expired."""
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        entry = CacheEntry(
            value="test_value",
            timestamp=old_timestamp,
            ttl_seconds=3600  # 1 hour TTL, but entry is 2 hours old
        )
        
        assert entry.is_expired()
    
    def test_cache_entry_never_expires(self):
        """Test cache entry with zero TTL (never expires)."""
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=1)
        entry = CacheEntry(
            value="test_value",
            timestamp=old_timestamp,
            ttl_seconds=0  # Never expires
        )
        
        assert not entry.is_expired()
    
    def test_cache_entry_access(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(
            value="test_value",
            timestamp=datetime.now(timezone.utc),
            ttl_seconds=3600
        )
        
        # Access the entry
        value = entry.access()
        
        assert value == "test_value"
        assert entry.access_count == 1
        assert entry.last_access is not None
        
        # Access again
        entry.access()
        assert entry.access_count == 2
    
    def test_cache_entry_age(self):
        """Test cache entry age calculation."""
        timestamp = datetime.now(timezone.utc) - timedelta(seconds=30)
        entry = CacheEntry(
            value="test_value",
            timestamp=timestamp,
            ttl_seconds=3600
        )
        
        age = entry.get_age_seconds()
        assert 25 <= age <= 35  # Should be around 30 seconds


class TestTTLCache:
    """Test cases for TTLCache class."""
    
    def test_ttl_cache_creation(self):
        """Test TTL cache creation."""
        cache = TTLCache(default_ttl_seconds=1800, max_size=500)
        
        assert cache.default_ttl_seconds == 1800
        assert cache.max_size == 500
        assert len(cache._cache) == 0
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = TTLCache()
        
        cache.set("key1", "value1")
        result = cache.get("key1")
        
        assert result == "value1"
    
    def test_cache_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = TTLCache()
        
        result = cache.get("nonexistent")
        
        assert result is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = TTLCache()
        
        # Set with very short TTL
        cache.set("key1", "value1", ttl_seconds=0.1)
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        assert cache.get("key1") is None
    
    def test_cache_custom_ttl(self):
        """Test cache with custom TTL per entry."""
        cache = TTLCache(default_ttl_seconds=3600)
        
        cache.set("key1", "value1", ttl_seconds=1800)  # Custom TTL
        cache.set("key2", "value2")  # Default TTL
        
        # Both should be available
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # Check that custom TTL was used
        entry1 = cache._cache["key1"]
        entry2 = cache._cache["key2"]
        
        assert entry1.ttl_seconds == 1800
        assert entry2.ttl_seconds == 3600
    
    def test_cache_delete(self):
        """Test cache entry deletion."""
        cache = TTLCache()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Delete the entry
        deleted = cache.delete("key1")
        assert deleted is True
        assert cache.get("key1") is None
        
        # Try to delete non-existent key
        deleted = cache.delete("nonexistent")
        assert deleted is False
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = TTLCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert len(cache._cache) == 2
        
        cache.clear()
        
        assert len(cache._cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = TTLCache()
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['size'] == 0
        
        # Set and get some values
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_cache_max_size_eviction(self):
        """Test cache eviction when max size is reached."""
        cache = TTLCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Cache should be full
        assert len(cache._cache) == 2
        
        # Add another entry, should trigger eviction
        cache.set("key3", "value3")
        
        # Should still have max 2 entries
        assert len(cache._cache) <= 2
        
        # key3 should be present
        assert cache.get("key3") == "value3"
    
    def test_cache_expired_eviction(self):
        """Test eviction of expired entries."""
        cache = TTLCache(max_size=2)
        
        # Add entries with short TTL
        cache.set("key1", "value1", ttl_seconds=0.1)
        cache.set("key2", "value2", ttl_seconds=0.1)
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Add new entry, should trigger expired eviction
        cache.set("key3", "value3")
        
        # Expired entries should be gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_cache_entries_info(self):
        """Test getting cache entries information."""
        cache = TTLCache()
        
        cache.set("key1", "value1", ttl_seconds=1800)
        cache.get("key1")  # Access it
        
        entries_info = cache.get_entries_info()
        
        assert len(entries_info) == 1
        entry_info = entries_info[0]
        
        assert entry_info['key'] == "key1"
        assert entry_info['ttl_seconds'] == 1800
        assert entry_info['is_expired'] is False
        assert entry_info['access_count'] == 1
        assert entry_info['last_access'] is not None
    
    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = TTLCache()
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    value = f"value_{thread_id}_{i}"
                    
                    cache.set(key, value)
                    result = cache.get(key)
                    
                    if result == value:
                        results.append(True)
                    else:
                        results.append(False)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(results), "Some cache operations failed"


class TestCacheWarmer:
    """Test cases for CacheWarmer class."""
    
    def test_cache_warmer_creation(self):
        """Test cache warmer creation."""
        warmer = CacheWarmer(max_workers=2)
        
        assert warmer.max_workers == 2
        assert warmer._executor is not None
    
    def test_warm_cache_entry(self):
        """Test warming a single cache entry."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        def value_factory():
            return "warmed_value"
        
        # Warm the cache
        future = warmer.warm_cache(cache, "key1", value_factory)
        result = future.result(timeout=5)
        
        assert result is True
        assert cache.get("key1") == "warmed_value"
        
        warmer.shutdown()
    
    def test_warm_cache_skip_existing(self):
        """Test that warming skips existing entries by default."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        # Set existing value
        cache.set("key1", "existing_value")
        
        def value_factory():
            return "warmed_value"
        
        # Try to warm (should skip)
        future = warmer.warm_cache(cache, "key1", value_factory)
        result = future.result(timeout=5)
        
        assert result is False  # Skipped
        assert cache.get("key1") == "existing_value"  # Unchanged
        
        warmer.shutdown()
    
    def test_warm_cache_force(self):
        """Test forcing cache warming even with existing entry."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        # Set existing value
        cache.set("key1", "existing_value")
        
        def value_factory():
            return "warmed_value"
        
        # Force warming
        future = warmer.warm_cache(cache, "key1", value_factory, force=True)
        result = future.result(timeout=5)
        
        assert result is True
        assert cache.get("key1") == "warmed_value"  # Updated
        
        warmer.shutdown()
    
    def test_warm_multiple_entries(self):
        """Test warming multiple cache entries."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        entries = [
            ("key1", lambda: "value1", 1800),
            ("key2", lambda: "value2", 3600),
            ("key3", lambda: "value3", None)
        ]
        
        # Warm multiple entries
        futures = warmer.warm_multiple(cache, entries)
        
        # Wait for all to complete
        results = [future.result(timeout=5) for future in futures]
        
        assert all(results)
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
        warmer.shutdown()
    
    def test_warm_cache_error_handling(self):
        """Test error handling in cache warming."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        def failing_factory():
            raise ValueError("Factory failed")
        
        # Warm with failing factory
        future = warmer.warm_cache(cache, "key1", failing_factory)
        result = future.result(timeout=5)
        
        assert result is False  # Failed
        assert cache.get("key1") is None  # Not cached
        
        warmer.shutdown()
    
    def test_wait_for_warming(self):
        """Test waiting for warming completion."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        def slow_factory():
            time.sleep(0.1)
            return "slow_value"
        
        # Start warming
        warmer.warm_cache(cache, "key1", slow_factory)
        
        # Wait for completion
        warmer.wait_for_warming(timeout=5)
        
        # Should be completed
        assert cache.get("key1") == "slow_value"
        
        warmer.shutdown()


class TestTTLCacheDecorator:
    """Test cases for ttl_cache decorator."""
    
    def test_ttl_cache_decorator_basic(self):
        """Test basic ttl_cache decorator functionality."""
        call_count = 0
        
        @ttl_cache(ttl_seconds=3600)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
        
        # Call with different args
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2  # Incremented
    
    def test_ttl_cache_decorator_expiration(self):
        """Test ttl_cache decorator with expiration."""
        call_count = 0
        
        @ttl_cache(ttl_seconds=0.1)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Second call (should recompute)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 2  # Incremented
    
    def test_ttl_cache_decorator_methods(self):
        """Test ttl_cache decorator methods."""
        @ttl_cache(ttl_seconds=3600)
        def test_function(x):
            return x * 2
        
        # Test cache methods exist
        assert hasattr(test_function, 'cache')
        assert hasattr(test_function, 'cache_clear')
        assert hasattr(test_function, 'cache_info')
        
        # Use the function
        test_function(5)
        
        # Check cache info
        info = test_function.cache_info()
        assert info['hits'] == 0  # First call is always a miss
        assert info['sets'] == 1
        
        # Clear cache
        test_function.cache_clear()
        assert test_function.cache_info()['size'] == 0


class TestCachedPropertyTTL:
    """Test cases for cached_property_ttl decorator."""
    
    def test_cached_property_ttl_basic(self):
        """Test basic cached_property_ttl functionality."""
        call_count = 0
        
        class TestClass:
            @cached_property_ttl(ttl_seconds=3600)
            def expensive_property(self):
                nonlocal call_count
                call_count += 1
                return "expensive_result"
        
        obj = TestClass()
        
        # First access
        result1 = obj.expensive_property
        assert result1 == "expensive_result"
        assert call_count == 1
        
        # Second access (should use cache)
        result2 = obj.expensive_property
        assert result2 == "expensive_result"
        assert call_count == 1  # Not incremented
    
    def test_cached_property_ttl_expiration(self):
        """Test cached_property_ttl with expiration."""
        call_count = 0
        
        class TestClass:
            @cached_property_ttl(ttl_seconds=0.1)
            def expensive_property(self):
                nonlocal call_count
                call_count += 1
                return f"result_{call_count}"
        
        obj = TestClass()
        
        # First access
        result1 = obj.expensive_property
        assert result1 == "result_1"
        assert call_count == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Second access (should recompute)
        result2 = obj.expensive_property
        assert result2 == "result_2"
        assert call_count == 2
    
    def test_cached_property_ttl_deletion(self):
        """Test cached_property_ttl deletion."""
        call_count = 0
        
        class TestClass:
            @cached_property_ttl(ttl_seconds=3600)
            def expensive_property(self):
                nonlocal call_count
                call_count += 1
                return f"result_{call_count}"
        
        obj = TestClass()
        
        # Access property
        result1 = obj.expensive_property
        assert result1 == "result_1"
        assert call_count == 1
        
        # Delete cached value
        del obj.expensive_property
        
        # Access again (should recompute)
        result2 = obj.expensive_property
        assert result2 == "result_2"
        assert call_count == 2


class TestCacheKeyGeneration:
    """Test cases for cache key generation."""
    
    def test_make_cache_key_basic(self):
        """Test basic cache key generation."""
        key = _make_cache_key("func_name", (1, 2, 3), {"a": 1, "b": 2})
        
        assert "func_name" in key
        assert "1|2|3" in key
        assert "a=1" in key
        assert "b=2" in key
    
    def test_make_cache_key_no_args(self):
        """Test cache key generation with no arguments."""
        key = _make_cache_key("func_name", (), {})
        
        assert key == "func_name"
    
    def test_make_cache_key_args_only(self):
        """Test cache key generation with only positional args."""
        key = _make_cache_key("func_name", (1, 2), {})
        
        assert key == "func_name:1|2"
    
    def test_make_cache_key_kwargs_only(self):
        """Test cache key generation with only keyword args."""
        key = _make_cache_key("func_name", (), {"a": 1, "b": 2})
        
        assert key == "func_name:a=1|b=2"
    
    def test_make_cache_key_complex_objects(self):
        """Test cache key generation with complex objects."""
        # Should handle complex objects by converting to string
        key = _make_cache_key("func_name", ([1, 2, 3], {"nested": "dict"}), {})
        
        assert "func_name" in key
        # Should not raise exception


class TestGlobalCacheInstances:
    """Test cases for global cache instances."""
    
    def test_spot_data_cache_exists(self):
        """Test that spot_data_cache global instance exists."""
        assert spot_data_cache is not None
        assert isinstance(spot_data_cache, TTLCache)
        assert spot_data_cache.default_ttl_seconds == 3600
    
    def test_analysis_cache_exists(self):
        """Test that analysis_cache global instance exists."""
        assert analysis_cache is not None
        assert isinstance(analysis_cache, TTLCache)
        assert analysis_cache.default_ttl_seconds == 1800
    
    def test_cache_warmer_exists(self):
        """Test that cache_warmer global instance exists."""
        assert cache_warmer is not None
        assert isinstance(cache_warmer, CacheWarmer)
        assert cache_warmer.max_workers == 3
    
    def test_global_caches_independent(self):
        """Test that global caches are independent."""
        # Clear both caches
        spot_data_cache.clear()
        analysis_cache.clear()
        
        # Set values in each
        spot_data_cache.set("key1", "spot_value")
        analysis_cache.set("key1", "analysis_value")
        
        # Should be independent
        assert spot_data_cache.get("key1") == "spot_value"
        assert analysis_cache.get("key1") == "analysis_value"
        
        # Clean up
        spot_data_cache.clear()
        analysis_cache.clear()


class TestCacheIntegration:
    """Integration tests for caching functionality."""
    
    def test_cache_with_warming_integration(self):
        """Test integration between cache and warming."""
        cache = TTLCache()
        warmer = CacheWarmer()
        
        # Define value factories
        def factory1():
            return "warmed_value_1"
        
        def factory2():
            return "warmed_value_2"
        
        # Warm multiple entries
        entries = [
            ("key1", factory1, 1800),
            ("key2", factory2, 3600)
        ]
        
        futures = warmer.warm_multiple(cache, entries)
        warmer.wait_for_warming(timeout=5)
        
        # Check that all entries were warmed
        assert cache.get("key1") == "warmed_value_1"
        assert cache.get("key2") == "warmed_value_2"
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['sets'] == 2
        
        warmer.shutdown()
    
    def test_decorator_with_global_cache(self):
        """Test decorator integration with global cache instances."""
        # Use global cache in decorator
        call_count = 0
        
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Manually cache using global cache
        spot_data_cache.set("test_key", test_function(5))
        
        # Verify it's cached
        assert spot_data_cache.get("test_key") == 10
        
        # Clean up
        spot_data_cache.clear()