"""
Advanced caching utilities for AWS Spot Price Analyzer.

This module provides comprehensive caching functionality including TTL-based caching,
cache warming strategies, and integration with functools.lru_cache.
"""

import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future

from src.utils.exceptions import CacheError


logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """
    Cache entry with TTL and metadata.
    
    Attributes:
        value: Cached value
        timestamp: When the entry was created
        ttl_seconds: Time-to-live in seconds
        access_count: Number of times accessed
        last_access: Last access timestamp
    """
    value: T
    timestamp: datetime
    ttl_seconds: float
    access_count: int = 0
    last_access: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # Never expires
        
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()
    
    def access(self) -> T:
        """Access the cached value and update access metadata."""
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc)
        return self.value


class TTLCache(Generic[T]):
    """
    Thread-safe TTL-based cache with advanced features.
    
    This cache provides TTL-based expiration, access tracking,
    and cache warming capabilities.
    """
    
    def __init__(self, default_ttl_seconds: float = 3600, max_size: int = 1000):
        """
        Initialize TTL cache.
        
        Args:
            default_ttl_seconds: Default TTL for cache entries
            max_size: Maximum number of cache entries
        """
        self.default_ttl_seconds = default_ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                logger.debug(f"Cache entry expired for key: {key}")
                del self._cache[key]
                self._stats['misses'] += 1
                return None
            
            self._stats['hits'] += 1
            return entry.access()
    
    def set(self, key: str, value: T, ttl_seconds: Optional[float] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL for this entry (uses default if None)
        """
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds
        
        with self._lock:
            # Evict expired entries if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_expired()
                
                # If still full, evict least recently used
                if len(self._cache) >= self.max_size:
                    self._evict_lru()
            
            entry = CacheEntry(
                value=value,
                timestamp=datetime.now(timezone.utc),
                ttl_seconds=ttl_seconds
            )
            
            self._cache[key] = entry
            self._stats['sets'] += 1
            
            logger.debug(f"Cached value for key: {key} (TTL: {ttl_seconds}s)")
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Deleted cache entry for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'sets': self._stats['sets'],
                'default_ttl_seconds': self.default_ttl_seconds
            }
    
    def get_entries_info(self) -> List[Dict[str, Any]]:
        """Get information about all cache entries."""
        with self._lock:
            entries = []
            current_time = datetime.now(timezone.utc)
            
            for key, entry in self._cache.items():
                entry_info = {
                    'key': key,
                    'age_seconds': entry.get_age_seconds(),
                    'ttl_seconds': entry.ttl_seconds,
                    'is_expired': entry.is_expired(),
                    'access_count': entry.access_count,
                    'last_access': entry.last_access.isoformat() if entry.last_access else None,
                    'created': entry.timestamp.isoformat()
                }
                entries.append(entry_info)
            
            return entries
    
    def _evict_expired(self) -> None:
        """Evict all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._stats['evictions'] += 1
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find entry with oldest last_access (or oldest timestamp if never accessed)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access or self._cache[k].timestamp
        )
        
        del self._cache[lru_key]
        self._stats['evictions'] += 1
        logger.debug(f"Evicted LRU cache entry: {lru_key}")


class CacheWarmer:
    """
    Cache warming utility for preloading frequently accessed data.
    
    This class provides strategies for warming caches to improve performance
    by preloading data before it's requested.
    """
    
    def __init__(self, max_workers: int = 3):
        """
        Initialize cache warmer.
        
        Args:
            max_workers: Maximum number of worker threads for warming
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._warming_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
    
    def warm_cache(
        self,
        cache: TTLCache,
        key: str,
        value_factory: Callable[[], T],
        ttl_seconds: Optional[float] = None,
        force: bool = False
    ) -> Future[bool]:
        """
        Warm cache entry asynchronously.
        
        Args:
            cache: Cache to warm
            key: Cache key
            value_factory: Function to generate cache value
            ttl_seconds: TTL for cached value
            force: Force warming even if entry exists
            
        Returns:
            Future that resolves to True if warming succeeded
        """
        with self._lock:
            # Cancel existing warming for this key
            if key in self._warming_futures:
                self._warming_futures[key].cancel()
            
            # Check if warming is needed
            if not force and cache.get(key) is not None:
                # Entry already exists and not forcing
                future = Future()
                future.set_result(False)
                return future
            
            # Submit warming task
            future = self._executor.submit(
                self._warm_entry, cache, key, value_factory, ttl_seconds
            )
            self._warming_futures[key] = future
            
            # Clean up completed futures
            self._cleanup_futures()
            
            return future
    
    def warm_multiple(
        self,
        cache: TTLCache,
        entries: List[Tuple[str, Callable[[], T], Optional[float]]],
        force: bool = False
    ) -> List[Future[bool]]:
        """
        Warm multiple cache entries.
        
        Args:
            cache: Cache to warm
            entries: List of (key, value_factory, ttl_seconds) tuples
            force: Force warming even if entries exist
            
        Returns:
            List of futures for warming tasks
        """
        futures = []
        for key, value_factory, ttl_seconds in entries:
            future = self.warm_cache(cache, key, value_factory, ttl_seconds, force)
            futures.append(future)
        
        return futures
    
    def wait_for_warming(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all warming tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        with self._lock:
            futures = list(self._warming_futures.values())
        
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.warning(f"Cache warming task failed: {e}")
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the cache warmer.
        
        Args:
            wait: Whether to wait for pending tasks
        """
        self._executor.shutdown(wait=wait)
    
    def _warm_entry(
        self,
        cache: TTLCache,
        key: str,
        value_factory: Callable[[], T],
        ttl_seconds: Optional[float]
    ) -> bool:
        """
        Warm a single cache entry.
        
        Args:
            cache: Cache to warm
            key: Cache key
            value_factory: Function to generate value
            ttl_seconds: TTL for cached value
            
        Returns:
            True if warming succeeded
        """
        try:
            logger.debug(f"Warming cache entry: {key}")
            value = value_factory()
            cache.set(key, value, ttl_seconds)
            logger.debug(f"Successfully warmed cache entry: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to warm cache entry {key}: {e}")
            return False
    
    def _cleanup_futures(self) -> None:
        """Clean up completed futures."""
        completed_keys = [
            key for key, future in self._warming_futures.items()
            if future.done()
        ]
        
        for key in completed_keys:
            del self._warming_futures[key]


def ttl_cache(ttl_seconds: float = 3600, maxsize: int = 128):
    """
    Decorator for TTL-based caching of function results.
    
    This decorator combines functools.lru_cache with TTL-based expiration.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        maxsize: Maximum cache size
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Create TTL cache for this function
        cache = TTLCache(default_ttl_seconds=ttl_seconds, max_size=maxsize)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key = _make_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        # Add cache management methods
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = cache.get_stats
        
        return wrapper
    
    return decorator


def cached_property_ttl(ttl_seconds: float = 3600):
    """
    Decorator for TTL-based caching of property values.
    
    Args:
        ttl_seconds: Time-to-live for cached property value
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> property:
        cache_attr = f"_cached_{func.__name__}"
        timestamp_attr = f"_cached_{func.__name__}_timestamp"
        
        def getter(self):
            now = time.time()
            
            # Check if we have cached value and it's not expired
            if (hasattr(self, cache_attr) and 
                hasattr(self, timestamp_attr) and
                now - getattr(self, timestamp_attr) < ttl_seconds):
                return getattr(self, cache_attr)
            
            # Compute and cache new value
            value = func(self)
            setattr(self, cache_attr, value)
            setattr(self, timestamp_attr, now)
            
            return value
        
        def deleter(self):
            if hasattr(self, cache_attr):
                delattr(self, cache_attr)
            if hasattr(self, timestamp_attr):
                delattr(self, timestamp_attr)
        
        return property(getter, None, deleter)
    
    return decorator


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Create cache key from function name and arguments.
    
    Args:
        func_name: Function name
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    try:
        # Convert args and kwargs to strings
        args_str = "|".join(str(arg) for arg in args)
        kwargs_str = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Combine into cache key
        parts = [func_name]
        if args_str:
            parts.append(args_str)
        if kwargs_str:
            parts.append(kwargs_str)
        
        return ":".join(parts)
        
    except Exception as e:
        # Fallback to hash-based key
        import hashlib
        content = f"{func_name}:{args}:{kwargs}"
        return hashlib.md5(content.encode()).hexdigest()


# Global cache instances for common use cases
spot_data_cache = TTLCache[Any](default_ttl_seconds=3600, max_size=100)
analysis_cache = TTLCache[Any](default_ttl_seconds=1800, max_size=50)

# Global cache warmer
cache_warmer = CacheWarmer(max_workers=3)