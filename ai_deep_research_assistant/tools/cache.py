"""
In-memory caching system for AI Deep Research Assistant.

This module provides an async-safe LRU cache with TTL (Time To Live) functionality
for caching search results, scraped content, and other expensive operations to
improve performance and reduce API usage.
"""
import asyncio
import time
import hashlib
import json
import logging
from typing import Any, Optional, Dict, List, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Represents a cached entry with value, timestamp, and metadata"""
    value: T
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()


class AsyncLRUCache:
    """
    Async-safe LRU cache with TTL support for high-performance caching.
    
    Features:
    - TTL (Time To Live) support with automatic expiry
    - LRU (Least Recently Used) eviction policy
    - Thread-safe operations with asyncio locks
    - Cache statistics and monitoring
    - Configurable size limits
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 900,  # 15 minutes
        cleanup_interval: float = 300,  # 5 minutes
        enable_stats: bool = True
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds (15 minutes)
            cleanup_interval: How often to run cleanup in seconds
            enable_stats: Whether to track cache statistics
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_stats = enable_stats
        
        # Thread-safe cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0,
            'sets': 0,
            'deletes': 0
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._should_cleanup = True
        
    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._should_cleanup = True
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info(f"🧹 Started cache cleanup task (interval: {self.cleanup_interval}s)")
    
    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        self._should_cleanup = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 Stopped cache cleanup task")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                if self.enable_stats:
                    self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if entry is expired
            if entry.is_expired():
                del self._cache[key]
                if self.enable_stats:
                    self._stats['expired'] += 1
                    self._stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.update_access()
            
            if self.enable_stats:
                self._stats['hits'] += 1
            
            logger.debug(f"💾 Cache hit for key: {key[:16]}...")
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            
            # Check if we need to evict entries
            while len(self._cache) >= self.max_size:
                # Remove least recently used item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if self.enable_stats:
                    self._stats['evictions'] += 1
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            self._cache[key] = entry
            
            if self.enable_stats:
                self._stats['sets'] += 1
            
            logger.debug(f"💾 Cache set for key: {key[:16]}... (TTL: {ttl}s)")
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self.enable_stats:
                    self._stats['deletes'] += 1
                logger.debug(f"🗑️ Cache delete for key: {key[:16]}...")
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries from the cache"""
        async with self._lock:
            self._cache.clear()
            logger.info("🧹 Cache cleared")
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        current_time = time.time()
        
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time > (entry.timestamp + entry.ttl)
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
            
            if self.enable_stats:
                self._stats['expired'] += removed_count
        
        if removed_count > 0:
            logger.debug(f"🧹 Cleaned up {removed_count} expired cache entries")
        
        return removed_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'expired': self._stats['expired'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'total_requests': total_requests
            }
    
    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all cache keys, optionally filtered by pattern.
        
        Args:
            pattern: Optional pattern to filter keys
            
        Returns:
            List of cache keys
        """
        async with self._lock:
            keys = list(self._cache.keys())
            
            if pattern:
                import fnmatch
                keys = [key for key in keys if fnmatch.fnmatch(key, pattern)]
            
            return keys
    
    async def _periodic_cleanup(self):
        """Background task to periodically clean up expired entries"""
        while self._should_cleanup:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if self._should_cleanup:  # Check again after sleep
                    await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error during cache cleanup: {e}")


class CacheManager:
    """
    High-level cache manager for different types of cached data.
    
    Provides specialized caching for:
    - Search results
    - Scraped content  
    - Model responses
    - Research findings
    """
    
    def __init__(
        self,
        search_cache_ttl: float = 900,    # 15 minutes
        content_cache_ttl: float = 3600,  # 1 hour
        response_cache_ttl: float = 1800, # 30 minutes
        max_size: int = 1000
    ):
        """
        Initialize the cache manager.
        
        Args:
            search_cache_ttl: TTL for search results cache
            content_cache_ttl: TTL for scraped content cache
            response_cache_ttl: TTL for model responses cache
            max_size: Maximum cache size
        """
        self.cache = AsyncLRUCache(
            max_size=max_size,
            default_ttl=search_cache_ttl
        )
        
        self.search_cache_ttl = search_cache_ttl
        self.content_cache_ttl = content_cache_ttl
        self.response_cache_ttl = response_cache_ttl
        
    async def start(self):
        """Start the cache manager"""
        await self.cache.start_cleanup_task()
        logger.info("🚀 Cache manager started")
    
    async def stop(self):
        """Stop the cache manager"""
        await self.cache.stop_cleanup_task()
        logger.info("🛑 Cache manager stopped")
    
    def _generate_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate a cache key from data"""
        # Sort data for consistent key generation
        sorted_data = json.dumps(data, sort_keys=True, default=str)
        key_hash = hashlib.md5(sorted_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def cache_search_results(
        self,
        query: str,
        search_type: str,
        results: List[Dict[str, Any]],
        count: int = 8
    ) -> str:
        """Cache search results and return cache key"""
        key_data = {
            'query': query.lower().strip(),
            'search_type': search_type,
            'count': count
        }
        
        cache_key = self._generate_key('search', key_data)
        await self.cache.set(cache_key, results, ttl=self.search_cache_ttl)
        
        return cache_key
    
    async def get_cached_search_results(
        self,
        query: str,
        search_type: str,
        count: int = 8
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        key_data = {
            'query': query.lower().strip(),
            'search_type': search_type,
            'count': count
        }
        
        cache_key = self._generate_key('search', key_data)
        return await self.cache.get(cache_key)
    
    async def cache_scraped_content(
        self,
        url: str,
        content: Dict[str, Any]
    ) -> str:
        """Cache scraped content and return cache key"""
        key_data = {'url': url}
        cache_key = self._generate_key('content', key_data)
        await self.cache.set(cache_key, content, ttl=self.content_cache_ttl)
        
        return cache_key
    
    async def get_cached_scraped_content(
        self,
        url: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached scraped content"""
        key_data = {'url': url}
        cache_key = self._generate_key('content', key_data)
        return await self.cache.get(cache_key)
    
    async def cache_model_response(
        self,
        prompt_hash: str,
        response: Dict[str, Any],
        model_name: str
    ) -> str:
        """Cache model response and return cache key"""
        key_data = {
            'prompt_hash': prompt_hash,
            'model': model_name
        }
        
        cache_key = self._generate_key('response', key_data)
        await self.cache.set(cache_key, response, ttl=self.response_cache_ttl)
        
        return cache_key
    
    async def get_cached_model_response(
        self,
        prompt_hash: str,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached model response"""
        key_data = {
            'prompt_hash': prompt_hash,
            'model': model_name
        }
        
        cache_key = self._generate_key('response', key_data)
        return await self.cache.get(cache_key)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        base_stats = await self.cache.get_stats()
        
        # Add cache manager specific stats
        search_keys = await self.cache.get_keys('search:*')
        content_keys = await self.cache.get_keys('content:*')
        response_keys = await self.cache.get_keys('response:*')
        
        return {
            **base_stats,
            'search_entries': len(search_keys),
            'content_entries': len(content_keys),
            'response_entries': len(response_keys),
            'ttl_config': {
                'search': self.search_cache_ttl,
                'content': self.content_cache_ttl,
                'response': self.response_cache_ttl
            }
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance (async singleton pattern).
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        from .settings import get_settings
        
        settings = get_settings()
        _cache_manager = CacheManager(
            max_size=settings.cache_max_size,
            search_cache_ttl=settings.cache_ttl_minutes * 60
        )
        await _cache_manager.start()
    
    return _cache_manager


if __name__ == "__main__":
    """Test cache functionality when run directly"""
    
    async def test_cache():
        print("🧪 Testing cache functionality...")
        
        # Test basic cache operations
        cache = AsyncLRUCache(max_size=5, default_ttl=2)
        await cache.start_cleanup_task()
        
        # Test set and get
        await cache.set('key1', 'value1')
        result = await cache.get('key1')
        print(f"✅ Cache set/get: {result}")
        
        # Test TTL expiry
        await cache.set('key2', 'value2', ttl=1)
        print("⏳ Waiting for TTL expiry...")
        await asyncio.sleep(1.5)
        expired_result = await cache.get('key2')
        print(f"✅ TTL expiry test: {expired_result} (should be None)")
        
        # Test LRU eviction
        for i in range(6):
            await cache.set(f'key{i+3}', f'value{i+3}')
        
        # key1 should be evicted
        evicted_result = await cache.get('key1')
        print(f"✅ LRU eviction test: {evicted_result} (should be None)")
        
        # Test stats
        stats = await cache.get_stats()
        print(f"📊 Cache stats: {stats}")
        
        await cache.stop_cleanup_task()
        
        # Test cache manager
        manager = CacheManager()
        await manager.start()
        
        # Test search result caching
        search_results = [{'title': 'Test', 'url': 'https://example.com'}]
        cache_key = await manager.cache_search_results('test query', 'general', search_results)
        cached_results = await manager.get_cached_search_results('test query', 'general')
        
        print(f"✅ Search cache test: {len(cached_results) if cached_results else 0} results")
        
        manager_stats = await manager.get_cache_stats()
        print(f"📊 Manager stats: {manager_stats}")
        
        await manager.stop()
    
    # Run the test
    asyncio.run(test_cache())