import logging
import asyncio
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import hashlib

from .errors import ProcessingError, handle_errors, ErrorBoundary
from .interfaces import PerformanceMetrics, ResourceUsage

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Asynchronous caching system with persistence and auto-cleanup.
    
    Features:
    - Memory and disk caching
    - Automatic cleanup
    - Data compression
    - Secure persistence
    - Usage metrics
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_memory_items: int = 1000,
        max_disk_items: int = 10000,
        expiration_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Limits and configuration
        self.max_memory_items = max_memory_items
        self.max_disk_items = max_disk_items
        self.expiration_hours = expiration_hours
        
        # Caches
        self.memory_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.modified_keys: Set[str] = set()
        
        # Executor for I/O operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Maintenance tasks
        self._save_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = PerformanceMetrics(
            resource_usage=ResourceUsage()
        )

    async def initialize(self) -> None:
        """Initialize the cache system."""
        try:
            # Start maintenance tasks
            self._save_task = asyncio.create_task(self._periodic_save())
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            # Load persistent cache
            await self._load_cache()
            logger.info("Cache system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing cache: {e}")
            raise ProcessingError("Cache initialization failed", {"error": str(e)})

    @handle_errors(default_return=None)
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Key to look up
            
        Returns:
            Cached value or None if not exists
        """
        # Check memory cache
        if key in self.memory_cache:
            self.access_times[key] = datetime.now()
            self.metrics.cache_hits += 1
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            # Load from disk
            cached_data = await self._read_cache_file(cache_path)
            if not cached_data:
                self.metrics.cache_misses += 1
                return None
            
            # Check expiration
            if self._is_expired(cached_data["timestamp"]):
                await self._remove_cache_file(cache_path)
                self.metrics.cache_misses += 1
                return None
            
            # Move to memory cache
            self.memory_cache[key] = cached_data["data"]
            self.access_times[key] = datetime.now()
            self.metrics.cache_hits += 1
            
            return cached_data["data"]
        
        self.metrics.cache_misses += 1
        return None

    @handle_errors()
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Save a value to cache.
        
        Args:
            key: Key to store under
            value: Value to store
            ttl: Time to live in seconds (optional)
        """
        # Prepare data
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "ttl": ttl,
            "data": value
        }
        
        # Save to memory
        self.memory_cache[key] = value
        self.access_times[key] = datetime.now()
        self.modified_keys.add(key)
        
        # Check memory limit
        if len(self.memory_cache) > self.max_memory_items:
            await self._evict_memory_items()
        
        # Save to disk
        cache_path = self._get_cache_path(key)
        await self._write_cache_file(cache_path, cache_data)

        # Update metrics
        self.metrics.total_requests += 1

    def _get_cache_path(self, key: str) -> Path:
        """Generate path for cache file."""
        # Use hash to avoid filename issues
        safe_key = base64.urlsafe_b64encode(
            hashlib.sha256(key.encode()).digest()
        ).decode()
        return self.cache_dir / f"{safe_key}.cache"

    @handle_errors(default_return=None)
    async def _read_cache_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read a cache file safely."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._read_file,
            path
        )

    def _read_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read and deserialize a cache file."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return None

    @handle_errors()
    async def _write_cache_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Write a cache file safely."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._write_file,
            path,
            data
        )

    def _write_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Serialize and write data to a file."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            raise ProcessingError("File write failed", {"path": str(path)})

    @handle_errors()
    async def _remove_cache_file(self, path: Path) -> None:
        """Remove a cache file safely."""
        if path.exists():
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                path.unlink
            )

    async def _evict_memory_items(self) -> None:
        """Remove least used items from memory."""
        try:
            # Sort by access time
            sorted_items = sorted(
                self.access_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove 25% of oldest items
            items_to_remove = len(sorted_items) // 4
            
            for key, _ in sorted_items[:items_to_remove]:
                self.memory_cache.pop(key, None)
                self.access_times.pop(key, None)
                
        except Exception as e:
            logger.error(f"Error evicting memory items: {e}")

    def _is_expired(self, timestamp: str) -> bool:
        """Check if a timestamp has expired."""
        try:
            item_time = datetime.fromisoformat(timestamp)
            expiration = item_time + timedelta(hours=self.expiration_hours)
            return datetime.now() > expiration
        except Exception as e:
            logger.error(f"Error checking expiration: {e}")
            return True

    async def _periodic_save(self) -> None:
        """Periodically save modified items to disk."""
        try:
            while True:
                await asyncio.sleep(60)  # Every minute
                
                modified = self.modified_keys.copy()
                self.modified_keys.clear()
                
                for key in modified:
                    if key in self.memory_cache:
                        cache_path = self._get_cache_path(key)
                        await self._write_cache_file(
                            cache_path,
                            {
                                "timestamp": datetime.now().isoformat(),
                                "data": self.memory_cache[key]
                            }
                        )
                        
        except asyncio.CancelledError:
            logger.info("Periodic save task cancelled")
        except Exception as e:
            logger.error(f"Error in periodic save: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up expired items."""
        try:
            while True:
                await asyncio.sleep(3600)  # Every hour
                
                # Clean memory
                now = datetime.now()
                expired_keys = [
                    key for key, access_time in self.access_times.items()
                    if (now - access_time).total_seconds() > self.expiration_hours * 3600
                ]
                
                for key in expired_keys:
                    self.memory_cache.pop(key, None)
                    self.access_times.pop(key, None)
                    cache_path = self._get_cache_path(key)
                    await self._remove_cache_file(cache_path)
                    
                # Clean disk
                await self._cleanup_disk_cache()
                
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

    @handle_errors()
    async def _cleanup_disk_cache(self) -> None:
        """Clean old files from disk."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        if len(cache_files) <= self.max_disk_items:
            return
            
        # Sort by modification time
        sorted_files = sorted(
            cache_files,
            key=lambda p: p.stat().st_mtime
        )
        
        # Remove oldest files
        files_to_remove = len(sorted_files) - self.max_disk_items
        
        for path in sorted_files[:files_to_remove]:
            await self._remove_cache_file(path)

    async def _load_cache(self) -> None:
        """Load initial cache from disk."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            loaded = 0
            
            for cache_file in cache_files[:self.max_memory_items]:
                cached_data = await self._read_cache_file(cache_file)
                if cached_data and not self._is_expired(cached_data["timestamp"]):
                    key = cache_file.stem
                    self.memory_cache[key] = cached_data["data"]
                    self.access_times[key] = datetime.now()
                    loaded += 1
                    
            logger.info(f"Loaded {loaded} items from disk cache")
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.cache"))),
            "modified_keys": len(self.modified_keys),
            "performance_metrics": {
                "total_requests": self.metrics.total_requests,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "hit_ratio": (
                    self.metrics.cache_hits / 
                    (self.metrics.cache_hits + self.metrics.cache_misses)
                    if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
                    else 0
                )
            }
        }

    async def cleanup(self) -> None:
        """Clean up system resources."""
        try:
            # Cancel periodic tasks
            if self._save_task:
                self._save_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
                
            # Save modified items
            modified = self.modified_keys.copy()
            for key in modified:
                if key in self.memory_cache:
                    cache_path = self._get_cache_path(key)
                    await self._write_cache_file(
                        cache_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "data": self.memory_cache[key]
                        }
                    )
                    
            # Clear memory
            self.memory_cache.clear()
            self.access_times.clear()
            self.modified_keys.clear()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            logger.info("Cache system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")