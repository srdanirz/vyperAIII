import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CacheManager:
    """Enhanced cache manager with async support and better error handling"""
    
    def __init__(self, cache_dir: str = "cache", expiration_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.expiration_hours = expiration_hours
        self.memory_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.modified_keys: Set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._save_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize cache manager and start background tasks"""
        try:
            # Start periodic save task
            self._save_task = asyncio.create_task(self._periodic_save())
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            # Load existing cache
            await self._load_cache()
            
        except Exception as e:
            logger.error(f"Error initializing cache manager: {e}")
            raise

    def _get_cache_key(self, prompt: str) -> str:
        """Generate unique cache key"""
        # Normalize prompt
        normalized_prompt = " ".join(prompt.lower().split())
        
        # Use first 100 chars for readability while still maintaining uniqueness
        truncated_prompt = normalized_prompt[:100]
        
        # Combine hash and truncated prompt for better debugging
        return f"{truncated_prompt}_{hash(normalized_prompt)}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.cache"

    async def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached result with async file operations"""
        try:
            cache_key = self._get_cache_key(prompt)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.access_times[cache_key] = datetime.now()
                return self.memory_cache[cache_key]
                
            # Check file cache
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                return None

            # Use thread pool for file operations
            loop = asyncio.get_running_loop()
            cached_data = await loop.run_in_executor(
                self._executor,
                self._read_cache_file,
                cache_path
            )
            
            if not cached_data:
                return None

            # Check expiration
            cached_time = cached_data.get('timestamp')
            if cached_time:
                expiration = cached_time + timedelta(hours=self.expiration_hours)
                if datetime.now() > expiration:
                    await self._remove_cache_file(cache_path)
                    return None

            # Update memory cache
            self.memory_cache[cache_key] = cached_data.get('data')
            self.access_times[cache_key] = datetime.now()
            
            return cached_data.get('data')

        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None

    async def set(self, prompt: str, data: Dict[str, Any]) -> None:
        """Set cache value with async file operations"""
        try:
            cache_key = self._get_cache_key(prompt)
            cache_data = {
                'timestamp': datetime.now(),
                'data': data
            }
            
            # Update memory cache
            self.memory_cache[cache_key] = data
            self.access_times[cache_key] = datetime.now()
            self.modified_keys.add(cache_key)
            
            # Schedule save to disk
            cache_path = self._get_cache_path(cache_key)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self._write_cache_file,
                cache_path,
                cache_data
            )
            
        except Exception as e:
            logger.error(f"Error setting cache data: {e}")

    def _read_cache_file(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Read cache file (runs in thread pool)"""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading cache file {cache_path}: {e}")
            return None

    def _write_cache_file(self, cache_path: Path, cache_data: Dict[str, Any]) -> None:
        """Write cache file (runs in thread pool)"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")

    async def _remove_cache_file(self, cache_path: Path) -> None:
        """Remove cache file"""
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            logger.error(f"Error removing cache file {cache_path}: {e}")

    async def _load_cache(self) -> None:
        """Load existing cache files into memory"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cached_data = await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._read_cache_file,
                        cache_file
                    )
                    
                    if cached_data and 'data' in cached_data:
                        cache_key = cache_file.stem
                        self.memory_cache[cache_key] = cached_data['data']
                        self.access_times[cache_key] = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Error loading cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    async def _periodic_save(self) -> None:
        """Periodically save modified cache entries"""
        try:
            while True:
                await asyncio.sleep(60)  # Save every minute
                
                modified_keys = self.modified_keys.copy()
                self.modified_keys.clear()
                
                for key in modified_keys:
                    cache_path = self._get_cache_path(key)
                    cache_data = {
                        'timestamp': datetime.now(),
                        'data': self.memory_cache[key]
                    }
                    
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._write_cache_file,
                        cache_path,
                        cache_data
                    )
                    
        except Exception as e:
            logger.error(f"Error in periodic save: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodically cleanup expired cache entries"""
        try:
            while True:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                current_time = datetime.now()
                
                # Cleanup memory cache
                expired_keys = [
                    key for key, access_time in self.access_times.items()
                    if (current_time - access_time).total_seconds() > self.expiration_hours * 3600
                ]
                
                for key in expired_keys:
                    self.memory_cache.pop(key, None)
                    self.access_times.pop(key, None)
                    cache_path = self._get_cache_path(key)
                    await self._remove_cache_file(cache_path)
                    
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Cancel background tasks
            if self._save_task:
                self._save_task.cancel()
                
            if self._cleanup_task:
                self._cleanup_task.cancel()
                
            # Save any remaining modified entries
            modified_keys = self.modified_keys.copy()
            for key in modified_keys:
                cache_path = self._get_cache_path(key)
                cache_data = {
                    'timestamp': datetime.now(),
                    'data': self.memory_cache[key]
                }
                await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._write_cache_file,
                    cache_path,
                    cache_data
                )
                
            # Shutdown thread pool
            self._executor.shutdown(wait=True)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "memory_cache_size": len(self.memory_cache),
            "modified_entries": len(self.modified_keys),
            "cache_files": len(list(self.cache_dir.glob("*.cache"))),
            "oldest_entry": min(self.access_times.values()).isoformat() if self.access_times else None,
            "newest_entry": max(self.access_times.values()).isoformat() if self.access_times else None,
            "expiration_hours": self.expiration_hours,
            "cache_directory": str(self.cache_dir)
        }