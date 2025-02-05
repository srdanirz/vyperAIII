import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import base64

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Cache Manager con soporte asíncrono y limpieza periódica.
    """
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
        """Inicia las tareas de guardado periódico y limpieza."""
        try:
            self._save_task = asyncio.create_task(self._periodic_save())
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            await self._load_cache()
        except Exception as e:
            logger.error(f"Error initializing cache manager: {e}")
            raise

    def _get_cache_key(self, prompt: str) -> str:
        normalized_prompt = " ".join(prompt.lower().split())
        truncated_prompt = normalized_prompt[:100]
        return f"{truncated_prompt}_{hash(normalized_prompt)}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.cache"

    async def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Obtiene un valor de caché."""
        try:
            cache_key = self._get_cache_key(prompt)
            if cache_key in self.memory_cache:
                self.access_times[cache_key] = datetime.now()
                return self.memory_cache[cache_key]
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                return None
            loop = asyncio.get_running_loop()
            cached_data = await loop.run_in_executor(
                self._executor,
                self._read_cache_file,
                cache_path
            )
            if not cached_data:
                return None
            expiry = cached_data['timestamp'] + timedelta(hours=self.expiration_hours)
            if datetime.now() > expiry:
                await self._remove_cache_file(cache_path)
                return None
            self.memory_cache[cache_key] = cached_data['data']
            self.access_times[cache_key] = datetime.now()
            return cached_data['data']
        except Exception as e:
            logger.error(f"Error getting cached data: {e}", exc_info=True)
            return None

    async def set(self, prompt: str, data: Dict[str, Any]) -> None:
        """Setea un valor en caché."""
        try:
            cache_key = self._get_cache_key(prompt)
            cache_data = {
                'timestamp': datetime.now(),
                'data': data
            }
            self.memory_cache[cache_key] = data
            self.access_times[cache_key] = datetime.now()
            self.modified_keys.add(cache_key)
            cache_path = self._get_cache_path(cache_key)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self._write_cache_file,
                cache_path,
                cache_data
            )
        except Exception as e:
            logger.error(f"Error setting cache data: {e}", exc_info=True)

    def _read_cache_file(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with cache_path.open('rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading cache file {cache_path}: {e}", exc_info=True)
            return None

    def _write_cache_file(self, cache_path: Path, cache_data: Dict[str, Any]) -> None:
        try:
            with cache_path.open('wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}", exc_info=True)

    async def _remove_cache_file(self, cache_path: Path) -> None:
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            logger.error(f"Error removing cache file {cache_path}: {e}", exc_info=True)

    async def _load_cache(self) -> None:
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
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
            logger.error(f"Error loading cache: {e}", exc_info=True)

    async def _periodic_save(self) -> None:
        """Guarda en disco las entradas modificadas cada cierto tiempo."""
        try:
            while True:
                await asyncio.sleep(60)
                modified = self.modified_keys.copy()
                self.modified_keys.clear()
                for key in modified:
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
        except asyncio.CancelledError:
            logger.info("Periodic save task cancelled.")
        except Exception as e:
            logger.error(f"Error in periodic save: {e}", exc_info=True)

    async def _periodic_cleanup(self) -> None:
        """Limpia entradas expiradas cada hora."""
        try:
            while True:
                await asyncio.sleep(3600)
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
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled.")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Cancela tareas en curso y persiste datos."""
        if self._save_task:
            self._save_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        modified = self.modified_keys.copy()
        for key in modified:
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
        self._executor.shutdown(wait=True)
