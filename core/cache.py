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

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Sistema de caché asíncrono con persistencia y limpieza automática.
    
    Features:
    - Caché en memoria y disco
    - Limpieza automática
    - Compresión de datos
    - Persistencia segura
    - Métricas de uso
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
        
        # Límites y configuración
        self.max_memory_items = max_memory_items
        self.max_disk_items = max_disk_items
        self.expiration_hours = expiration_hours
        
        # Cachés
        self.memory_cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.modified_keys: Set[str] = set()
        
        # Executor para operaciones de I/O
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Tareas de mantenimiento
        self._save_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Métricas
        self.metrics = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "evictions": 0,
            "writes": 0
        }

    async def initialize(self) -> None:
        """Inicializa el sistema de caché."""
        try:
            # Iniciar tareas de mantenimiento
            self._save_task = asyncio.create_task(self._periodic_save())
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            # Cargar caché persistente
            await self._load_cache()
            logger.info("Cache system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing cache: {e}")
            raise ProcessingError("Cache initialization failed", {"error": str(e)})

    @handle_errors(default_return=None)
    async def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del caché.
        
        Args:
            key: Clave a buscar
            
        Returns:
            Valor cacheado o None si no existe
        """
        # Verificar caché en memoria
        if key in self.memory_cache:
            self.access_times[key] = datetime.now()
            self.metrics["memory_hits"] += 1
            return self.memory_cache[key]
        
        # Verificar caché en disco
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            # Cargar desde disco
            cached_data = await self._read_cache_file(cache_path)
            if not cached_data:
                return None
            
            # Verificar expiración
            if self._is_expired(cached_data["timestamp"]):
                await self._remove_cache_file(cache_path)
                self.metrics["evictions"] += 1
                return None
            
            # Mover a caché en memoria
            self.memory_cache[key] = cached_data["data"]
            self.access_times[key] = datetime.now()
            self.metrics["disk_hits"] += 1
            
            return cached_data["data"]
        
        self.metrics["misses"] += 1
        return None

    @handle_errors()
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Guarda un valor en caché.
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos (opcional)
        """
        # Preparar datos
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "ttl": ttl,
            "data": value
        }
        
        # Guardar en memoria
        self.memory_cache[key] = value
        self.access_times[key] = datetime.now()
        self.modified_keys.add(key)
        self.metrics["writes"] += 1
        
        # Verificar límite de memoria
        if len(self.memory_cache) > self.max_memory_items:
            await self._evict_memory_items()
        
        # Guardar en disco
        cache_path = self._get_cache_path(key)
        await self._write_cache_file(cache_path, cache_data)

    def _get_cache_path(self, key: str) -> Path:
        """Genera path para archivo de caché."""
        # Usar hash para evitar problemas con nombres de archivo
        safe_key = base64.urlsafe_b64encode(
            hashlib.sha256(key.encode()).digest()
        ).decode()
        return self.cache_dir / f"{safe_key}.cache"

    @handle_errors(default_return=None)
    async def _read_cache_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Lee un archivo de caché de forma segura."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._read_file,
            path
        )

    def _read_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Lee y deserializa un archivo de caché."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return None

    @handle_errors()
    async def _write_cache_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Escribe un archivo de caché de forma segura."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._write_file,
            path,
            data
        )

    def _write_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Serializa y escribe datos a un archivo."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            raise ProcessingError("File write failed", {"path": str(path)})

    @handle_errors()
    async def _remove_cache_file(self, path: Path) -> None:
        """Elimina un archivo de caché de forma segura."""
        if path.exists():
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                path.unlink
            )

    async def _evict_memory_items(self) -> None:
        """Elimina items menos usados de la memoria."""
        try:
            # Ordenar por tiempo de acceso
            sorted_items = sorted(
                self.access_times.items(),
                key=lambda x: x[1]
            )
            
            # Eliminar 25% de los items más antiguos
            items_to_remove = len(sorted_items) // 4
            
            for key, _ in sorted_items[:items_to_remove]:
                self.memory_cache.pop(key, None)
                self.access_times.pop(key, None)
                self.metrics["evictions"] += 1
                
        except Exception as e:
            logger.error(f"Error evicting memory items: {e}")

    def _is_expired(self, timestamp: str) -> bool:
        """Verifica si un timestamp ha expirado."""
        try:
            item_time = datetime.fromisoformat(timestamp)
            expiration = item_time + timedelta(hours=self.expiration_hours)
            return datetime.now() > expiration
        except Exception as e:
            logger.error(f"Error checking expiration: {e}")
            return True

    async def _periodic_save(self) -> None:
        """Guarda periódicamente items modificados en disco."""
        try:
            while True:
                await asyncio.sleep(60)  # Cada minuto
                
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
        """Limpia periódicamente items expirados."""
        try:
            while True:
                await asyncio.sleep(3600)  # Cada hora
                
                # Limpiar memoria
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
                    self.metrics["evictions"] += 1
                    
                # Limpiar disco
                await self._cleanup_disk_cache()
                
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

    @handle_errors()
    async def _cleanup_disk_cache(self) -> None:
        """Limpia archivos antiguos del disco."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        if len(cache_files) <= self.max_disk_items:
            return
            
        # Ordenar por fecha de modificación
        sorted_files = sorted(
            cache_files,
            key=lambda p: p.stat().st_mtime
        )
        
        # Eliminar archivos más antiguos
        files_to_remove = len(sorted_files) - self.max_disk_items
        
        for path in sorted_files[:files_to_remove]:
            await self._remove_cache_file(path)
            self.metrics["evictions"] += 1

    async def _load_cache(self) -> None:
        """Carga caché inicial desde disco."""
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
        """Obtiene estadísticas del sistema de caché."""
        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.cache"))),
            "modified_keys": len(self.modified_keys),
            "metrics": self.metrics
        }

    async def cleanup(self) -> None:
        """Limpia recursos del sistema de caché."""
        try:
            # Cancelar tareas periódicas
            if self._save_task:
                self._save_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
                
            # Guardar items modificados
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
                    
            # Limpiar memoria
            self.memory_cache.clear()
            self.access_times.clear()
            self.modified_keys.clear()
            
            # Cerrar executor
            self._executor.shutdown(wait=True)
            
            logger.info("Cache system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")