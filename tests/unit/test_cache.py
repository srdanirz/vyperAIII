import pytest
import asyncio
import zlib
import base64
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import pickle
import os
from unittest.mock import Mock, patch, AsyncMock

from core.cache import CacheManager
from core.errors import ProcessingError
from core.interfaces import PerformanceMetrics, ResourceUsage

@pytest.fixture
async def cache_manager():
    """Create test cache manager instance."""
    cache_dir = Path("test_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir()
    
    manager = CacheManager(
        cache_dir=str(cache_dir),
        max_memory_items=100,
        max_disk_items=50,
        expiration_hours=1,
        compression_level=6
    )
    await manager.initialize()
    
    yield manager
    
    await manager.cleanup()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

@pytest.mark.asyncio
async def test_data_compression(cache_manager):
    """Test data compression functionality."""
    # Crear datos de prueba grandes
    large_data = {
        "array": list(range(1000)),
        "text": "test" * 1000
    }
    
    # Guardar en caché
    await cache_manager.set("compression_test", large_data)
    
    # Verificar que está comprimido en disco
    cache_path = cache_manager._get_cache_path("compression_test")
    with open(cache_path, "rb") as f:
        compressed_data = f.read()
    
    # Verificar que los datos están comprimidos
    original_size = len(pickle.dumps(large_data))
    compressed_size = len(compressed_data)
    assert compressed_size < original_size
    
    # Verificar que podemos recuperar los datos correctamente
    retrieved_data = await cache_manager.get("compression_test")
    assert retrieved_data == large_data

@pytest.mark.asyncio
async def test_compression_levels(cache_manager):
    """Test different compression levels."""
    test_data = "test" * 1000
    
    # Probar diferentes niveles de compresión
    sizes = []
    for level in [1, 6, 9]:
        cache_manager.compression_level = level
        await cache_manager.set(f"compression_level_{level}", test_data)
        
        cache_path = cache_manager._get_cache_path(f"compression_level_{level}")
        sizes.append(os.path.getsize(cache_path))
    
    # Verificar que mayor nivel = mejor compresión
    assert sizes[0] > sizes[1] > sizes[2]

@pytest.mark.asyncio
async def test_cache_metrics(cache_manager):
    """Test cache metrics collection."""
    # Realizar operaciones de caché
    await cache_manager.set("metrics_test_1", "value1")
    await cache_manager.set("metrics_test_2", "value2")
    
    # Hits
    await cache_manager.get("metrics_test_1")
    await cache_manager.get("metrics_test_1")
    
    # Misses
    await cache_manager.get("nonexistent_key")
    
    # Verificar métricas
    stats = await cache_manager.get_stats()
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 1
    assert abs(stats["hit_ratio"] - 0.666) < 0.01

@pytest.mark.asyncio
async def test_memory_eviction(cache_manager):
    """Test memory cache eviction."""
    # Reducir límite de memoria
    cache_manager.max_memory_items = 5
    
    # Agregar items hasta forzar evicción
    for i in range(10):
        await cache_manager.set(f"eviction_test_{i}", f"value_{i}")
        await asyncio.sleep(0.1)  # Asegurar timestamps diferentes
    
    # Verificar que los items más antiguos fueron removidos
    memory_items = len(cache_manager.memory_cache)
    assert memory_items <= cache_manager.max_memory_items
    
    # Verificar que los items más recientes permanecen
    for i in range(5, 10):
        assert await cache_manager.get(f"eviction_test_{i}") is not None

@pytest.mark.asyncio
async def test_concurrent_access(cache_manager):
    """Test concurrent cache access."""
    # Crear múltiples operaciones concurrentes
    async def cache_operation(i: int):
        await cache_manager.set(f"concurrent_{i}", f"value_{i}")
        await cache_manager.get(f"concurrent_{i}")
    
    # Ejecutar operaciones concurrentemente
    tasks = [cache_operation(i) for i in range(50)]
    await asyncio.gather(*tasks)
    
    # Verificar resultados
    assert len(cache_manager.memory_cache) <= cache_manager.max_memory_items
    assert len(cache_manager.modified_keys) == 0  # Debería estar vacío después de guardar

@pytest.mark.asyncio
async def test_error_handling(cache_manager):
    """Test error handling in cache operations."""
    # Probar valor no serializable
    with pytest.raises(ProcessingError):
        await cache_manager.set("error_test", lambda x: x)
    
    # Probar archivo corrupto
    cache_path = cache_manager._get_cache_path("corrupt_test")
    cache_path.write_bytes(b"corrupted data")
    
    result = await cache_manager.get("corrupt_test")
    assert result is None

@pytest.mark.asyncio
async def test_periodic_cleanup(cache_manager):
    """Test periodic cleanup functionality."""
    # Configurar para expiración rápida
    cache_manager.expiration_hours = 0.001  # ~3.6 segundos
    
    # Agregar datos
    await cache_manager.set("cleanup_test", "test_value")
    
    # Esperar expiración
    await asyncio.sleep(4)
    
    # Forzar limpieza
    await cache_manager._periodic_cleanup()
    
    # Verificar que los datos fueron limpiados
    result = await cache_manager.get("cleanup_test")
    assert result is None

@pytest.mark.asyncio
async def test_disk_cleanup(cache_manager):
    """Test disk cache cleanup."""
    # Reducir límite de disco
    cache_manager.max_disk_items = 3
    
    # Agregar más items que el límite
    for i in range(5):
        await cache_manager.set(f"disk_test_{i}", f"value_{i}")
        # Forzar escritura a disco
        await cache_manager._periodic_save()
    
    # Forzar limpieza
    await cache_manager._cleanup_disk_cache()
    
    # Verificar límite de archivos
    cache_files = list(Path(cache_manager.cache_dir).glob("*.cache"))
    assert len(cache_files) <= cache_manager.max_disk_items

@pytest.mark.asyncio
async def test_status_reporting(cache_manager):
    """Test status reporting functionality."""
    # Realizar algunas operaciones
    await cache_manager.set("status_test", "test_value")
    await cache_manager.get("status_test")
    
    # Obtener estado
    status = await cache_manager.get_status()
    
    # Verificar campos requeridos
    assert "initialized" in status
    assert "active_tasks" in status
    assert "cache_stats" in status
    assert "system_metrics" in status
    
    # Verificar valores específicos
    assert status["initialized"] is True
    assert isinstance(status["system_metrics"], PerformanceMetrics)

@pytest.mark.asyncio
async def test_recovery_from_interrupted_save(cache_manager):
    """Test recovery from interrupted save operation."""
    # Simular interrupción durante guardado
    async def mock_write_file(*args):
        raise ProcessingError("Write interrupted")
    
    with patch.object(cache_manager, '_write_file', side_effect=mock_write_file):
        # Intentar guardar
        await cache_manager.set("recovery_test", "test_value")
        
        # Verificar que está en memoria a pesar del error
        assert "recovery_test" in cache_manager.memory_cache
        
        # Verificar que está marcado para reintento
        assert "recovery_test" in cache_manager.modified_keys

@pytest.mark.asyncio
async def test_compression_error_handling(cache_manager):
    """Test handling of compression errors."""
    # Simular error de compresión
    def mock_compress(*args):
        raise zlib.error("Compression failed")
    
    with patch('zlib.compress', side_effect=mock_compress):
        # El sistema debería manejar el error y guardar sin comprimir
        await cache_manager.set("compression_error_test", "test_value")
        
        # Verificar que los datos se guardaron
        result = await cache_manager.get("compression_error_test")
        assert result == "test_value"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.cache"])