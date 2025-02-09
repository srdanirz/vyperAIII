import pytest
import asyncio
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import hashlib

from core.cache import CacheManager
from core.errors import ProcessingError
from core.interfaces import PerformanceMetrics, ResourceUsage

@pytest.fixture
async def cache_manager():
    """Create test cache manager instance."""
    manager = CacheManager(cache_dir="test_cache")
    await manager.initialize()
    yield manager
    await manager.cleanup()
    # Cleanup test directory
    if Path("test_cache").exists():
        for file in Path("test_cache").glob("*"):
            file.unlink()
        Path("test_cache").rmdir()

@pytest.fixture
def mock_executor():
    """Mock ThreadPoolExecutor."""
    mock = Mock()
    mock.submit = Mock()
    mock.shutdown = Mock()
    return mock

@pytest.mark.asyncio
async def test_cache_operations(cache_manager):
    """Test basic cache operations."""
    # Test set and get
    key = "test_key"
    value = {"data": "test_value"}
    
    await cache_manager.set(key, value)
    result = await cache_manager.get(key)
    
    assert result == value
    
    # Test non-existent key
    result = await cache_manager.get("nonexistent")
    assert result is None
    
    # Test overwrite
    new_value = {"data": "new_value"}
    await cache_manager.set(key, new_value)
    result = await cache_manager.get(key)
    assert result == new_value

@pytest.mark.asyncio
async def test_cache_persistence(cache_manager):
    """Test cache persistence to disk."""
    key = "persist_test"
    value = {"data": "persist_value"}
    
    # Save to cache
    await cache_manager.set(key, value)
    
    # Create new cache manager (simulating restart)
    new_manager = CacheManager(cache_dir="test_cache")
    await new_manager.initialize()
    
    # Verify data persisted
    result = await new_manager.get(key)
    assert result == value
    
    await new_manager.cleanup()

@pytest.mark.asyncio
async def test_cache_expiration(cache_manager):
    """Test cache item expiration."""
    key = "expiring_key"
    value = {"data": "expiring_value"}
    ttl = 1  # 1 second TTL
    
    await cache_manager.set(key, value, ttl=ttl)
    
    # Verify value exists
    result = await cache_manager.get(key)
    assert result == value
    
    # Wait for expiration
    await asyncio.sleep(ttl + 0.1)
    
    # Verify value expired
    result = await cache_manager.get(key)
    assert result is None

@pytest.mark.asyncio
async def test_memory_cache_eviction(cache_manager):
    """Test memory cache eviction when limit reached."""
    original_max_items = cache_manager.max_memory_items
    cache_manager.max_memory_items = 2
    
    # Add items to trigger eviction
    await cache_manager.set("key1", "value1")
    await cache_manager.set("key2", "value2")
    await cache_manager.set("key3", "value3")
    
    # Verify oldest item was evicted
    assert await cache_manager.get("key1") is None
    assert await cache_manager.get("key2") is not None
    assert await cache_manager.get("key3") is not None
    
    # Restore original limit
    cache_manager.max_memory_items = original_max_items

@pytest.mark.asyncio
async def test_concurrent_access(cache_manager):
    """Test concurrent cache access."""
    key = "concurrent_key"
    iterations = 100
    
    # Create concurrent set operations
    async def set_value(i):
        await cache_manager.set(f"{key}_{i}", f"value_{i}")
    
    # Execute concurrent operations
    tasks = [set_value(i) for i in range(iterations)]
    await asyncio.gather(*tasks)
    
    # Verify all values were set correctly
    results = await asyncio.gather(*[
        cache_manager.get(f"{key}_{i}")
        for i in range(iterations)
    ])
    
    assert all(results[i] == f"value_{i}" for i in range(iterations))

@pytest.mark.asyncio
async def test_error_handling(cache_manager):
    """Test error handling in cache operations."""
    # Test invalid value serialization
    with pytest.raises(ProcessingError):
        await cache_manager.set("invalid_key", lambda x: x)  # Functions can't be pickled
    
    # Test corrupted cache file
    cache_path = Path("test_cache") / "corrupted.cache"
    cache_path.write_bytes(b"corrupted data")
    
    result = await cache_manager._read_cache_file(cache_path)
    assert result is None

@pytest.mark.asyncio
async def test_cache_metrics(cache_manager):
    """Test cache metrics collection."""
    await cache_manager.set("metric_key1", "value1")
    await cache_manager.get("metric_key1")  # Hit
    await cache_manager.get("nonexistent")  # Miss
    
    # Verify metrics
    stats = await cache_manager.get_stats()
    assert stats["cache_hits"] > 0
    assert stats["cache_misses"] > 0
    assert "hit_ratio" in stats

@pytest.mark.asyncio
async def test_periodic_cleanup(cache_manager):
    """Test periodic cleanup of expired items."""
    # Add items with short TTL
    for i in range(5):
        await cache_manager.set(f"cleanup_key_{i}", f"value_{i}", ttl=1)
    
    # Wait for items to expire
    await asyncio.sleep(1.1)
    
    # Trigger cleanup
    await cache_manager._cleanup_expired()
    
    # Verify items were cleaned up
    for i in range(5):
        result = await cache_manager.get(f"cleanup_key_{i}")
        assert result is None

@pytest.mark.asyncio
async def test_disk_cache_management(cache_manager):
    """Test disk cache size management."""
    original_max_items = cache_manager.max_disk_items
    cache_manager.max_disk_items = 2
    
    # Add items to trigger disk cleanup
    for i in range(3):
        await cache_manager.set(f"disk_key_{i}", f"value_{i}")
        # Force disk write
        await cache_manager._save_to_disk()
    
    # Verify disk cache size is maintained
    cache_files = list(Path("test_cache").glob("*.cache"))
    assert len(cache_files) <= cache_manager.max_disk_items
    
    # Restore original limit
    cache_manager.max_disk_items = original_max_items

@pytest.mark.asyncio
async def test_cache_compression(cache_manager):
    """Test cache data compression."""
    # Large data to trigger compression
    large_data = {"data": "x" * 1000000}
    
    await cache_manager.set("large_key", large_data)
    
    # Verify data was compressed
    cache_file = next(Path("test_cache").glob("*.cache"))
    assert cache_file.stat().st_size < len(pickle.dumps(large_data))
    
    # Verify retrieval works
    result = await cache_manager.get("large_key")
    assert result == large_data

@pytest.mark.asyncio
async def test_cache_validation(cache_manager):
    """Test cache data validation."""
    # Valid data
    valid_data = {"key": "value"}
    await cache_manager.set("valid_key", valid_data)
    
    # Invalid data types
    invalid_data = [
        lambda x: x,  # Function
        object(),    # Custom object
        asyncio.Lock()  # Non-picklable object
    ]
    
    for data in invalid_data:
        with pytest.raises(ProcessingError):
            await cache_manager.set("invalid_key", data)

@pytest.mark.asyncio
async def test_cache_initialization(cache_manager):
    """Test cache initialization process."""
    # Add some test data
    await cache_manager.set("init_key", "init_value")
    
    # Create new cache manager to test initialization
    new_manager = CacheManager(cache_dir="test_cache")
    await new_manager.initialize()
    
    # Verify cache was loaded correctly
    assert await new_manager.get("init_key") == "init_value"
    assert new_manager._initialized
    
    await new_manager.cleanup()

@pytest.mark.asyncio
async def test_cache_performance(cache_manager):
    """Test cache performance metrics."""
    start_time = datetime.now()
    
    # Perform multiple operations
    for i in range(100):
        await cache_manager.set(f"perf_key_{i}", f"value_{i}")
        await cache_manager.get(f"perf_key_{i}")
    
    stats = await cache_manager.get_stats()
    
    # Verify performance metrics
    assert "performance_metrics" in stats
    assert stats["performance_metrics"]["hit_ratio"] > 0
    assert stats["performance_metrics"]["total_requests"] >= 200

@pytest.mark.asyncio
async def test_resource_cleanup(cache_manager):
    """Test resource cleanup on shutdown."""
    # Add some data and tasks
    await cache_manager.set("cleanup_key", "cleanup_value")
    
    # Perform cleanup
    await cache_manager.cleanup()
    
    # Verify resources were cleaned up
    assert len(cache_manager.memory_cache) == 0
    assert not list(Path("test_cache").glob("*.cache"))
    assert cache_manager._save_task.cancelled()
    assert cache_manager._cleanup_task.cancelled()

@pytest.mark.asyncio
async def test_cache_recovery(cache_manager):
    """Test cache recovery from corrupted state."""
    # Simulate corrupted cache file
    corrupted_path = Path("test_cache") / "corrupted.cache"
    corrupted_path.write_bytes(b"corrupted data")
    
    # Initialize new cache manager
    recovery_manager = CacheManager(cache_dir="test_cache")
    await recovery_manager.initialize()
    
    # Verify manager initialized successfully despite corruption
    assert recovery_manager._initialized
    
    # Add new data
    await recovery_manager.set("recovery_key", "recovery_value")
    result = await recovery_manager.get("recovery_key")
    assert result == "recovery_value"
    
    await recovery_manager.cleanup()

@pytest.mark.asyncio
async def test_status_reporting(cache_manager):
    """Test status reporting functionality."""
    # Add some test data
    await cache_manager.set("status_key", "status_value")
    
    # Get status
    status = await cache_manager.get_status()
    
    # Verify status content
    assert "memory_usage" in status
    assert "disk_usage" in status
    assert "metrics" in status
    assert "health_status" in status

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.cache"])