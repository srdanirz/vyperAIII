import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from prometheus_client import CollectorRegistry
import os
import psutil

from monitoring.monitoring_manager import MonitoringManager, MetricValidation
from core.interfaces import ResourceUsage

@pytest.fixture
async def monitoring_manager():
    manager = MonitoringManager()
    yield manager
    await manager.cleanup()

@pytest.fixture
def mock_prometheus():
    with patch('prometheus_client.push_to_gateway') as mock:
        yield mock

@pytest.fixture
def mock_aiohttp_session():
    async def mock_post(*args, **kwargs):
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        return mock_response

    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post = mock_post
        yield mock_session

@pytest.mark.asyncio
async def test_metric_validation(monitoring_manager):
    """Test metric validation rules."""
    # Valid metrics
    assert monitoring_manager._validate_metric("system_load", 50.0)
    assert monitoring_manager._validate_metric("error_rate", 0.5)
    
    # Invalid metrics
    assert not monitoring_manager._validate_metric("system_load", -1.0)
    assert not monitoring_manager._validate_metric("system_load", 101.0)
    assert not monitoring_manager._validate_metric("error_rate", 1.5)
    
    # Invalid types
    assert not monitoring_manager._validate_metric("system_load", "50")
    
    # Custom validation
    monitoring_manager.METRIC_VALIDATORS["custom"] = MetricValidation(
        custom_validator=lambda x: x % 2 == 0
    )
    assert monitoring_manager._validate_metric("custom", 2)
    assert not monitoring_manager._validate_metric("custom", 3)

@pytest.mark.asyncio
async def test_record_metric_thread_safety(monitoring_manager):
    """Test thread-safe metric recording."""
    # Create multiple concurrent metric recordings
    metric_name = "test_metric"
    tasks = [
        monitoring_manager.record_metric(metric_name, i)
        for i in range(100)
    ]
    
    await asyncio.gather(*tasks)
    
    # Verify all metrics were recorded correctly
    assert len(monitoring_manager.metrics_cache[metric_name]) == 100
    assert sorted(monitoring_manager.metrics_cache[metric_name]) == list(range(100))

@pytest.mark.asyncio
async def test_alert_generation_and_notification(
    monitoring_manager,
    mock_aiohttp_session
):
    """Test alert creation and notification system."""
    # Configure test environment
    os.environ["SLACK_WEBHOOK_URL"] = "http://test.webhook"
    os.environ["ALERT_EMAIL"] = "test@example.com"
    
    # Trigger alert
    await monitoring_manager.record_metric("error_rate", 0.15)
    
    # Verify alert was created
    assert len(monitoring_manager.active_alerts) == 1
    alert = list(monitoring_manager.active_alerts.values())[0]
    assert alert["severity"] == "warning"
    assert "error_rate" in alert["title"]
    
    # Verify notifications were sent
    mock_aiohttp_session.assert_called_once()

@pytest.mark.asyncio
async def test_system_metrics_collection(monitoring_manager):
    """Test system metrics collection."""
    with patch('psutil.cpu_percent') as mock_cpu, \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('psutil.disk_usage') as mock_disk:
        
        # Configure mocks
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(
            used=1000,
            available=9000,
            percent=10.0
        )
        mock_disk.return_value = Mock(percent=20.0)
        
        # Run one cycle of monitoring
        await monitoring_manager._monitor_system_metrics()
        
        # Verify metrics were recorded
        assert monitoring_manager.metrics_cache["cpu_usage"][-1] == 50.0
        assert monitoring_manager.metrics_cache["memory_usage"][-1] == 1000
        assert monitoring_manager.resource_usage.cpu_percent == 50.0
        assert monitoring_manager.resource_usage.memory_percent == 10.0
        assert monitoring_manager.resource_usage.disk_usage_percent == 20.0

@pytest.mark.asyncio
async def test_alert_lifecycle(monitoring_manager):
    """Test complete alert lifecycle."""
    # Create alert
    alert_data = {
        "severity": "warning",
        "title": "Test Alert",
        "description": "Test Description",
        "metric_data": {"value": 42}
    }
    
    await monitoring_manager._create_alert(**alert_data)
    
    # Verify alert was created
    assert len(monitoring_manager.active_alerts) == 1
    assert len(monitoring_manager.alert_history) == 1
    
    # Age the alert
    alert_id = list(monitoring_manager.active_alerts.keys())[0]
    alert = monitoring_manager.active_alerts[alert_id]
    alert["timestamp"] = (datetime.now() - timedelta(hours=2)).isoformat()
    
    # Run cleanup
    monitoring_manager._cleanup_old_alerts()
    
    # Verify alert was resolved
    assert len(monitoring_manager.active_alerts) == 0
    assert monitoring_manager.alert_history[0]["status"] == "resolved"

@pytest.mark.asyncio
async def test_prometheus_integration(monitoring_manager, mock_prometheus):
    """Test Prometheus metrics integration."""
    # Configure test environment
    os.environ["PROMETHEUS_PUSHGATEWAY"] = "localhost:9091"
    
    # Record metric
    await monitoring_manager.record_metric("test_metric", 42.0)
    
    # Verify metric was pushed to Prometheus
    mock_prometheus.assert_called_once()
    
    # Verify registry content
    assert isinstance(monitoring_manager.registry, CollectorRegistry)

@pytest.mark.asyncio
async def test_metric_cache_management(monitoring_manager):
    """Test metric cache size management."""
    metric_name = "test_metric"
    
    # Fill cache beyond limit
    for i in range(2000):
        await monitoring_manager.record_metric(metric_name, i)
    
    # Verify cache was trimmed
    assert len(monitoring_manager.metrics_cache[metric_name]) == 1000
    assert monitoring_manager.metrics_cache[metric_name][-1] == 1999

@pytest.mark.asyncio
async def test_cleanup(monitoring_manager):
    """Test cleanup process."""
    # Record some data
    await monitoring_manager.record_metric("test_metric", 42.0)
    await monitoring_manager._create_alert(
        severity="warning",
        title="Test",
        description="Test",
        metric_data={}
    )
    
    # Perform cleanup
    await monitoring_manager.cleanup()
    
    # Verify everything was cleaned up
    assert len(monitoring_manager.metrics_cache) == 0
    assert len(monitoring_manager.active_alerts) == 0
    assert monitoring_manager._should_stop
    assert all(task.cancelled() for task in monitoring_manager._monitoring_tasks)

@pytest.mark.asyncio
async def test_concurrent_alert_management(monitoring_manager):
    """Test concurrent alert management."""
    # Create multiple alerts concurrently
    alert_tasks = [
        monitoring_manager._create_alert(
            severity="warning",
            title=f"Test Alert {i}",
            description="Test",
            metric_data={}
        )
        for i in range(10)
    ]
    
    await asyncio.gather(*alert_tasks)
    
    # Verify all alerts were created correctly
    assert len(monitoring_manager.active_alerts) == 10
    assert len(monitoring_manager.alert_history) == 10
    
    # Verify no race conditions in alert IDs
    alert_ids = set(monitoring_manager.active_alerts.keys())
    assert len(alert_ids) == 10

@pytest.mark.asyncio
async def test_status_reporting(monitoring_manager):
    """Test status reporting functionality."""
    # Record some test data
    await monitoring_manager.record_metric("test_metric", 42.0)
    await monitoring_manager._create_alert(
        severity="warning",
        title="Test",
        description="Test",
        metric_data={}
    )
    
    # Get status
    status = await monitoring_manager.get_status()
    
    # Verify status content
    assert "active_metrics" in status
    assert "active_alerts" in status
    assert "metrics_cache_size" in status
    assert "resource_usage" in status
    assert "monitoring_tasks" in status
    
    # Verify status accuracy
    assert status["active_alerts"] == 1
    assert "test_metric" in status["metrics_cache_size"]
    assert isinstance(status["resource_usage"], dict)
    assert status["monitoring_tasks"]["total"] == 2