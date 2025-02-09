import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from monitoring.monitoring_manager import MonitoringManager

@pytest.fixture
async def monitoring_manager():
    manager = MonitoringManager()
    yield manager
    await manager.cleanup()

@pytest.mark.asyncio
async def test_record_metric(monitoring_manager):
    # Test registrar una métrica
    metric_name = "test_metric"
    value = 42.0
    
    await monitoring_manager.record_metric(metric_name, value)
    
    # Verificar que la métrica se guardó
    assert metric_name in monitoring_manager.metrics_cache
    assert monitoring_manager.metrics_cache[metric_name][-1] == value

@pytest.mark.asyncio
async def test_alert_generation(monitoring_manager):
    # Test generación de alertas
    metric_name = "error_rate"
    value = 0.15  # Por encima del umbral de 0.1
    
    await monitoring_manager.record_metric(metric_name, value)
    
    # Verificar que se generó una alerta
    assert len(monitoring_manager.active_alerts) > 0
    alert = list(monitoring_manager.active_alerts.values())[0]
    assert alert["severity"] == "warning"
    assert metric_name in alert["description"]

@pytest.mark.asyncio
async def test_metrics_report(monitoring_manager):
    # Preparar datos de prueba
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    
    # Registrar algunas métricas
    await monitoring_manager.record_metric("test_metric", 1.0)
    await monitoring_manager.record_metric("test_metric", 2.0)
    
    # Obtener reporte
    report = await monitoring_manager.get_metrics_report(start_time, end_time)
    
    # Verificar estructura del reporte
    assert "period" in report
    assert "system" in report
    assert "performance" in report
    assert "alerts" in report

@pytest.mark.asyncio
async def test_monitoring_tasks(monitoring_manager):
    # Verificar que las tareas de monitoreo están activas
    assert len(monitoring_manager._monitoring_tasks) == 2
    
    # Verificar que las tareas están corriendo
    for task in monitoring_manager._monitoring_tasks:
        assert not task.done()
    
    # Cleanup debería cancelar las tareas
    await monitoring_manager.cleanup()
    
    # Verificar que las tareas fueron canceladas
    for task in monitoring_manager._monitoring_tasks:
        assert task.cancelled() or task.done()

@pytest.mark.asyncio
async def test_system_metrics_collection():
    with patch('psutil.Process') as mock_process:
        # Configurar mock
        mock_process.return_value.memory_info.return_value.rss = 1024
        
        manager = MonitoringManager()
        
        # Esperar un ciclo de recolección
        await asyncio.sleep(1)
        
        # Verificar que se llamó a memory_info
        mock_process.return_value.memory_info.assert_called()
        
        await manager.cleanup()

@pytest.mark.asyncio
async def test_alert_lifecycle(monitoring_manager):
    # Crear una alerta
    alert_data = {
        "severity": "warning",
        "title": "Test Alert",
        "description": "Test Description",
        "metric_data": {"value": 42}
    }
    
    await monitoring_manager._send_alert(**alert_data)
    
    # Verificar que la alerta está activa
    assert len(monitoring_manager.active_alerts) == 1
    assert len(monitoring_manager.alert_history) == 1
    
    # Esperar a que la alerta expire
    alert_id = list(monitoring_manager.active_alerts.keys())[0]
    alert = monitoring_manager.active_alerts[alert_id]
    alert["timestamp"] = (datetime.now() - timedelta(hours=2)).isoformat()
    
    # Ejecutar ciclo de monitoreo
    await monitoring_manager._monitor_active_alerts()
    
    # Verificar que la alerta fue resuelta
    assert len(monitoring_manager.active_alerts) == 0

@pytest.mark.asyncio
async def test_metrics_calculations(monitoring_manager):
    metric_name = "test_calc"
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Registrar valores
    for value in values:
        await monitoring_manager.record_metric(metric_name, value)
    
    # Verificar cálculos
    assert monitoring_manager._calculate_average(metric_name) == 3.0
    assert monitoring_manager._calculate_peak(metric_name) == 5.0
    assert abs(monitoring_manager._calculate_percentile(metric_name, 95) - 5.0) < 0.1

@pytest.mark.asyncio
async def test_status(monitoring_manager):
    status = await monitoring_manager.get_status()
    
    assert "active_metrics" in status
    assert "active_alerts" in status
    assert "metrics_cache_size" in status
    assert "performance" in status
    assert "resources" in status