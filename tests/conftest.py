import pytest
import asyncio
import os
from pathlib import Path
from typing import Generator, AsyncGenerator, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
from datetime import datetime

from core.interfaces import (
    PerformanceMetrics,
    ResourceUsage,
    Team,
    TeamMember,
    RequestContext,
    ProcessingMode,
    Priority
)

# Configuración de ambiente de testing
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the test directory path."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def data_dir(test_dir: Path) -> Path:
    """Return the test data directory path."""
    data_dir = test_dir / "data" / "test_files"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test."""
    # Setup
    os.environ["TESTING"] = "true"
    os.environ["ENVIRONMENT"] = "testing"
    
    yield
    
    # Teardown
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "ENVIRONMENT" in os.environ:
        del os.environ["ENVIRONMENT"]

# Fixtures específicos para testing del sistema edge
@pytest.fixture
def mock_edge_node():
    """Mock para nodo edge."""
    return {
        "node_id": "test_node",
        "capabilities": ["compute", "storage"],
        "resources": ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage_percent=70.0,
            network_bytes_sent=1000,
            network_bytes_received=2000,
            active_threads=5
        ),
        "status": "ready",
        "metrics": PerformanceMetrics()
    }

@pytest.fixture
def mock_edge_task():
    """Mock para tarea edge."""
    return {
        "task_id": "test_task",
        "type": "processing",
        "data": {"input": "test_data"},
        "required_capabilities": ["compute"],
        "priority": Priority.MEDIUM,
        "submitted_at": datetime.now().isoformat()
    }

# Mock fixtures comunes
@pytest.fixture
def mock_llm() -> Mock:
    """Mock para modelos de lenguaje."""
    mock = Mock()
    mock.agenerate = AsyncMock(return_value=Mock(
        generations=[[Mock(message=Mock(content="Test response"))]]
    ))
    return mock

@pytest.fixture
def mock_cache() -> Mock:
    """Mock para sistema de caché."""
    mock = Mock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_storage() -> Mock:
    """Mock para sistema de almacenamiento."""
    mock = Mock()
    mock.save = AsyncMock(return_value="file_path")
    mock.load = AsyncMock(return_value=b"file_content")
    mock.delete = AsyncMock(return_value=True)
    return mock

@pytest.fixture
def mock_monitoring() -> Mock:
    """Mock para sistema de monitoreo."""
    mock = Mock()
    mock.record_metric = AsyncMock()
    mock.get_metrics = AsyncMock(return_value={})
    mock.create_alert = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_plugin_manager() -> Mock:
    """Mock para gestor de plugins."""
    mock = Mock()
    mock.load_plugin = AsyncMock(return_value=True)
    mock.execute_hook = AsyncMock(return_value=[])
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_team() -> Team:
    """Mock para equipo de trabajo."""
    return Team(
        id="test_team",
        name="Test Team",
        members={
            "member1": TeamMember(
                id="member1",
                role="analyst",
                capabilities=["analysis", "research"]
            )
        },
        objectives=["test_objective"],
        created_at=datetime.now()
    )

@pytest.fixture
def mock_request_context() -> RequestContext:
    """Mock para contexto de solicitud."""
    return RequestContext(
        request_id="test_request",
        timestamp=datetime.now(),
        mode=ProcessingMode.STANDARD,
        priority=Priority.MEDIUM,
        metadata={"test": True}
    )

@pytest.fixture
def mock_aiohttp_session():
    """Mock para sesiones HTTP asíncronas."""
    async def mock_post(*args, **kwargs):
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        return mock_response

    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post = mock_post
        mock_session.return_value.__aenter__.return_value.get = mock_post
        yield mock_session

@pytest.fixture
def mock_performance_metrics() -> PerformanceMetrics:
    """Mock para métricas de rendimiento."""
    return PerformanceMetrics(
        resource_usage=ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage_percent=70.0,
            network_bytes_sent=1000,
            network_bytes_received=2000,
            active_threads=5
        )
    )