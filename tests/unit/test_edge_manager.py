import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from typing import Dict, Any, List, Optional

from edge.edge_manager import EdgeManager, EdgeNode
from core.interfaces import ResourceUsage, PerformanceMetrics

@pytest.fixture
async def edge_manager():
    """Create a test instance of EdgeManager."""
    manager = EdgeManager()
    yield manager
    await manager.cleanup()

@pytest.fixture
def mock_node():
    """Create a mock edge node."""
    return EdgeNode(
        node_id="test_node",
        capabilities={"compute", "storage"},
        resources=ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage_percent=70.0
        ),
        status="ready",
        load=0.5,
        metrics=PerformanceMetrics()
    )

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
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
async def test_register_node(edge_manager):
    """Test node registration."""
    success = await edge_manager.register_node(
        node_id="test_node",
        capabilities=["compute", "storage"],
        resources={
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_usage_percent": 70.0
        }
    )
    
    assert success
    assert "test_node" in edge_manager.nodes
    assert len(edge_manager.node_groups["compute"]) == 1
    assert len(edge_manager.node_groups["storage"]) == 1
    assert edge_manager.performance_metrics.total_requests == 1

@pytest.mark.asyncio
async def test_node_validation(edge_manager):
    """Test node validation during registration."""
    # Test with invalid resources
    invalid_success = await edge_manager.register_node(
        node_id="invalid_node",
        capabilities=["compute"],
        resources={
            "cpu_percent": 5.0,  # Below minimum
            "memory_percent": 5.0,  # Below minimum
            "disk_usage_percent": 5.0  # Below minimum
        }
    )
    
    assert not invalid_success
    assert "invalid_node" not in edge_manager.nodes

@pytest.mark.asyncio
async def test_task_submission(edge_manager, mock_node):
    """Test task submission and processing."""
    # Register test node
    edge_manager.nodes[mock_node.node_id] = mock_node
    
    # Submit task
    result = await edge_manager.submit_task(
        task={"type": "test_task", "data": "test_data"},
        required_capabilities=["compute"]
    )
    
    assert result["status"] == "success"
    assert "task_id" in result
    assert mock_node.node_id in result["node_id"]

@pytest.mark.asyncio
async def test_optimal_node_selection(edge_manager):
    """Test selection of optimal node for task."""
    # Register multiple nodes
    nodes = []
    for i in range(3):
        node = EdgeNode(
            node_id=f"node_{i}",
            capabilities={"compute", "storage"},
            resources=ResourceUsage(
                cpu_percent=50.0 + i*10,
                memory_percent=60.0 + i*10,
                disk_usage_percent=70.0 + i*10
            ),
            status="ready",
            load=0.2 * i,
            metrics=PerformanceMetrics()
        )
        edge_manager.nodes[node.node_id] = node
        nodes.append(node)
    
    # Select optimal node
    selected_node = await edge_manager._select_optimal_node(["compute"])
    
    assert selected_node is not None
    assert selected_node.load == min(node.load for node in nodes)

@pytest.mark.asyncio
async def test_node_failure_handling(edge_manager, mock_node):
    """Test handling of node failures."""
    # Register node
    edge_manager.nodes[mock_node.node_id] = mock_node
    
    # Submit task
    task_result = await edge_manager.submit_task(
        task={"type": "test_task", "data": "test_data"},
        required_capabilities=["compute"]
    )
    
    # Simulate node failure
    await edge_manager._handle_node_failure(mock_node.node_id)
    
    assert mock_node.node_id not in edge_manager.nodes
    assert len(edge_manager.active_tasks) == 0

@pytest.mark.asyncio
async def test_task_monitoring(edge_manager, mock_node):
    """Test task status monitoring."""
    # Register node and submit task
    edge_manager.nodes[mock_node.node_id] = mock_node
    task_result = await edge_manager.submit_task(
        task={"type": "test_task", "data": "test_data"},
        required_capabilities=["compute"]
    )
    
    # Get task status
    status = await edge_manager.get_task_status(task_result["task_id"])
    
    assert status is not None
    assert "status" in status
    assert "node_id" in status

@pytest.mark.asyncio
async def test_load_balancing(edge_manager):
    """Test load balancing between nodes."""
    # Register multiple nodes
    for i in range(3):
        await edge_manager.register_node(
            node_id=f"node_{i}",
            capabilities=["compute"],
            resources={
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "disk_usage_percent": 70.0
            }
        )
    
    # Submit multiple tasks
    tasks = []
    for i in range(5):
        task = edge_manager.submit_task(
            task={"type": "test_task", "id": i},
            required_capabilities=["compute"]
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Verify load distribution
    node_tasks = {}
    for result in results:
        node_id = result["node_id"]
        node_tasks[node_id] = node_tasks.get(node_id, 0) + 1
    
    max_tasks = max(node_tasks.values())
    min_tasks = min(node_tasks.values())
    assert max_tasks - min_tasks <= 1  # Load should be balanced

@pytest.mark.asyncio
async def test_resource_monitoring(edge_manager, mock_node):
    """Test resource monitoring of nodes."""
    # Register node
    edge_manager.nodes[mock_node.node_id] = mock_node
    
    # Monitor for one cycle
    await edge_manager._monitor_nodes()
    
    # Verify resource monitoring
    assert edge_manager.nodes[mock_node.node_id].resources.cpu_percent is not None
    assert edge_manager.nodes[mock_node.node_id].resources.memory_percent is not None
    assert edge_manager.nodes[mock_node.node_id].resources.disk_usage_percent is not None

@pytest.mark.asyncio
async def test_cleanup(edge_manager, mock_node):
    """Test cleanup process."""
    # Register node and submit task
    edge_manager.nodes[mock_node.node_id] = mock_node
    await edge_manager.submit_task(
        task={"type": "test_task", "data": "test_data"},
        required_capabilities=["compute"]
    )
    
    # Perform cleanup
    await edge_manager.cleanup()
    
    assert len(edge_manager.nodes) == 0
    assert len(edge_manager.active_tasks) == 0
    assert len(edge_manager.node_groups) == 0

@pytest.mark.asyncio
async def test_health_status(edge_manager):
    """Test health status reporting."""
    # Register a healthy node
    await edge_manager.register_node(
        node_id="healthy_node",
        capabilities=["compute"],
        resources={
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_usage_percent": 70.0
        }
    )
    
    # Get health status
    status = await edge_manager.get_health_status()
    
    assert status["status"] == "healthy"
    assert status["nodes"]["total"] == 1
    assert status["nodes"]["active"] == 1
    assert status["nodes"]["failed"] == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=edge.edge_manager"])