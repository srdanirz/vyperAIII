import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from core.orchestrator import CoreOrchestrator
from core.errors import ProcessingError
from core.interfaces import (
    RequestContext,
    ProcessingMode,
    EngineMode,
    Priority,
    Team
)

@pytest.fixture
async def orchestrator():
    """Create a test orchestrator instance."""
    orchestrator = CoreOrchestrator(
        api_key="test_key",
        engine_mode="openai"
    )
    yield orchestrator
    await orchestrator.cleanup()

@pytest.fixture
def mock_team_manager():
    """Mock for TeamManager."""
    mock = Mock()
    mock.create_team = AsyncMock()
    mock.assign_task = AsyncMock()
    mock.get_team_status = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_agent_manager():
    """Mock for AgentManager."""
    mock = Mock()
    mock.execute = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_cache():
    """Mock for CacheManager."""
    mock = Mock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.mark.asyncio
async def test_process_request_success(
    orchestrator,
    mock_team_manager,
    mock_agent_manager,
    mock_cache
):
    """Test successful request processing."""
    # Setup mocks
    orchestrator.team_manager = mock_team_manager
    orchestrator.agent_manager = mock_agent_manager
    orchestrator.cache = mock_cache
    
    test_request = {
        "type": "analysis",
        "content": "Test content",
        "metadata": {"priority": "high"}
    }
    
    # Configure mock responses
    mock_team = Team(
        id="team1",
        name="test_team",
        members={},
        objectives=["analyze"]
    )
    mock_team_manager.create_team.return_value = mock_team
    mock_agent_manager.execute.return_value = {
        "result": "Test result",
        "status": "success"
    }
    
    # Process request
    result = await orchestrator.process_request(test_request)
    
    # Verify results
    assert result["status"] == "success"
    assert "results" in result
    assert "teams_involved" in result
    assert mock_team_manager.create_team.called
    assert mock_agent_manager.execute.called
    assert mock_cache.set.called

@pytest.mark.asyncio
async def test_process_request_cached(orchestrator, mock_cache):
    """Test request processing with cached result."""
    orchestrator.cache = mock_cache
    
    # Configure cache hit
    cached_result = {
        "status": "success",
        "result": "Cached result"
    }
    mock_cache.get.return_value = cached_result
    
    test_request = {
        "type": "analysis",
        "content": "Test content"
    }
    
    result = await orchestrator.process_request(test_request)
    
    assert result == cached_result
    assert not mock_cache.set.called

@pytest.mark.asyncio
async def test_process_request_error_handling(
    orchestrator,
    mock_team_manager,
    mock_agent_manager
):
    """Test error handling during request processing."""
    orchestrator.team_manager = mock_team_manager
    orchestrator.agent_manager = mock_agent_manager
    
    # Configure mock to raise error
    mock_team_manager.create_team.side_effect = ProcessingError("Team creation failed")
    
    test_request = {
        "type": "analysis",
        "content": "Test content"
    }
    
    result = await orchestrator.process_request(test_request)
    
    assert "error" in result
    assert result["error"]["message"] == "Team creation failed"
    assert orchestrator.system_state["error_count"] > 0

@pytest.mark.asyncio
async def test_team_coordination(orchestrator, mock_team_manager):
    """Test team coordination and task distribution."""
    orchestrator.team_manager = mock_team_manager
    
    # Configure mock teams
    teams = [
        Team(id=f"team{i}", name=f"Team {i}", members={}, objectives=["test"])
        for i in range(3)
    ]
    mock_team_manager.create_team.side_effect = teams
    
    test_request = {
        "type": "complex_analysis",
        "content": "Test content requiring multiple teams"
    }
    
    await orchestrator.process_request(test_request)
    
    # Verify team creation and coordination
    assert mock_team_manager.create_team.call_count == len(teams)
    assert mock_team_manager.assign_task.call_count >= len(teams)

@pytest.mark.asyncio
async def test_resource_management(orchestrator):
    """Test resource management and cleanup."""
    # Create some test resources
    orchestrator.active_teams = {"team1": Mock(), "team2": Mock()}
    orchestrator.pending_tasks = ["task1", "task2"]
    
    # Perform cleanup
    await orchestrator.cleanup()
    
    # Verify resources were cleaned up
    assert len(orchestrator.active_teams) == 0
    assert len(orchestrator.pending_tasks) == 0
    assert orchestrator.system_state["status"] == "cleaned"

@pytest.mark.asyncio
async def test_priority_handling(orchestrator, mock_team_manager):
    """Test handling of request priorities."""
    orchestrator.team_manager = mock_team_manager
    
    # Create requests with different priorities
    requests = [
        {
            "type": "analysis",
            "content": f"Test content {i}",
            "metadata": {"priority": priority}
        }
        for i, priority in enumerate(["low", "high", "medium"])
    ]
    
    # Process requests concurrently
    tasks = [orchestrator.process_request(req) for req in requests]
    results = await asyncio.gather(*tasks)
    
    # Verify priority handling
    task_assignments = mock_team_manager.assign_task.call_args_list
    assert len(task_assignments) == len(requests)
    priorities = [call.kwargs.get("priority", Priority.MEDIUM) for call in task_assignments]
    assert priorities.count(Priority.HIGH) == 1

@pytest.mark.asyncio
async def test_engine_mode_switching(orchestrator):
    """Test switching between different engine modes."""
    # Test OpenAI mode
    result_openai = await orchestrator.process_request({
        "type": "analysis",
        "content": "Test content",
        "metadata": {"engine_mode": "openai"}
    })
    assert orchestrator.engine_mode == EngineMode.OPENAI
    
    # Test DeepSeek mode
    result_deepseek = await orchestrator.process_request({
        "type": "analysis",
        "content": "Test content",
        "metadata": {"engine_mode": "deepseek"}
    })
    assert orchestrator.engine_mode == EngineMode.DEEPSEEK

@pytest.mark.asyncio
async def test_monitoring_integration(orchestrator, mock_monitoring):
    """Test integration with monitoring system."""
    orchestrator.monitoring = mock_monitoring
    
    await orchestrator.process_request({
        "type": "analysis",
        "content": "Test content"
    })
    
    # Verify metrics were recorded
    assert mock_monitoring.record_metric.called
    metrics_recorded = mock_monitoring.record_metric.call_args_list
    assert len(metrics_recorded) > 0

@pytest.mark.asyncio
async def test_edge_processing(orchestrator, mock_edge_manager):
    """Test edge processing capabilities."""
    orchestrator.edge_manager = mock_edge_manager
    
    result = await orchestrator.process_request({
        "type": "analysis",
        "content": "Test content",
        "metadata": {"requires_edge": True}
    })
    
    assert mock_edge_manager.process_task.called
    assert "edge_processing" in result

@pytest.mark.asyncio
async def test_blockchain_integration(orchestrator, mock_blockchain):
    """Test blockchain integration for auditing."""
    orchestrator.blockchain = mock_blockchain
    
    await orchestrator.process_request({
        "type": "analysis",
        "content": "Test content",
        "metadata": {"requires_blockchain": True}
    })
    
    assert mock_blockchain.record_execution.called

@pytest.mark.asyncio
async def test_system_state(orchestrator):
    """Test system state monitoring and updates."""
    initial_state = await orchestrator.get_status()
    assert "active_teams" in initial_state
    assert "pending_tasks" in initial_state
    assert "performance_metrics" in initial_state
    
    # Process a request
    await orchestrator.process_request({
        "type": "analysis",
        "content": "Test content"
    })
    
    updated_state = await orchestrator.get_status()
    assert updated_state["metrics"]["total_requests"] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.orchestrator"])