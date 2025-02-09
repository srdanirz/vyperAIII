import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from agents.base_agent import BaseAgent
from core.errors import ProcessingError

# Test implementation of BaseAgent
class TestAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_result = kwargs.get("test_result", {"result": "success"})
        self.should_fail = kwargs.get("should_fail", False)
    
    async def _execute(self) -> Dict[str, Any]:
        if self.should_fail:
            raise ProcessingError("Test execution failed")
        return self.test_result

@pytest.fixture
def base_agent():
    """Create a test agent instance."""
    return TestAgent(
        task="test_task",
        openai_api_key="test_key",
        metadata={"test": True},
        shared_data={"shared": "data"}
    )

@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization."""
    agent = TestAgent(
        task="test_task",
        openai_api_key="test_key",
        metadata={"test": True},
        shared_data={"shared": "data"}
    )
    
    assert agent.task == "test_task"
    assert agent.openai_api_key == "test_key"
    assert agent.metadata == {"test": True}
    assert agent.shared_data == {"shared": "data"}
    assert agent.execution_start is None
    assert agent.execution_end is None

@pytest.mark.asyncio
async def test_successful_execution(base_agent):
    """Test successful agent execution."""
    result = await base_agent.execute()
    
    assert result["result"] == "success"
    assert "metadata" in result
    assert "execution_time" in result["metadata"]
    assert "execution_start" in result["metadata"]
    assert "execution_end" in result["metadata"]
    assert result["agent_type"] == "TestAgent"

@pytest.mark.asyncio
async def test_failed_execution():
    """Test failed agent execution."""
    agent = TestAgent(
        task="test_task",
        openai_api_key="test_key",
        should_fail=True
    )
    
    result = await agent.execute()
    
    assert "error" in result
    assert result["error"] == "Test execution failed"
    assert "metadata" in result
    assert result["agent_type"] == "TestAgent"

@pytest.mark.asyncio
async def test_execution_timing(base_agent):
    """Test execution timing measurement."""
    start_time = datetime.now()
    result = await base_agent.execute()
    end_time = datetime.now()
    
    assert base_agent.execution_start >= start_time
    assert base_agent.execution_end <= end_time
    assert result["metadata"]["execution_time"] >= 0

@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test concurrent agent execution."""
    agents = [
        TestAgent(
            task=f"task_{i}",
            openai_api_key="test_key",
            test_result={"result": f"success_{i}"}
        )
        for i in range(5)
    ]
    
    results = await asyncio.gather(*[agent.execute() for agent in agents])
    
    assert len(results) == 5
    assert all(r["result"].startswith("success_") for r in results)
    assert len(set(r["result"] for r in results)) == 5  # All unique

@pytest.mark.asyncio
async def test_metadata_handling(base_agent):
    """Test metadata handling in execution."""
    # Add runtime metadata
    base_agent.metadata["runtime"] = "test_runtime"
    
    result = await base_agent.execute()
    
    assert "metadata" in result
    assert result["metadata"]["test"] is True
    assert result["metadata"]["runtime"] == "test_runtime"

@pytest.mark.asyncio
async def test_shared_data_access(base_agent):
    """Test shared data access during execution."""
    # Modify shared data
    base_agent.shared_data["additional"] = "value"
    
    result = await base_agent.execute()
    
    assert base_agent.shared_data["shared"] == "data"
    assert base_agent.shared_data["additional"] == "value"

@pytest.mark.asyncio
async def test_error_metadata(base_agent):
    """Test metadata in error cases."""
    agent = TestAgent(
        task="test_task",
        openai_api_key="test_key",
        metadata={"test": True},
        should_fail=True
    )
    
    result = await agent.execute()
    
    assert "error" in result
    assert "metadata" in result
    assert result["metadata"]["test"] is True

@pytest.mark.asyncio
async def test_execution_state():
    """Test agent execution state transitions."""
    agent = TestAgent(
        task="test_task",
        openai_api_key="test_key"
    )
    
    assert agent.execution_start is None
    assert agent.execution_end is None
    
    await agent.execute()
    
    assert agent.execution_start is not None
    assert agent.execution_end is not None
    assert agent.execution_end > agent.execution_start

@pytest.mark.asyncio
async def test_result_type_coercion(base_agent):
    """Test result type coercion to dict."""
    # Test with non-dict result
    agent = TestAgent(
        task="test_task",
        openai_api_key="test_key",
        test_result="simple string result"
    )
    
    result = await agent.execute()
    
    assert isinstance(result, dict)
    assert "result" in result
    assert result["result"] == "simple string result"

@pytest.mark.asyncio
async def test_long_running_execution():
    """Test long-running agent execution."""
    async def delayed_execute():
        await asyncio.sleep(0.1)
        return {"result": "delayed"}
    
    agent = TestAgent(
        task="test_task",
        openai_api_key="test_key"
    )
    agent._execute = delayed_execute
    
    result = await agent.execute()
    
    assert result["result"] == "delayed"
    assert result["metadata"]["execution_time"] >= 0.1

@pytest.mark.asyncio
async def test_cleanup_hook():
    """Test agent cleanup hook."""
    cleanup_called = False
    
    class CleanupTestAgent(TestAgent):
        async def cleanup(self):
            nonlocal cleanup_called
            cleanup_called = True
    
    agent = CleanupTestAgent(
        task="test_task",
        openai_api_key="test_key"
    )
    
    await agent.cleanup()
    assert cleanup_called

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.base_agent"])