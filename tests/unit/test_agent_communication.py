import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from agents.agent_communication import AgentCommunicationSystem

@pytest.fixture
async def comm_system():
    """Create a test communication system instance."""
    system = AgentCommunicationSystem(api_key="test_key", engine_mode="openai")
    yield system
    await system.cleanup()

@pytest.fixture
def mock_llm():
    """Mock for language model."""
    mock = Mock()
    mock.agenerate = AsyncMock(return_value=Mock(
        generations=[[Mock(message=Mock(content="Test dialogue"))]]
    ))
    return mock

@pytest.mark.asyncio
async def test_send_message(comm_system):
    """Test message sending functionality."""
    # Send test message
    message = await comm_system.send_message(
        from_agent="TestAgent",
        content="Test message",
        priority=1,
        metadata={"test": True}
    )
    
    # Verify message was queued
    queued_message = await comm_system.message_queue.get()
    assert queued_message["from_agent"] == "TestAgent"
    assert queued_message["content"] == "Test message"
    assert queued_message["priority"] == 1
    assert queued_message["metadata"]["test"] is True
    
    # Verify message is in history
    assert len(comm_system.message_history) == 1
    assert comm_system.message_history[0]["content"] == "Test message"

@pytest.mark.asyncio
async def test_get_messages(comm_system):
    """Test message retrieval functionality."""
    # Add test messages
    messages = [
        {
            "from_agent": f"Agent{i}",
            "content": f"Message {i}",
            "priority": i
        }
        for i in range(3)
    ]
    
    for msg in messages:
        await comm_system.send_message(**msg)
    
    # Get all messages
    retrieved = await comm_system.get_messages()
    assert len(retrieved) == 3
    
    # Test filtering by agent
    filtered = await comm_system.get_messages(filter_agent="Agent1")
    assert len(filtered) == 1
    assert filtered[0]["from_agent"] == "Agent1"
    
    # Test limit
    limited = await comm_system.get_messages(limit=2)
    assert len(limited) == 2

@pytest.mark.asyncio
async def test_message_priority(comm_system):
    """Test message priority handling."""
    # Send messages with different priorities
    await comm_system.send_message("AgentA", "Low priority", priority=3)
    await comm_system.send_message("AgentB", "High priority", priority=1)
    await comm_system.send_message("AgentC", "Medium priority", priority=2)
    
    # Get messages and verify order
    messages = await comm_system.get_messages()
    priorities = [msg["priority"] for msg in messages]
    assert priorities == sorted(priorities)  # Should be sorted by priority

@pytest.mark.asyncio
async def test_generate_dialogue(comm_system, mock_llm):
    """Test dialogue generation."""
    comm_system.llm = mock_llm
    
    # Generate dialogue
    dialogue = await comm_system.generate_dialogue(
        task="Test task",
        participants=["AgentA", "AgentB"]
    )
    
    # Verify dialogue generation
    assert isinstance(dialogue, list)
    assert mock_llm.agenerate.called
    assert len(dialogue) > 0

@pytest.mark.asyncio
async def test_broadcast_message(comm_system):
    """Test broadcast message functionality."""
    # Add active agents
    comm_system.active_agents = {"AgentA", "AgentB", "AgentC"}
    
    # Broadcast message
    await comm_system.broadcast_message(
        "Broadcast test",
        exclude_agents=["AgentB"]
    )
    
    # Verify message was sent to correct agents
    messages = await comm_system.get_messages()
    recipients = {msg["from_agent"] for msg in messages}
    assert "System" in recipients
    assert len(messages) == 2  # Should exclude AgentB

@pytest.mark.asyncio
async def test_message_cleanup(comm_system):
    """Test message cleanup functionality."""
    # Add test messages
    await comm_system.send_message("TestAgent", "Test message")
    comm_system.active_agents.add("TestAgent")
    
    # Perform cleanup
    await comm_system.cleanup()
    
    # Verify cleanup
    assert comm_system.message_queue.empty()
    assert len(comm_system.active_agents) == 0
    assert not comm_system.conversation_context

@pytest.mark.asyncio
async def test_conversation_summary(comm_system):
    """Test conversation summary generation."""
    # Setup test state
    await comm_system.send_message("AgentA", "Message 1")
    comm_system.active_agents.add("AgentA")
    comm_system.conversation_context["test"] = "context"
    
    # Get summary
    summary = comm_system.get_conversation_summary()
    
    # Verify summary content
    assert summary["total_messages"] == 1
    assert "AgentA" in summary["active_agents"]
    assert summary["conversation_context"]["test"] == "context"
    assert summary["last_message"] is not None

@pytest.mark.asyncio
async def test_message_validation(comm_system):
    """Test message validation and cleaning."""
    # Test with valid message
    await comm_system.send_message("TestAgent", "Valid message")
    assert len(comm_system.message_history) == 1
    
    # Test with empty content
    with pytest.raises(Exception):
        await comm_system.send_message("TestAgent", "")
    
    # Test content cleaning
    await comm_system.send_message("TestAgent", "**Bold** [text]")
    last_message = comm_system.message_history[-1]
    assert last_message["content"] == "Bold text"  # Should remove formatting

@pytest.mark.asyncio
async def test_error_handling(comm_system):
    """Test error handling in communication system."""
    # Test with invalid priority
    with pytest.raises(Exception):
        await comm_system.send_message("TestAgent", "Test", priority=-1)
    
    # Test with invalid agent name
    with pytest.raises(Exception):
        await comm_system.send_message("", "Test")
    
    # Test with oversized message
    with pytest.raises(Exception):
        await comm_system.send_message("TestAgent", "x" * 1000000)

@pytest.mark.asyncio
async def test_agent_management(comm_system):
    """Test agent registration and management."""
    # Register agents
    comm_system.active_agents.add("AgentA")
    comm_system.active_agents.add("AgentB")
    
    # Test agent presence
    assert "AgentA" in comm_system.active_agents
    assert "AgentB" in comm_system.active_agents
    assert "AgentC" not in comm_system.active_agents
    
    # Remove agent
    comm_system.active_agents.remove("AgentA")
    assert "AgentA" not in comm_system.active_agents

@pytest.mark.asyncio
async def test_conversation_context(comm_system):
    """Test conversation context management."""
    # Add context
    comm_system.conversation_context["key1"] = "value1"
    comm_system.conversation_context["key2"] = {"nested": "value"}
    
    # Verify context
    assert comm_system.conversation_context["key1"] == "value1"
    assert comm_system.conversation_context["key2"]["nested"] == "value"
    
    # Clear context
    comm_system.conversation_context.clear()
    assert not comm_system.conversation_context

@pytest.mark.asyncio
async def test_concurrent_message_handling(comm_system):
    """Test handling of concurrent messages."""
    # Create multiple concurrent message sends
    tasks = [
        comm_system.send_message(
            f"Agent{i}",
            f"Message {i}",
            priority=i % 3
        )
        for i in range(10)
    ]
    
    # Execute concurrently
    await asyncio.gather(*tasks)
    
    # Verify all messages were processed
    messages = await comm_system.get_messages()
    assert len(messages) == 10
    
    # Verify message ordering
    priorities = [msg["priority"] for msg in messages]
    assert priorities == sorted(priorities)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.agent_communication"])