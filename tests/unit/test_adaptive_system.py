import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from core.llm import get_llm  # Corregida la importación
from core.adaptive import AdaptiveSystem
from core.adaptive import ADAPTATION_MODES, ADAPTATION_TYPES

@pytest.fixture
async def adaptive_system():
    """Create a test instance of AdaptiveSystem."""
    system = AdaptiveSystem()
    yield system
    await system.cleanup()

@pytest.fixture
def mock_learning_metrics():
    return {
        "patterns_detected": 0,
        "successful_adaptations": 0,
        "failed_adaptations": 0,
        "average_success_rate": 0.0
    }

@pytest.mark.asyncio
async def test_process_interaction(adaptive_system):
    """Test basic interaction processing."""
    input_data = {
        "text": "Test interaction",
        "type": "query"
    }
    context = {"user_id": "test_user"}

    result = await adaptive_system.process_interaction(input_data, context)

    assert "response" in result
    assert "adaptations_applied" in result
    assert "confidence" in result
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_feature_extraction(adaptive_system):
    """Test feature extraction from interactions."""
    input_data = {
        "text": "Complex test interaction with specific patterns",
        "type": "query",
        "metadata": {"importance": "high"}
    }
    context = {"user_expertise": "expert"}

    features = await adaptive_system._extract_features(input_data, context)

    assert "text_length" in features
    assert "complexity" in features
    assert "formality" in features
    assert features["text_length"] > 0

@pytest.mark.asyncio
async def test_pattern_detection(adaptive_system):
    """Test pattern detection in features."""
    features = {
        "text_length": 50,
        "complexity": 0.7,
        "formality": 0.8,
        "sentiment": 0.6
    }

    patterns = await adaptive_system._detect_patterns(features)

    assert isinstance(patterns, list)
    if patterns:
        assert hasattr(patterns[0], "pattern_id")
        assert hasattr(patterns[0], "features")
        assert hasattr(patterns[0], "success_rate")

@pytest.mark.asyncio
async def test_system_adaptation(adaptive_system):
    """Test system adaptation based on patterns."""
    patterns = [Mock(
        success_rate=0.9,
        features={
            "complexity": 0.8,
            "formality": 0.7
        }
    )]
    current_features = {
        "complexity": 0.6,
        "formality": 0.5
    }

    adaptations = await adaptive_system._adapt_system(patterns, current_features)

    assert isinstance(adaptations, list)
    if adaptations:
        assert "type" in adaptations[0]
        assert "from" in adaptations[0]
        assert "to" in adaptations[0]
        assert "confidence" in adaptations[0]

@pytest.mark.asyncio
async def test_adaptive_response_generation(adaptive_system):
    """Test generation of adaptive responses."""
    input_data = {"text": "Test input"}
    adaptations = [
        {
            "type": "complexity",
            "from": 5,
            "to": 7,
            "confidence": 0.8
        }
    ]
    context = {"user_id": "test_user"}

    response = await adaptive_system._generate_adaptive_response(
        input_data,
        adaptations,
        context
    )

    assert "content" in response
    assert "success" in response
    assert "confidence" in response
    assert "adaptation_level" in response
    assert "personalization_score" in response

@pytest.mark.asyncio
async def test_continuous_adaptation(adaptive_system):
    """Test continuous adaptation process."""
    # Simulate some patterns
    for i in range(3):
        await adaptive_system.process_interaction(
            {"text": f"Test interaction {i}"},
            {"user_id": "test_user"}
        )

    # Let the system adapt
    await asyncio.sleep(1)

    # Check metrics
    status = await adaptive_system.get_status()
    assert "current_state" in status
    assert "active_patterns" in status
    assert "learning_metrics" in status

@pytest.mark.asyncio
async def test_error_handling(adaptive_system):
    """Test error handling in the adaptive system."""
    # Test with invalid input
    result = await adaptive_system.process_interaction(None, None)
    assert "error" in result

    # Test with malformed input
    result = await adaptive_system.process_interaction(
        {"invalid": "data"},
        {}
    )
    assert result["confidence"] < 0.5

@pytest.mark.asyncio
async def test_state_persistence(adaptive_system):
    """Test system state persistence."""
    # Process some interactions
    await adaptive_system.process_interaction(
        {"text": "Test persistence"},
        {"user_id": "test_user"}
    )

    # Get initial state
    initial_state = adaptive_system.current_state.copy()

    # Cleanup and reinitialize
    await adaptive_system.cleanup()
    await adaptive_system.process_interaction(
        {"text": "Test after cleanup"},
        {"user_id": "test_user"}
    )

    # Compare states
    assert adaptive_system.current_state != initial_state

@pytest.mark.asyncio
async def test_adaptation_modes(adaptive_system):
    """Test different adaptation modes."""
    # Test learning mode
    await adaptive_system.process_interaction(
        {"text": "Learning mode test", "mode": ADAPTATION_MODES["LEARNING"]},
        {"user_id": "test_user"}
    )
    
    # Test optimization mode
    await adaptive_system.process_interaction(
        {"text": "Optimization mode test", "mode": ADAPTATION_MODES["OPTIMIZING"]},
        {"user_id": "test_user"}
    )

    status = await adaptive_system.get_status()
    assert status["current_state"]["interaction_mode"] in ADAPTATION_MODES.values()

@pytest.mark.asyncio
async def test_concurrent_interactions(adaptive_system):
    """Test handling of concurrent interactions."""
    # Create multiple concurrent interactions
    interactions = [
        adaptive_system.process_interaction(
            {"text": f"Concurrent test {i}"},
            {"user_id": f"user_{i}"}
        )
        for i in range(5)
    ]

    # Execute concurrently
    results = await asyncio.gather(*interactions)

    assert len(results) == 5
    assert all("response" in r for r in results)

@pytest.mark.asyncio
async def test_resource_management(adaptive_system):
    """Test resource management during adaptation."""
    # Process resource-intensive interaction
    result = await adaptive_system.process_interaction(
        {
            "text": "Resource intensive test",
            "complexity": "high",
            "resources_required": {"cpu": 0.8, "memory": 0.7}
        },
        {"user_id": "test_user"}
    )

    assert result["metadata"]["patterns_matched"] >= 0
    assert "adaptation_level" in result["metadata"]

@pytest.mark.asyncio
async def test_cleanup(adaptive_system):
    """Test cleanup process."""
    # Add some test data
    await adaptive_system.process_interaction(
        {"text": "Test before cleanup"},
        {"user_id": "test_user"}
    )

    # Perform cleanup
    await adaptive_system.cleanup()

    # Verify state
    status = await adaptive_system.get_status()
    assert status["active_patterns"] == 0
    assert status["current_state"]["interaction_mode"] == "normal"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.adaptive"])