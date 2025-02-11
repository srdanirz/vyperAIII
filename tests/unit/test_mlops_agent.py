import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import optuna
import json
from typing import Dict, Any

from agents.mlops_agent import MLOpsAgent
from core.errors import ProcessingError

@pytest.fixture
async def mlops_agent():
    """Create a test MLOps agent instance."""
    agent = MLOpsAgent(
        api_key="test_key",
        engine_mode="openai",
        metadata={"test": True}
    )
    yield agent
    await agent.cleanup()

@pytest.fixture
def mock_model_data():
    """Mock model performance data."""
    return {
        "model_name": "test_model",
        "predictions": [1, 0, 1, 1, 0],
        "ground_truth": [1, 0, 1, 0, 0],
        "metrics": {
            "accuracy": 0.8,
            "latency": 0.05
        }
    }

@pytest.mark.asyncio
async def test_monitor_performance(mlops_agent, mock_model_data):
    """Test model performance monitoring."""
    result = await mlops_agent.monitor_performance(
        model_name=mock_model_data["model_name"],
        predictions=mock_model_data["predictions"],
        ground_truth=mock_model_data["ground_truth"]
    )
    
    assert "metrics" in result
    assert "drift_detected" in result
    assert isinstance(result["metrics"]["accuracy"], float)
    assert "timestamp" in result
    assert result["model_name"] == mock_model_data["model_name"]

@pytest.mark.asyncio
async def test_optimize_model(mlops_agent):
    """Test model optimization process."""
    result = await mlops_agent.optimize_model(
        model_name="test_model",
        optimization_objective="latency",
        n_trials=5
    )
    
    assert "best_parameters" in result
    assert "best_value" in result
    assert "optimization_history" in result
    assert len(result["optimization_history"]) == 5
    assert all("value" in trial for trial in result["optimization_history"])

@pytest.mark.asyncio
async def test_run_ab_test(mlops_agent):
    """Test A/B testing functionality."""
    test_data = [
        {"input": "test", "expected": 1} for _ in range(10)
    ]
    
    result = await mlops_agent.run_ab_test(
        model_a="model_a",
        model_b="model_b",
        test_data=test_data,
        metrics=["accuracy", "latency"]
    )
    
    assert "models" in result
    assert "results" in result
    assert "significance" in result
    assert "winner" in result
    assert result["timestamp"] is not None

@pytest.mark.asyncio
async def test_track_experiment(mlops_agent):
    """Test experiment tracking."""
    experiment = await mlops_agent.track_experiment(
        experiment_name="test_experiment",
        config={"learning_rate": 0.01},
        metrics={"accuracy": 0.85, "loss": 0.15}
    )
    
    assert experiment["name"] == "test_experiment"
    assert "config" in experiment
    assert "metrics" in experiment
    assert experiment["status"] == "completed"
    assert "timestamp" in experiment

@pytest.mark.asyncio
async def test_detect_data_drift(mlops_agent):
    """Test data drift detection."""
    import pandas as pd
    
    reference_data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(0, 1, 1000)
    })
    
    current_data = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1, 1000),  # Shifted distribution
        "feature2": np.random.normal(0, 1, 1000)
    })
    
    result = await mlops_agent.detect_data_drift(
        reference_data=reference_data,
        current_data=current_data,
        features=["feature1", "feature2"]
    )
    
    assert "drift_detected" in result
    assert "feature_metrics" in result
    assert "timestamp" in result
    assert result["feature_metrics"]["feature1"]["drift_detected"]
    assert not result["feature_metrics"]["feature2"]["drift_detected"]

@pytest.mark.asyncio
async def test_performance_degradation_detection(mlops_agent):
    """Test performance degradation detection."""
    # Simulate historical performance
    mlops_agent.system_metrics["model_performance"]["test_model"] = {
        "current_metrics": {
            "accuracy": 0.9,
            "latency": 0.05
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Test with degraded performance
    result = await mlops_agent._detect_performance_drift(
        model_name="test_model",
        current_metrics={
            "accuracy": 0.7,  # Significant drop
            "latency": 0.08  # Increased latency
        }
    )
    
    assert result  # Should detect degradation

@pytest.mark.asyncio
async def test_optimization_trigger(mlops_agent):
    """Test automatic optimization triggering."""
    # Mock degradation detection
    mlops_agent._detect_performance_drift = AsyncMock(return_value=True)
    
    # Trigger optimization
    await mlops_agent._trigger_optimization("test_model")
    
    # Verify optimization was triggered
    assert len(mlops_agent.training_state["active_jobs"]) > 0
    assert mlops_agent.metrics["total_jobs"] > 0

@pytest.mark.asyncio
async def test_model_evaluation(mlops_agent):
    """Test model evaluation process."""
    test_data = [
        {"input": "test", "expected": 1} for _ in range(5)
    ]
    
    result = await mlops_agent._evaluate_model(
        model_name="test_model",
        test_data=test_data
    )
    
    assert "model_name" in result
    assert "metrics" in result
    assert "predictions" in result
    assert "execution_time" in result
    assert len(result["predictions"]) == 5

@pytest.mark.asyncio
async def test_kl_divergence_calculation(mlops_agent):
    """Test KL divergence calculation for distributions."""
    import pandas as pd
    
    # Create two slightly different distributions
    p = pd.Series(np.random.normal(0, 1, 1000))
    q = pd.Series(np.random.normal(0.2, 1, 1000))
    
    divergence = mlops_agent._calculate_kl_divergence(p, q)
    
    assert isinstance(divergence, float)
    assert divergence > 0  # KL divergence should be positive

@pytest.mark.asyncio
async def test_significance_calculation(mlops_agent):
    """Test statistical significance calculation."""
    metrics_a = {"accuracy": 0.85, "latency": 0.05}
    metrics_b = {"accuracy": 0.82, "latency": 0.06}
    
    result = await mlops_agent._calculate_significance(metrics_a, metrics_b)
    
    assert "metrics" in result
    assert "winner" in result
    assert "confidence" in result
    assert all("significant" in v for v in result["metrics"].values())

@pytest.mark.asyncio
async def test_cleanup(mlops_agent):
    """Test cleanup process."""
    # Add some test data
    mlops_agent.experiment_history.append({
        "name": "test_experiment",
        "status": "completed"
    })
    
    await mlops_agent.cleanup()
    
    assert len(mlops_agent.training_state["active_jobs"]) == 0
    assert len(mlops_agent.training_state["model_versions"]) == 0
    assert mlops_agent.metrics == {
        "total_jobs": 0,
        "successful_jobs": 0,
        "failed_jobs": 0,
        "average_improvement": 0.0
    }

@pytest.mark.asyncio
async def test_status_reporting(mlops_agent):
    """Test status reporting functionality."""
    status = await mlops_agent.get_status()
    
    assert status["agent_type"] == "MLOpsAgent"
    assert "engine_mode" in status
    assert "active_experiments" in status
    assert "system_metrics" in status
    assert "optimization_config" in status

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.mlops_agent"])