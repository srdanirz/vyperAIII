import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from agents.data_processing_agent import DataProcessingAgent
from core.llm import get_llm

@pytest.fixture
def mock_llm():
    mock = Mock()
    mock.agenerate = AsyncMock(return_value=Mock(
        generations=[[Mock(message=Mock(content="Processed data analysis"))]]
    ))
    return mock

@pytest.fixture
async def data_processing_agent(mock_llm):
    with patch('agents.data_processing_agent.get_llm', return_value=mock_llm):
        agent = DataProcessingAgent(
            task="process test data",
            openai_api_key="test_key",
            metadata={"engine_mode": "openai"},
            shared_data={"raw_data": "test data"}
        )
        yield agent

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=5),
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    })

@pytest.mark.asyncio
async def test_basic_execution(data_processing_agent):
    """Test basic agent execution."""
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert "processed_data" in result
    assert isinstance(result["metadata"], dict)
    assert "execution_time" in result["metadata"]

@pytest.mark.asyncio
async def test_data_processing_with_dataframe(data_processing_agent, sample_dataframe):
    """Test processing with pandas DataFrame."""
    data_processing_agent.shared_data["raw_data"] = sample_dataframe
    
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert "processed_data" in result
    assert "metadata" in result
    assert result["metadata"]["execution_time"] > 0

@pytest.mark.asyncio
async def test_error_handling(data_processing_agent):
    """Test error handling during processing."""
    # Set invalid data
    data_processing_agent.shared_data["raw_data"] = None
    
    result = await data_processing_agent.execute()
    
    assert "error" in result
    assert result["metadata"]["execution_time"] > 0

@pytest.mark.asyncio
async def test_different_data_types(data_processing_agent):
    """Test processing different types of data."""
    test_cases = [
        # Dictionary data
        {
            "input": {"key1": "value1", "key2": "value2"},
            "expected_type": "dictionary"
        },
        # List data
        {
            "input": [1, 2, 3, 4, 5],
            "expected_type": "list"
        },
        # JSON string
        {
            "input": '{"name": "test", "value": 123}',
            "expected_type": "json"
        },
        # CSV string
        {
            "input": "col1,col2\nval1,val2\nval3,val4",
            "expected_type": "csv"
        }
    ]
    
    for case in test_cases:
        data_processing_agent.shared_data["raw_data"] = case["input"]
        result = await data_processing_agent.execute()
        assert result is not None
        assert "processed_data" in result

@pytest.mark.asyncio
async def test_data_transformation(data_processing_agent, sample_dataframe):
    """Test data transformation capabilities."""
    # Test aggregation
    data_processing_agent.shared_data["raw_data"] = sample_dataframe
    data_processing_agent.task = "aggregate data by category"
    
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert "processed_data" in result

@pytest.mark.asyncio
async def test_metadata_handling(data_processing_agent):
    """Test handling of metadata during processing."""
    # Add custom metadata
    data_processing_agent.metadata["custom_field"] = "test_value"
    data_processing_agent.metadata["processing_flags"] = ["clean", "normalize"]
    
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert "metadata" in result
    assert "custom_field" in result["metadata"]
    assert "processing_flags" in result["metadata"]

@pytest.mark.asyncio
async def test_shared_data_access(data_processing_agent, sample_dataframe):
    """Test access and modification of shared data."""
    # Setup shared data
    data_processing_agent.shared_data = {
        "raw_data": sample_dataframe,
        "settings": {"groupby": "category"},
        "previous_results": {"some": "data"}
    }
    
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert "processed_data" in result

@pytest.mark.asyncio
async def test_large_dataset_handling(data_processing_agent):
    """Test handling of large datasets."""
    # Create large dataset
    large_df = pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=10000),
        'value': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    data_processing_agent.shared_data["raw_data"] = large_df
    
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert "processed_data" in result
    assert "metadata" in result
    assert result["metadata"]["execution_time"] > 0

@pytest.mark.asyncio
async def test_concurrent_processing(data_processing_agent, sample_dataframe):
    """Test concurrent processing capabilities."""
    # Create multiple processing tasks
    tasks = []
    for i in range(5):
        agent_copy = DataProcessingAgent(
            task=f"process task {i}",
            openai_api_key="test_key",
            metadata={"task_id": i},
            shared_data={"raw_data": sample_dataframe.copy()}
        )
        tasks.append(agent_copy.execute())
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    assert all("processed_data" in result for result in results)

@pytest.mark.asyncio
async def test_error_recovery(data_processing_agent, sample_dataframe):
    """Test error recovery capabilities."""
    # Simulate a processing error
    error_df = sample_dataframe.copy()
    error_df.iloc[0, 0] = None  # Introduce an error
    
    data_processing_agent.shared_data["raw_data"] = error_df
    
    result = await data_processing_agent.execute()
    
    assert result is not None
    assert ("processed_data" in result or "error" in result)
    assert "metadata" in result

@pytest.mark.asyncio
async def test_input_validation(data_processing_agent):
    """Test input data validation."""
    invalid_inputs = [
        None,
        "",
        123,
        True,
        lambda x: x
    ]
    
    for invalid_input in invalid_inputs:
        data_processing_agent.shared_data["raw_data"] = invalid_input
        result = await data_processing_agent.execute()
        
        assert result is not None
        assert "error" in result or "processed_data" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.data_processing_agent"])