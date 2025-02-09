import pytest
import asyncio
import os
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock

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
    return test_dir / "data" / "test_files"

@pytest.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test."""
    # Setup
    os.environ["TESTING"] = "true"
    
    yield
    
    # Teardown
    if "TESTING" in os.environ:
        del os.environ["TESTING"]

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
    return mock