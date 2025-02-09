import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from aiohttp import ClientTimeout
import ssl
import json
from datetime import datetime, timedelta

from core.llm.deepseek import DeepSeekChat, MessageResponse, MessageGeneration, Message
from core.errors import APIError

@pytest.fixture
def mock_response():
    return {
        "choices": [{
            "message": {
                "content": "Test response",
                "role": "assistant"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def deepseek_chat():
    return DeepSeekChat(
        api_key="test_key",
        model="test-model",
        temperature=0.7
    )

@pytest.fixture
def mock_aiohttp_session():
    async def mock_post(*args, **kwargs):
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                }
            }]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        return mock_response

    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post = mock_post
        yield mock_session

@pytest.mark.asyncio
async def test_initialization():
    """Test DeepSeekChat initialization."""
    # Test successful initialization
    chat = DeepSeekChat(api_key="test_key")
    assert chat.api_key == "test_key"
    assert isinstance(chat.ssl_context, ssl.SSLContext)
    
    # Test initialization without API key
    with pytest.raises(ValueError, match="DeepSeek API key is required"):
        DeepSeekChat(api_key=None)

@pytest.mark.asyncio
async def test_rate_limiting(deepseek_chat):
    """Test rate limiting functionality."""
    # Configure rate limit
    deepseek_chat.rate_limit = 2
    
    # Make requests
    await deepseek_chat._check_rate_limit()
    await deepseek_chat._check_rate_limit()
    
    # Third request should wait
    start_time = datetime.now()
    await deepseek_chat._check_rate_limit()
    elapsed = (datetime.now() - start_time).total_seconds()
    
    assert elapsed >= 60  # Should have waited for rate limit reset

@pytest.mark.asyncio
async def test_message_generation(deepseek_chat, mock_aiohttp_session, mock_response):
    """Test message generation."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Configure mock
    mock_aiohttp_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.json = \
        AsyncMock(return_value=mock_response)
    
    # Test successful generation
    response = await deepseek_chat.agenerate([messages])
    
    assert response.generations[0][0].message.content == "Test response"
    assert response.generations[0][0].message.role == "assistant"

@pytest.mark.asyncio
async def test_error_handling(deepseek_chat, mock_aiohttp_session):
    """Test error handling."""
    # Test API error
    mock_aiohttp_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.status = 400
    mock_aiohttp_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.text = \
        AsyncMock(return_value="API Error")
    
    with pytest.raises(APIError) as exc_info:
        await deepseek_chat.agenerate([[{"role": "user", "content": "Hello"}]])
    
    assert "DeepSeek API error" in str(exc_info.value)
    
    # Test timeout
    mock_aiohttp_session.return_value.__aenter__.side_effect = asyncio.TimeoutError()
    
    with pytest.raises(APIError) as exc_info:
        await deepseek_chat.agenerate([[{"role": "user", "content": "Hello"}]])
    
    assert "Request timeout" in str(exc_info.value)

@pytest.mark.asyncio
async def test_response_validation(deepseek_chat):
    """Test response validation."""
    # Valid response
    valid_response = MessageResponse(
        content="Test",
        role="assistant",
        model="test-model"
    )
    assert await deepseek_chat.validate_response(valid_response)
    
    # Invalid response - missing content
    invalid_response = Mock()
    invalid_response.generations = [[Mock(message=Mock())]]
    assert not await deepseek_chat.validate_response(invalid_response)
    
    # Continuación de test_response_validation
    # Invalid response - empty generations
    invalid_response = Mock()
    invalid_response.generations = []
    assert not await deepseek_chat.validate_response(invalid_response)
    
    # Test with expected format
    expected_format = {
        "content": str,
        "role": str
    }
    assert await deepseek_chat.validate_response(valid_response, expected_format)

@pytest.mark.asyncio
async def test_completion_endpoint(deepseek_chat, mock_aiohttp_session):
    """Test the completion endpoint."""
    prompt = "Test prompt"
    response = await deepseek_chat.acompletion(prompt)
    
    assert isinstance(response, str)
    assert response == "Test response"

@pytest.mark.asyncio
async def test_message_interfaces():
    """Test message interface implementations."""
    # Test Message class
    message = Message("Test content", role="assistant")
    assert message.content == "Test content"
    assert message.role == "assistant"
    assert message.function_call is None
    
    # Test MessageGeneration class
    gen = MessageGeneration(
        message=message,
        model="test-model",
        usage={"total_tokens": 10}
    )
    assert gen.message.content == "Test content"
    assert gen.model == "test-model"
    assert gen.usage == {"total_tokens": 10}
    
    # Test MessageResponse class
    response = MessageResponse(
        content="Test content",
        role="assistant",
        model="test-model",
        usage={"total_tokens": 10}
    )
    assert len(response.generations) == 1
    assert response.generations[0][0].message.content == "Test content"

@pytest.mark.asyncio
async def test_ssl_configuration(deepseek_chat):
    """Test SSL context configuration."""
    assert isinstance(deepseek_chat.ssl_context, ssl.SSLContext)
    assert deepseek_chat.ssl_context.verify_mode == ssl.CERT_REQUIRED

@pytest.mark.asyncio
async def test_request_preparation(deepseek_chat):
    """Test request preparation and validation."""
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                }
            }]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        mock_post.return_value = mock_response
        
        await deepseek_chat.agenerate([messages])
        
        # Verify request preparation
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test_key"
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        
        payload = json.loads(call_kwargs["json"]["messages"][0]["content"])
        assert payload == "Hello"

@pytest.mark.asyncio
async def test_error_detail_preservation(deepseek_chat, mock_aiohttp_session):
    """Test that error details are preserved in exceptions."""
    error_response = {
        "error": {
            "message": "Invalid request",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    }
    
    mock_aiohttp_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.status = 401
    mock_aiohttp_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.text = \
        AsyncMock(return_value=json.dumps(error_response))
    
    with pytest.raises(APIError) as exc_info:
        await deepseek_chat.agenerate([[{"role": "user", "content": "Hello"}]])
    
    error_details = exc_info.value.details
    assert "status_code" in error_details
    assert error_details["status_code"] == 401
    assert "response" in error_details

@pytest.mark.asyncio
async def test_concurrent_requests(deepseek_chat, mock_aiohttp_session):
    """Test handling of concurrent requests."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Make multiple concurrent requests
    tasks = [
        deepseek_chat.agenerate([messages])
        for _ in range(5)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Verify all requests completed successfully
    assert len(responses) == 5
    assert all(response.generations[0][0].message.content == "Test response" 
              for response in responses)

@pytest.mark.asyncio
async def test_timeout_configuration(deepseek_chat):
    """Test timeout configuration and handling."""
    # Configure custom timeout
    deepseek_chat.extra_config["timeout"] = 5
    
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.side_effect = \
            asyncio.TimeoutError()
        
        with pytest.raises(APIError) as exc_info:
            await deepseek_chat.agenerate(
                [[{"role": "user", "content": "Hello"}]]
            )
        
        assert "timeout" in exc_info.value.details
        assert exc_info.value.details["timeout"] == 5

if __name__ == '__main__':
    pytest.main([__file__, '-v'])