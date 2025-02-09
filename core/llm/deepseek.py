import logging
import aiohttp
import ssl
import certifi
from typing import List, Dict, Any

from .base import BaseLLM
from ..errors import APIError

logger = logging.getLogger(__name__)

class DeepSeekChat(BaseLLM):
    """DeepSeek Chat implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.deepseek.com/v1"
        
        # SSL context with certifi
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.load_verify_locations(cafile=certifi.where())
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")

    async def agenerate(
        self,
        messages_batch: List[List[Dict[str, str]]]
    ) -> Any:
        """Generate responses using DeepSeek Chat API."""
        try:
            if not messages_batch:
                raise ValueError("Messages batch cannot be empty")
            
            messages = self._prepare_messages(messages_batch[0])
            
            # Prepare payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False
            }
            
            if self.max_tokens:
                payload["max_tokens"] = self.max_tokens
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Create connector with SSL
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise APIError(
                            f"DeepSeek API error {resp.status}",
                            {"response": error_text}
                        )
                    
                    data = await resp.json()
            
            if "choices" not in data:
                raise APIError("Invalid response from DeepSeek API")
            
            # Create compatible response
            return MessageResponse(data["choices"][0]["message"]["content"])
            
        except Exception as e:
            await self._handle_api_error(e)

    async def acompletion(self, prompt: str, **kwargs) -> str:
        """Generate completion using DeepSeek Chat API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False,
                **kwargs
            }
            
            if self.max_tokens:
                payload["max_tokens"] = self.max_tokens
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise APIError(
                            f"DeepSeek API error {resp.status}",
                            {"response": error_text}
                        )
                    
                    data = await resp.json()
                    
            return data["choices"][0]["text"]
            
        except Exception as e:
            await self._handle_api_error(e)

    async def _handle_api_error(self, error: Exception) -> None:
        """Handle DeepSeek specific API errors."""
        error_msg = str(error)
        if "rate_limit" in error_msg.lower():
            raise APIError("DeepSeek rate limit exceeded", {"retry_after": 60})
        elif "invalid_api_key" in error_msg.lower():
            raise APIError("Invalid DeepSeek API key")
        elif isinstance(error, APIError):
            raise error
        else:
            raise APIError(f"DeepSeek API error: {error_msg}")

class MessageResponse:
    """Mock response object matching expected interface."""
    
    def __init__(self, content: str):
        self.generations = [[MockGeneration(content)]]

class MockGeneration:
    """Mock generation object matching expected interface."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)

class MockMessage:
    """Mock message object matching expected interface."""
    
    def __init__(self, content: str):
        self.content = content