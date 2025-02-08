import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI

from .base import BaseLLM
from ..errors import APIError

logger = logging.getLogger(__name__)

class OpenAIChat(BaseLLM):
    """OpenAI Chat implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(api_key=self.api_key)
        
    async def agenerate(
        self,
        messages_batch: List[List[Dict[str, str]]]
    ) -> Any:
        """Generate responses using OpenAI Chat API."""
        try:
            if not messages_batch:
                raise ValueError("Messages batch cannot be empty")
            
            messages = self._prepare_messages(messages_batch[0])
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Create response object matching expected interface
            return MessageResponse(response.choices[0].message.content)
            
        except Exception as e:
            await self._handle_api_error(e)
    
    async def acompletion(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI Chat API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
            
        except Exception as e:
            await self._handle_api_error(e)
            
    async def _handle_api_error(self, error: Exception) -> None:
        """Handle OpenAI specific API errors."""
        error_msg = str(error)
        if "rate_limit" in error_msg.lower():
            raise APIError("OpenAI rate limit exceeded", {"retry_after": 60})
        elif "invalid_api_key" in error_msg.lower():
            raise APIError("Invalid OpenAI API key")
        else:
            raise APIError(f"OpenAI API error: {error_msg}")

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