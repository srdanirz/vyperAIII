import logging
from typing import List, Dict, Any, Optional
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
                max_tokens=self.max_tokens,
                **self.extra_config
            )
            
            return {
                "generations": [[{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]]
            }
            
        except Exception as e:
            await self._handle_api_error(e)
    
    async def acompletion(
        self, 
        prompt: str,
        **kwargs
    ) -> str:
        """Generate completion using OpenAI Chat API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **{**self.extra_config, **kwargs}
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
        elif "context_length_exceeded" in error_msg.lower():
            raise APIError("OpenAI context length exceeded")
        else:
            raise APIError(f"OpenAI API error: {error_msg}")

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format specifically for OpenAI."""
        super()._validate_messages(messages)
        
        # Additional OpenAI-specific validations
        for msg in messages:
            if msg["role"] not in ["system", "user", "assistant", "function"]:
                raise ValueError(f"Invalid role for OpenAI: {msg['role']}")
            
            if "content" not in msg and msg["role"] != "function":
                raise ValueError("Messages must have content except for function messages")

    def _prepare_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API call."""
        self._validate_messages(messages)
        
        # Convert any custom message formats to OpenAI format
        prepared_messages = []
        for msg in messages:
            prepared_msg = {
                "role": msg["role"],
                "content": msg.get("content", "")
            }
            
            # Handle function messages
            if msg["role"] == "function":
                prepared_msg["name"] = msg.get("name", "unknown_function")
                if "function_call" in msg:
                    prepared_msg["function_call"] = msg["function_call"]
                    
            prepared_messages.append(prepared_msg)
            
        return prepared_messages