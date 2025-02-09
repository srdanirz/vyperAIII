import logging
import aiohttp
import ssl
import certifi
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from .base import BaseLLM
from ..errors import APIError

logger = logging.getLogger(__name__)

class DeepSeekChat(BaseLLM):
    """DeepSeek Chat implementation with robust error handling and full interface implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.deepseek.com/v1"
        
        # SSL context with certifi
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.load_verify_locations(cafile=certifi.where())
        
        # Validate API key
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")

        # Rate limiting configuration
        self.rate_limit = kwargs.get('rate_limit', 60)  # requests per minute
        self.request_count = 0
        self.last_reset = datetime.now()
        self._rate_limit_lock = asyncio.Lock()

    async def _check_rate_limit(self) -> None:
        """Implement rate limiting."""
        async with self._rate_limit_lock:
            now = datetime.now()
            if (now - self.last_reset).total_seconds() >= 60:
                self.request_count = 0
                self.last_reset = now
            
            if self.request_count >= self.rate_limit:
                wait_time = 60 - (now - self.last_reset).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    self.request_count = 0
                    self.last_reset = datetime.now()
            
            self.request_count += 1

    async def agenerate(
        self,
        messages_batch: List[List[Dict[str, str]]]
    ) -> Any:
        """Generate responses using DeepSeek Chat API with proper error handling."""
        try:
            if not messages_batch:
                raise ValueError("Messages batch cannot be empty")
            
            await self._check_rate_limit()
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
            
            timeout = aiohttp.ClientTimeout(total=self.extra_config.get("timeout", 30))
            
            # Create connector with SSL
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise APIError(
                            f"DeepSeek API error {resp.status}",
                            {
                                "status_code": resp.status,
                                "response": error_text,
                                "request": {
                                    "url": str(resp.url),
                                    "payload": payload
                                }
                            }
                        )
                    
                    data = await resp.json()
            
            if "choices" not in data:
                raise APIError(
                    "Invalid response from DeepSeek API",
                    {"response": data}
                )
            
            return MessageResponse(
                content=data["choices"][0]["message"]["content"],
                role=data["choices"][0]["message"].get("role", "assistant"),
                model=self.model,
                usage=data.get("usage", {})
            )
            
        except asyncio.TimeoutError:
            raise APIError("Request timeout", {"timeout": self.extra_config.get("timeout", 30)})
        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}", {"original_error": str(e)})
        except Exception as e:
            await self._handle_api_error(e)

    async def acompletion(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate completion using DeepSeek Chat API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.agenerate([messages])
            return response.generations[0][0].message.content
            
        except Exception as e:
            await self._handle_api_error(e)

    async def _handle_api_error(self, error: Exception) -> None:
        """Handle DeepSeek specific API errors with detailed information."""
        error_msg = str(error)
        error_data = getattr(error, 'details', {}) if isinstance(error, APIError) else {}
        
        if isinstance(error, asyncio.TimeoutError):
            raise APIError("Request timeout", {"timeout": self.extra_config.get("timeout", 30)})
        
        if "rate_limit" in error_msg.lower():
            raise APIError(
                "DeepSeek rate limit exceeded",
                {
                    "retry_after": 60,
                    "rate_limit": self.rate_limit,
                    **error_data
                }
            )
            
        if "invalid_api_key" in error_msg.lower():
            raise APIError(
                "Invalid DeepSeek API key",
                {"api_key_valid": False, **error_data}
            )
            
        if isinstance(error, APIError):
            raise error
            
        raise APIError(
            f"DeepSeek API error: {error_msg}",
            {
                "error_type": error.__class__.__name__,
                "traceback": getattr(error, "__traceback__", None),
                **error_data
            }
        )

    async def validate_response(
        self,
        response: Any,
        expected_format: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate response format comprehensively."""
        try:
            if not response:
                return False
                
            if not hasattr(response, "generations"):
                return False
                
            if not response.generations or not response.generations[0]:
                return False
                
            generation = response.generations[0][0]
            if not hasattr(generation, "message"):
                return False
                
            message = generation.message
            if not hasattr(message, "content"):
                return False
                
            if expected_format:
                # Validate against expected format
                for key, expected_type in expected_format.items():
                    if not hasattr(message, key):
                        return False
                    if not isinstance(getattr(message, key), expected_type):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return False

class MessageResponse:
    """Complete message response implementation."""
    
    def __init__(
        self,
        content: str,
        role: str = "assistant",
        model: str = "unknown",
        usage: Optional[Dict[str, int]] = None
    ):
        self.generations = [[
            MessageGeneration(
                message=Message(
                    content=content,
                    role=role
                ),
                model=model,
                usage=usage or {}
            )
        ]]

class MessageGeneration:
    """Message generation with complete interface."""
    
    def __init__(
        self,
        message: 'Message',
        model: str = "unknown",
        usage: Optional[Dict[str, int]] = None
    ):
        self.message = message
        self.model = model
        self.usage = usage or {}
        self.finish_reason = None

class Message:
    """Complete message interface."""
    
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role
        self.function_call = None