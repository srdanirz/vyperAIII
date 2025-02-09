from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Base class for Language Model interfaces."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_config = kwargs
        
        # Set default values for common parameters
        self.extra_config.setdefault("top_p", 1.0)
        self.extra_config.setdefault("frequency_penalty", 0.0)
        self.extra_config.setdefault("presence_penalty", 0.0)
        self.extra_config.setdefault("timeout", 30)
    
    @abstractmethod
    async def agenerate(
        self,
        messages_batch: List[List[Dict[str, str]]]
    ) -> Any:
        """
        Generate responses for a batch of message sequences.
        
        Args:
            messages_batch: List of message sequences, where each sequence is a list of
                          message dictionaries with 'role' and 'content' keys.
        
        Returns:
            Object with generations[0][0].message.content attribute containing the response
        """
        pass

    @abstractmethod
    async def acompletion(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate a completion for a single prompt.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters for the completion
        
        Returns:
            Generated text
        """
        pass

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format."""
        if not isinstance(messages, list):
            raise ValueError(f"Messages must be a list, got {type(messages)}")
            
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Message must be a dict, got {type(msg)}")
                
            if "role" not in msg:
                raise ValueError("Message must have 'role' key")
                
            if msg["role"] not in ["system", "user", "assistant", "function"]:
                raise ValueError(f"Invalid role: {msg['role']}")
                
            if "content" not in msg and msg["role"] != "function":
                raise ValueError("Message must have 'content' key")

    def _prepare_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare messages for API call."""
        self._validate_messages(messages)
        return messages.copy()

    async def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors."""
        logger.error(f"API Error: {str(error)}")
        raise

    async def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in a text.
        Default implementation gives rough estimate.
        Override in specific implementations for accuracy.
        """
        # Rough estimate: 4 characters per token
        return len(text) // 4

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        self.extra_config.update(kwargs)
        
        # Update main parameters if provided
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
        if "model" in kwargs:
            self.model = kwargs["model"]

    async def validate_response(
        self,
        response: Any,
        expected_format: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate response format.
        Override in specific implementations for detailed validation.
        """
        try:
            if not response:
                return False
                
            if not hasattr(response, "generations"):
                return False
                
            if not response.generations or not response.generations[0]:
                return False
                
            if not hasattr(response.generations[0][0], "message"):
                return False
                
            if not hasattr(response.generations[0][0].message, "content"):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return False

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Convenience method for chat completions.
        Uses agenerate internally but returns just the content string.
        """
        response = await self.agenerate([messages])
        return response.generations[0][0].message.content