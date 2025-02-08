from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

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
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Message must be a dict, got {type(msg)}")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Message must have 'role' and 'content' keys")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {msg['role']}")

    def _prepare_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare messages for API call."""
        self._validate_messages(messages)
        return messages

    async def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors."""
        logger.error(f"API Error: {str(error)}")
        raise