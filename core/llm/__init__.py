from typing import Optional
from .base import BaseLLM
from .openai import OpenAIChat
from .deepseek import DeepSeekChat

def get_llm(engine_mode: str, api_key: str, model: Optional[str] = None, **kwargs):
    """Factory function to get LLM instance."""
    if engine_mode.lower() == "deepseek":
        return DeepSeekChat(api_key=api_key, model=model or "deepseek-chat", **kwargs)
    else:
        return OpenAIChat(api_key=api_key, model=model or "gpt-4-turbo", **kwargs)

__all__ = ['BaseLLM', 'OpenAIChat', 'DeepSeekChat', 'get_llm']