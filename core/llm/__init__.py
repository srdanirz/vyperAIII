from .base import BaseLLM
from .openai import OpenAIChat
from .deepseek import core.llm

__all__ = ['BaseLLM', 'OpenAIChat', 'core.llm']

def get_llm(engine_mode: str, api_key: str, model: str = None, **kwargs):
    """Factory function to get LLM instance."""
    if engine_mode.lower() == "deepseek":
        return core.llm(api_key=api_key, model=model or "deepseek-chat", **kwargs)
    else:
        return OpenAIChat(api_key=api_key, model=model or "gpt-4-turbo", **kwargs)