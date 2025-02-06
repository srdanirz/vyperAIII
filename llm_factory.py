# llm_factory.py

from typing import Any
from langchain_openai import ChatOpenAI
from deepseek_chat import DeepSeekChat

def get_llm(
    engine_mode: str,
    api_key: str,
    model: str = "gpt-4-turbo",
    temperature: float = 0.7
) -> Any:
    """
    Retorna un objeto con la interfaz .agenerate(...):
      - Si engine_mode.lower()=="deepseek", retorna DeepSeekChat (API de DeepSeek).
      - Si no, retorna ChatOpenAI (API de OpenAI).
    """
    if engine_mode.lower() == "deepseek":
        # Forzamos a 'deepseek-chat' si detectamos un 'gpt-...':
        if model.startswith("gpt-"):
            model = "deepseek-chat"
        return DeepSeekChat(api_key=api_key, model=model, temperature=temperature)
    else:
        # Modo openai (default)
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
