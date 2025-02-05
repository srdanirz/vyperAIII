import logging
import os
import json
from typing import Dict, Any, Optional

import aiohttp  # Si Deepseek se consume vía HTTP
from dotenv import load_dotenv

from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
load_dotenv()  # Para cargar DEEPSEEK_API_KEY si no se carga en otro lugar

class DeepSeekAgent(BaseAgent):
    """
    Agente para interactuar con la API de Deepseek.
    Ajusta la lógica interna según las operaciones que desees realizar.
    """

    def __init__(
        self,
        task: str,
        openai_api_key: str,
        shared_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)

        # Leer la API Key de Deepseek de variables de entorno
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        # Puedes usar ChatOpenAI si planeas mezclar con LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.7
        )

        if not self.deepseek_api_key:
            logger.warning("No se encontró DEEPSEEK_API_KEY en .env")

    async def _execute(self) -> Dict[str, Any]:
        """
        Lógica principal para invocar los endpoints de Deepseek.
        Retorna un dict con el resultado.
        """
        if not self.deepseek_api_key:
            return {
                "error": "No Deepseek API Key found",
                "metadata": self.metadata
            }

        try:
            # Llama a la función que implementa tu integración
            deepseek_result = await self._call_deepseek_endpoint()
            # Retornar el resultado en un diccionario
            return {
                "deepseek_result": deepseek_result,
                "metadata": self.metadata
            }
        except Exception as e:
            logger.error(f"Error in DeepSeekAgent: {e}", exc_info=True)
            return {
                "error": str(e),
                "metadata": self.metadata
            }

    async def _call_deepseek_endpoint(self) -> Any:
        """
        Ejemplo de petición a la API de Deepseek usando aiohttp (o requests).
        Ajusta en base a la documentación real de Deepseek.
        """
        # Suponiendo un endpoint e.g. https://api.deepseek.ai/search
        url = "https://api.deepseek.ai/search"

        payload = {
            "query": self.task,
            # ...cualquier otro dato que Deepseek requiera...
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                else:
                    text = await resp.text()
                    raise ValueError(f"Deepseek returned status {resp.status}: {text}")
