# agents/deepseek_agent.py

import logging
import os
import json
from typing import Dict, Any, Optional

import aiohttp
from dotenv import load_dotenv

from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
load_dotenv()  # Para leer DEEPSEEK_API_KEY

class DeepSeekAgent(BaseAgent):
    """
    Agente para interactuar directamente con la API de Deepseek (HTTP).
    """

    def __init__(
        self,
        task: str,
        openai_api_key: str,
        shared_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        # Si planeas usar ChatOpenAI para mezclar...
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.7
        )

    async def _execute(self) -> Dict[str, Any]:
        if not self.deepseek_api_key:
            return {
                "error":"No DEEPSEEK_API_KEY found in env",
                "metadata": self.metadata
            }
        try:
            result = await self._call_deepseek_api()
            return {
                "deepseek_result": result,
                "metadata": self.metadata
            }
        except Exception as e:
            logger.error(f"Error in DeepSeekAgent: {e}", exc_info=True)
            return {
                "error": str(e),
                "metadata": self.metadata
            }

    async def _call_deepseek_api(self)->Any:
        url="https://api.deepseek.ai/search"
        payload={
            "query": self.task
        }
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url,json=payload,headers=headers) as resp:
                if resp.status==200:
                    return await resp.json()
                else:
                    text=await resp.text()
                    raise ValueError(f"DeepSeek error {resp.status}: {text}")
