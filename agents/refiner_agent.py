# agents/refiner_agent.py

import logging
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RefinerAgent(BaseAgent):
    """
    Agente para refinar queries o mejorar estrategias de búsqueda.
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        shared_data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4",
            temperature=0
        )

    async def _execute(self) -> Dict[str, Any]:
        try:
            return await self._generate_refinement_plan()
        except Exception as e:
            logger.error(f"RefinerAgent error: {e}", exc_info=True)
            return {"error": str(e)}

    async def _generate_refinement_plan(self)->Dict[str, Any]:
        msg=[
            {
                "role":"system",
                "content":"Eres un experto en refinamiento de queries."
            },
            {
                "role":"user",
                "content":f"Analiza esta tarea:\n{self.task}"
            }
        ]
        resp=await self.llm.agenerate([msg])
        content=resp.generations[0][0].message.content.strip()
        try:
            # Esperamos JSON
            return json.loads(content)
        except:
            return {"analysis": content}
