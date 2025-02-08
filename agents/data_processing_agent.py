# agents/data_processing_agent.py

import logging
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from core.llm import get_llm

logger = logging.getLogger(__name__)

class DataProcessingAgent(BaseAgent):
    """
    Agente para limpiar/transformar datos y/o describirlos usando un LLM.
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]]=None,
        shared_data: Optional[Dict[str,Any]]=None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        engine_mode = self.metadata.get("engine_mode","openai")
        self.llm = get_llm(engine_mode, openai_api_key, "gpt-4-turbo", 0.7)

    async def _execute(self) -> Dict[str, Any]:
        try:
            # Ejemplo: supón que hace limpiezas, transformaciones, etc. 
            # Aquí sólo simulamos pedirle a un LLM que explique la transform
            data_preview = str(self.shared_data.get("raw_data","No data"))
            messages = [
                {"role":"system","content":"Eres un experto en procesamiento de datos."},
                {"role":"user","content":f"Procesa estos datos:\n{data_preview}"}
            ]
            resp = await self.llm.agenerate([messages])
            text = resp.generations[0][0].message.content

            return {"processed_data": text}
        except Exception as e:
            logger.error(f"Error in DataProcessingAgent: {e}")
            return {"error": str(e)}
