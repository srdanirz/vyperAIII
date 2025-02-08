# agents/analysis_agent.py

import logging
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from core.llm import get_llm

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """
    Agente que analiza datos, usando un LLM para sintetizar conclusiones
    (si es que se requiere).
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        engine_mode = self.metadata.get("engine_mode", "openai")
        self.llm = get_llm(engine_mode, openai_api_key, model="gpt-4", temperature=0)

    async def _execute(self) -> Dict[str, Any]:
        try:
            # Ejemplo: usamos self.shared_data como "datos"
            data_str = str(self.shared_data.get("some_data", "No data available"))
            messages = [
                {"role":"system","content":"Eres un experto en análisis de datos."},
                {"role":"user","content":f"Analiza estos datos:\n{data_str}"}
            ]
            resp = await self.llm.agenerate([messages])
            analysis_text = resp.generations[0][0].message.content

            return {
                "analysis_result": analysis_text
            }
        except Exception as e:
            logger.error(f"Error in AnalysisAgent: {e}")
            return {"error": str(e)}
