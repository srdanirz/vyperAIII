# agents/coordination_agent.py

import logging
import asyncio
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from llm_factory import get_llm

logger = logging.getLogger(__name__)

class CoordinationAgent(BaseAgent):
    """
    Agente que coordina y genera una respuesta final, integrando
    resultados de varios sub-agentes.
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
        self.llm = get_llm(engine_mode, openai_api_key, model="gpt-4-turbo", temperature=0.2)

    async def _execute(self) -> Dict[str, Any]:
        try:
            # Disparamos varias subtareas en paralelo: resumen, detalle, ...
            tasks = [
                self._generate_concise_response(),
                self._generate_detailed_response(),
                self._extract_key_findings(),
                self._generate_collaboration_dialogue()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {
                "concise_response": results[0] if not isinstance(results[0], Exception) else "Error",
                "detailed_response": results[1] if not isinstance(results[1], Exception) else "Error",
                "key_findings": results[2] if not isinstance(results[2], Exception) else {},
                "collaboration_dialogue": results[3] if not isinstance(results[3], Exception) else {}
            }
        except Exception as e:
            logger.error(f"Error in CoordinationAgent: {e}")
            return {"error": str(e)}

    async def _generate_concise_response(self) -> str:
        messages = [
            {"role":"system","content":"Genera un resumen breve del estado actual."},
            {"role":"user","content": str(self.shared_data)}
        ]
        try:
            resp = await self.llm.agenerate([messages])
            return resp.generations[0][0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating concise response: {e}")
            return f"Error: {e}"

    async def _generate_detailed_response(self) -> str:
        messages = [
            {"role":"system","content":"Genera una explicación detallada con ejemplos y contexto."},
            {"role":"user","content": str(self.shared_data)}
        ]
        try:
            resp = await self.llm.agenerate([messages])
            return resp.generations[0][0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating detailed response: {e}")
            return f"Error: {e}"

    async def _extract_key_findings(self) -> Dict[str, Any]:
        messages = [
            {"role":"system","content":"Extrae conclusiones principales de la info."},
            {"role":"user","content": str(self.shared_data)}
        ]
        try:
            resp = await self.llm.agenerate([messages])
            text = resp.generations[0][0].message.content.strip()
            return {"key_points": text.split("\n")}
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return {"error": str(e)}

    async def _generate_collaboration_dialogue(self) -> Dict[str,Any]:
        messages = [
            {"role":"system","content":"Simula un diálogo colaborativo de todos los agentes involucrados."},
            {"role":"user","content": str(self.shared_data)}
        ]
        try:
            resp = await self.llm.agenerate([messages])
            return {"dialogue": resp.generations[0][0].message.content.strip()}
        except Exception as e:
            logger.error(f"Error generating collaboration dialogue: {e}")
            return {"error": str(e)}
