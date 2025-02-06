# agents/research_agent.py

import logging
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from llm_factory import get_llm

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Agente que investiga un tema usando un LLM.
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]]=None,
        shared_data: Optional[Dict[str,Any]]=None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        engine_mode = self.metadata.get("engine_mode", "openai")
        self.llm = get_llm(engine_mode, openai_api_key, model="gpt-4-turbo", temperature=0.3)

    async def _execute(self) -> Dict[str, Any]:
        try:
            context = f"Context: {str(self.shared_data)}"
            messages = [
                {"role":"system","content":"Eres un especialista en investigación."},
                {"role":"user","content":f"Investiga sobre: {self.task}\n{context}"}
            ]
            resp = await self.llm.agenerate([messages])
            text = resp.generations[0][0].message.content
            return {"research_findings": text}
        except Exception as e:
            logger.error(f"Error in ResearchAgent: {e}", exc_info=True)
            return {"error":str(e)}
