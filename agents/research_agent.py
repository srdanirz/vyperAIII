import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Agente especializado en investigación y síntesis de información
    de múltiples fuentes.
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
            model="gpt-4-turbo",
            temperature=0.3
        )

    async def _execute(self) -> Dict[str, Any]:
        """Realiza la investigación y genera un resultado en 'research_findings'."""
        try:
            context = self._prepare_research_context()
            messages = [
                {"role": "system", "content": "Eres un especialista en investigación de temas diversos."},
                {"role": "user", "content": f"Task: {self.task}\nContext: {context}"}
            ]
            response = await self.llm.agenerate([messages])
            return {
                "research_findings": response.generations[0][0].message.content
            }
        except Exception as e:
            logger.error(f"Error in ResearchAgent: {e}", exc_info=True)
            return {"error": str(e)}

    def _prepare_research_context(self) -> str:
        """Prepara contexto con resultados previos en shared_data."""
        parts = []
        for key, data in self.shared_data.items():
            if isinstance(data, dict) and "error" not in data:
                parts.append(f"{key}: {str(data)}")
        return "\n".join(parts)
