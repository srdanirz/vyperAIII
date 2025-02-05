import logging
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RefinerAgent(BaseAgent):
    """
    Agente para refinar queries y mejorar estrategias de búsqueda.
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
        """Genera un plan de refinamiento basado en la task."""
        try:
            return await self._generate_refinement_plan()
        except Exception as e:
            logger.error(f"Error in RefinerAgent: {e}", exc_info=True)
            return {"error": str(e)}

    async def _generate_refinement_plan(self) -> Dict[str, Any]:
        prompt = (
            f"Analiza esta tarea y provee un plan detallado de refinamiento:\n"
            f"'{self.task}'"
        )
        # Similar a antes, devuelves un JSON con la estrategia
        messages = [
            {
                "role": "system", 
                "content": (
                    "Eres un experto en optimización de búsqueda. Retorna la respuesta "
                    "como JSON con 'search_type', 'sources', 'data_points', 'search_constraints', "
                    "'expected_format' y 'analysis_notes'."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm.agenerate([messages])
        content = response.generations[0][0].message.content.strip()
        try:
            # Extracción de JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                "error": "Invalid JSON from the LLM",
                "raw_response": content
            }