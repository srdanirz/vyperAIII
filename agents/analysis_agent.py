import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """
    Agente encargado de realizar análisis sobre datos o información previa.
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4",
            temperature=0
        )

    async def _execute(self) -> Dict[str, Any]:
        """Ejecuta el flujo principal de análisis de datos."""
        try:
            data_to_analyze = self._extract_data_to_analyze()
            analysis_result = await self._analyze_data(data_to_analyze)
            
            # Se considera "complex" si la longitud del string es mayor a 100
            is_complex = False
            if isinstance(data_to_analyze, str):
                is_complex = len(data_to_analyze) > 100
            
            return {
                "analysis_result": analysis_result,
                "complexity": "complex" if is_complex else "simple",
                "data_analyzed": True
            }

        except Exception as e:
            logger.error(f"Error in AnalysisAgent: {e}", exc_info=True)
            raise

    def _extract_data_to_analyze(self) -> Any:
        """
        Extrae los datos para analizar desde shared_data o, en su defecto, usa la task.
        """
        if not self.shared_data:
            return self.task

        # Se busca en el shared_data algún resultado previo (por ejemplo, en 'research_result')
        possible_keys = ["result", "response", "content", "research_result"]
        for v in self.shared_data.values():
            if isinstance(v, dict):
                for pk in possible_keys:
                    if pk in v:
                        return v[pk]
        return self.task

    async def _analyze_data(self, data: Any) -> str:
        """Le pide a un LLM que analice los datos y devuelva conclusiones."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un experto analista de datos. Analiza la información "
                    "proporcionada de manera clara y con conclusiones concretas."
                )
            },
            {
                "role": "user",
                "content": f"Task: {self.task}\n\nDatos:\n{data}\n\nProvee un análisis detallado."
            }
        ]
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].message.content
