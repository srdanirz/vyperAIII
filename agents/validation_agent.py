import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent
import base64

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """
    Agente que valida los resultados de otros agentes (formato, consistencia, etc.).
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        shared_data: Dict[str, Any] = None,
        metadata: dict = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )

    async def _execute(self) -> Dict[str, Any]:
        """Valida los resultados presentes en shared_data."""
        try:
            validation_results = {}
            for key, data in self.shared_data.items():
                validation = await self._validate_result(key, data)
                validation_results[key] = validation
            
            overall_status = self._compute_overall_status(validation_results)
            return {
                "validation_summary": validation_results,
                "overall_status": overall_status
            }
        except Exception as e:
            logger.error(f"Error in ValidationAgent: {e}", exc_info=True)
            return {"error": str(e)}

    async def _validate_result(self, key: str, data: Any) -> Dict[str, Any]:
        """Valida un resultado específico usando el LLM."""
        prepared_data = self._prepare_data(data)
        messages = [
            {
                "role": "system",
                "content": (
                    "Analiza este contenido y determina si está completo, "
                    "correcto y con un formato válido. Responde de manera concisa."
                )
            },
            {
                "role": "user",
                "content": f"Validar {key}: {str(prepared_data)}"
            }
        ]
        response = await self.llm.agenerate([messages])
        text = response.generations[0][0].message.content.strip()
        is_valid = not any(w in text.lower() for w in ["error", "invalid", "incomplete"])
        return {
            "valid": is_valid,
            "notes": text
        }

    def _prepare_data(self, data: Any) -> Any:
        """Recorta o identifica data si es muy grande o binaria."""
        if isinstance(data, dict) and "content" in data:
            # si es base64
            if isinstance(data["content"], str) and self._looks_base64(data["content"]):
                return {"base64_content": f"{len(data['content'])} chars"}
        return data

    def _looks_base64(self, s: str) -> bool:
        try:
            base64.b64decode(s)
            return True
        except Exception:
            return False

    def _compute_overall_status(self, validation_results: Dict[str, Any]) -> str:
        if not validation_results:
            return "NO_DATA"
        all_valid = all(v.get("valid") for v in validation_results.values())
        return "PASSED" if all_valid else "FAILED"
