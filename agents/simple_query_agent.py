import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SimpleQueryAgent(BaseAgent):
    """
    Agente para manejar queries simples y directas (fechas, hora, definiciones cortas, etc.).
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
            model="gpt-3.5-turbo",
            temperature=0
        )

    async def _execute(self) -> Dict[str, Any]:
        """Determina si es query predefinida o requiere LLM."""
        try:
            direct_response = self._handle_predefined_query()
            if direct_response:
                return {"response": direct_response, "query_type": "predefined"}
            
            return await self._handle_dynamic_query()
        except Exception as e:
            logger.error(f"Error in SimpleQueryAgent: {e}", exc_info=True)
            return {"error": str(e)}

    def _handle_predefined_query(self) -> Optional[str]:
        """Maneja queries básicas sin necesidad de LLM (hora, fecha, etc.)."""
        task_lower = self.task.lower()
        current_date = datetime.now()

        if any(phrase in task_lower for phrase in ["qué hora es", "dime la hora"]):
            return f"Son las {current_date.strftime('%H:%M:%S')}"

        if any(phrase in task_lower for phrase in ["qué día es", "fecha de hoy"]):
            days_es = {
                'Monday': 'Lunes','Tuesday': 'Martes','Wednesday': 'Miércoles',
                'Thursday': 'Jueves','Friday': 'Viernes','Saturday': 'Sábado','Sunday': 'Domingo'
            }
            months_es = {
                'January': 'Enero','February': 'Febrero','March': 'Marzo','April': 'Abril',
                'May': 'Mayo','June': 'Junio','July': 'Julio','August': 'Agosto',
                'September': 'Septiembre','October': 'Octubre','November': 'Noviembre','December': 'Diciembre'
            }
            day_name = days_es[current_date.strftime('%A')]
            month_name = months_es[current_date.strftime('%B')]
            return f"Hoy es {day_name}, {current_date.day} de {month_name} de {current_date.year}"
        
        # Cálculos básicos
        if "cuánto es" in task_lower or "calcula" in task_lower:
            try:
                expression = re.split(r"cuánto es|calcula", task_lower)[-1].strip()
                result = eval(expression)
                return f"El resultado es {result}"
            except:
                return None
        
        return None

    async def _handle_dynamic_query(self) -> Dict[str, Any]:
        """Resuelve queries usando el LLM."""
        system_prompt = (
            "Eres un asistente útil que brinda respuestas concisas a preguntas simples. "
            "Si no estás seguro, dilo directamente."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.task}
        ]
        response = await self.llm.agenerate([messages])
        answer = response.generations[0][0].message.content.strip()
        return {
            "response": answer,
            "query_type": "dynamic"
        }
