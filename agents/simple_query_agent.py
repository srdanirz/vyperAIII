# agents/simple_query_agent.py

import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SimpleQueryAgent(BaseAgent):
    """
    Agente para queries simples (fecha, hora, definiciones cortas, etc.).
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
        try:
            direct = self._handle_predefined_query()
            if direct:
                return {"response": direct, "query_type":"predefined"}
            return await self._handle_dynamic_query()
        except Exception as e:
            logger.error(f"SimpleQueryAgent error: {e}", exc_info=True)
            return {"error": str(e)}

    def _handle_predefined_query(self)->Optional[str]:
        task_lower=self.task.lower()
        now=datetime.now()
        if "qué hora" in task_lower or "dime la hora" in task_lower:
            return f"Son las {now.strftime('%H:%M:%S')}"
        if "qué día es" in task_lower or "fecha de hoy" in task_lower:
            days_es={"Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles","Thursday":"Jueves",
                     "Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"}
            months_es={"January":"Enero","February":"Febrero","March":"Marzo","April":"Abril",
                       "May":"Mayo","June":"Junio","July":"Julio","August":"Agosto","September":"Septiembre",
                       "October":"Octubre","November":"Noviembre","December":"Diciembre"}
            day_name=days_es[now.strftime('%A')]
            month_name=months_es[now.strftime('%B')]
            return f"Hoy es {day_name}, {now.day} de {month_name} de {now.year}"
        if "cuánto es" in task_lower or "calcula" in task_lower:
            try:
                expr=re.split(r"cuánto es|calcula",task_lower)[-1].strip()
                val=eval(expr)
                return f"El resultado es {val}"
            except:
                pass
        return None

    async def _handle_dynamic_query(self)->Dict[str,Any]:
        sys_prompt="Eres un asistente simple, contesta brevemente."
        msg=[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":self.task}
        ]
        resp=await self.llm.agenerate([msg])
        ans=resp.generations[0][0].message.content.strip()
        return {"response": ans, "query_type":"dynamic"}
