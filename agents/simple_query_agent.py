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
    Agent specialized in handling simple, direct queries that don't require
    complex orchestration or multiple agent involvement.
    """

    def __init__(self, task: str, openai_api_key: str, partial_data: dict = None, metadata: dict = None):
        super().__init__(task, metadata)
        self.openai_api_key = openai_api_key
        self.partial_data = partial_data or {}
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )

    async def execute(self) -> Dict[str, Any]:
        """Execute the simple query task"""
        try:
            # First check if it's a predefined query that doesn't need LLM
            direct_response = self._handle_predefined_query()
            if direct_response:
                return {"response": direct_response}

            # If not predefined, use LLM for simple but dynamic queries
            return await self._handle_dynamic_query()

        except Exception as e:
            logger.error(f"Error in SimpleQueryAgent: {e}")
            return {"error": str(e)}

    def _handle_predefined_query(self) -> Optional[str]:
        """Handle predefined queries that don't need LLM"""
        task_lower = self.task.lower()
        current_date = datetime.now()

        # Time and date queries
        if any(phrase in task_lower for phrase in ["qué hora es", "dime la hora"]):
            return f"Son las {current_date.strftime('%H:%M:%S')}"
        
        if any(phrase in task_lower for phrase in ["qué día es", "fecha de hoy"]):
            # Format date based on detected language
            if "what" in task_lower:
                return f"Today is {current_date.strftime('%A, %B %d, %Y')}"
            
            # Traducción de días y meses al español
            days = {
                'Monday': 'Lunes',
                'Tuesday': 'Martes',
                'Wednesday': 'Miércoles',
                'Thursday': 'Jueves',
                'Friday': 'Viernes',
                'Saturday': 'Sábado',
                'Sunday': 'Domingo'
            }
            months = {
                'January': 'Enero',
                'February': 'Febrero',
                'March': 'Marzo',
                'April': 'Abril',
                'May': 'Mayo',
                'June': 'Junio',
                'July': 'Julio',
                'August': 'Agosto',
                'September': 'Septiembre',
                'October': 'Octubre',
                'November': 'Noviembre',
                'December': 'Diciembre'
            }
            
            day_name = days[current_date.strftime('%A')]
            month_name = months[current_date.strftime('%B')]
            return f"Hoy es {day_name}, {current_date.day} de {month_name} de {current_date.year}"
        
        # Basic calculations
        if "cuánto es" in task_lower or "calcula" in task_lower:
            try:
                # Extract numbers and operation
                expression = task_lower.split("es")[-1].strip()
                result = eval(expression)
                return f"El resultado es {result}"
            except:
                return None

        return None

    async def _handle_dynamic_query(self) -> Dict[str, Any]:
        """Handle simple but dynamic queries using LLM"""
        system_prompt = """You are a helpful assistant that provides direct, concise answers to simple questions. 
        Keep responses brief and to the point. If you're not sure about something, say so directly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.task}
        ]

        try:
            response = await self.llm.agenerate([messages])
            answer = response.generations[0][0].message.content.strip()
            
            return {
                "response": answer,
                "query_type": "dynamic"
            }
        except Exception as e:
            logger.error(f"Error in dynamic query handling: {e}")
            return {"error": f"Failed to process query: {str(e)}"}

    @staticmethod
    def can_handle_query(prompt: str) -> bool:
        """
        Determine if this agent can handle a given prompt
        """
        # Patterns for queries this agent can handle
        simple_patterns = [
            # Time and date
            r"^¿qué\s+(hora|día|fecha|tiempo|año)\s+es",  
            r"^qué\s+(hora|día|fecha|tiempo|año)\s+es",
            r"^dime\s+(la\s+)?(hora|fecha|día)",
            
            # Basic calculations  
            r"^(cuánto|cuanto)\s+es\s+[\d\s\+\-\*\/]+",
            r"^calcula\s+[\d\s\+\-\*\/]+",
            
            # Simple factual questions
            r"^¿(cuál|cual|qué|que)\s+es\s+(el|la|los|las)\s+", 
            r"^(cuál|cual|qué|que)\s+es\s+(el|la|los|las)\s+",
            
            # Basic conversions
            r"^convierte\s+\d+\s+\w+\s+a\s+\w+", 
            r"^¿cuántos|cuantos\s+\w+\s+hay\s+en\s+\d+\s+\w+",
            
            # Simple definitions 
            r"^¿qué|que\s+significa\s+",
            r"^define\s+",
            
            # Yes/No questions
            r"^¿es\s+verdad\s+que",
            r"^¿es\s+cierto\s+que",
        ]

        prompt_lower = prompt.lower()
        return any(re.match(pattern, prompt_lower) for pattern in simple_patterns)