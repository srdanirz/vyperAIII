import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
import asyncio

logger = logging.getLogger(__name__)

class AgentCommunicationSystem:
    """
    Centraliza la comunicación entre distintos agentes.
    Implementa una cola de mensajes con prioridades básicas y
    un mecanismo para generar "diálogos" simulados de colaboración.
    """
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.7
        )
        self.message_queue = asyncio.Queue()
        self.priorities = {
            "System": 0,
            "Team": 1
        }

    async def send_message(self, from_agent: str, content: str, priority: Optional[int] = None) -> None:
        """Envía un mensaje al sistema de comunicación con prioridad."""
        if priority is None:
            priority = self.priorities.get(from_agent, 2)
        clean_content = self._clean_content(content)
        
        await self.message_queue.put({
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "from_agent": from_agent,
            "content": clean_content
        })
        logger.debug(f"Message sent from [{from_agent}] with priority [{priority}]: {clean_content}")

    def _clean_content(self, content: str) -> str:
        """Limpia el contenido de marcadores especiales."""
        return (content
                .replace('[', '')
                .replace(']', '')
                .replace('**', '')
                .replace('*', '')
                .strip())

    async def generate_dialogue(self, task: str, participants: List[str]) -> List[Dict[str, str]]:
        """
        Genera un diálogo que muestre la colaboración de agentes
        con base en un 'task' y una lista de participantes.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"Task: {task}\n"
                    f"Team members: {', '.join(participants)}\n\n"
                    "Generate a solution where each team member contributes "
                    "their expertise to create exactly what was requested. "
                    "Show the actual development and creation of the solution, "
                    "not just planning."
                )
            },
            {"role": "user", "content": task}
        ]
        
        response = await self.llm.agenerate([messages])
        text_response = response.generations[0][0].message.content
        
        dialogue = []
        current_speaker = None
        current_content = []
        
        for line in text_response.split('\n'):
            line = self._clean_content(line)
            if not line:
                continue
            if ':' in line:
                # Se detecta cambio de hablante
                if current_speaker and current_content:
                    dialogue.append({
                        "speaker": current_speaker,
                        "content": ' '.join(current_content)
                    })
                    current_content = []
                parts = line.split(':', 1)
                current_speaker = parts[0].strip()
                if len(parts) > 1:
                    current_content = [parts[1].strip()]
            elif current_speaker:
                current_content.append(line)
        
        # Añadir el último bloque si quedó pendiente
        if current_speaker and current_content:
            dialogue.append({
                "speaker": current_speaker,
                "content": ' '.join(current_content)
            })
        
        return dialogue

    async def get_messages(self) -> List[Dict[str, Any]]:
        """Obtiene todos los mensajes en orden de prioridad y tiempo."""
        messages = []
        while not self.message_queue.empty():
            messages.append(await self.message_queue.get())
        messages.sort(key=lambda x: (x["priority"], x["timestamp"]))
        return messages
