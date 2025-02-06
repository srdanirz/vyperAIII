# agents/agent_communication.py

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

class AgentCommunicationSystem:
    """
    Sistema central de comunicación entre agentes.
    Implementa:
      - Cola de mensajes asíncrona
      - Priorización de mensajes
      - Logs estructurados
      - Simulación de diálogo colaborativo
    """
    def __init__(self, api_key: str, engine_mode: str = "openai"):
        self.api_key = api_key
        self.engine_mode = engine_mode
        self.message_queue = asyncio.Queue()
        
        # Crear LLM según el modo
        if engine_mode == "deepseek":
            from deepseek_chat import DeepSeekChat
            self.llm = DeepSeekChat(api_key=api_key)
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4-turbo",
                temperature=0.7
            )
        
        # Prioridades predefinidas
        self.priorities = {
            "System": 0,
            "Tech Director": 1,
            "Research Director": 2,
            "Analysis Director": 2,
            "Content Director": 2,
            "Quality Director": 2
        }
        
        # Historial de mensajes
        self.message_history: List[Dict[str, Any]] = []
        
        # Estado de la comunicación
        self.active_agents: Set[str] = set()
        self.conversation_context: Dict[str, Any] = {}

    async def send_message(
        self,
        from_agent: str,
        content: str,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Envía un mensaje al sistema de comunicación.
        """
        try:
            # Determinar prioridad
            if priority is None:
                priority = self.priorities.get(from_agent, 3)
            
            # Limpiar contenido
            clean_content = self._clean_content(content)
            
            # Crear mensaje
            message = {
                "id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "timestamp": datetime.now().isoformat(),
                "from_agent": from_agent,
                "content": clean_content,
                "priority": priority,
                "metadata": metadata or {},
                "engine_mode": self.engine_mode
            }
            
            # Encolar mensaje
            await self.message_queue.put(message)
            
            # Guardar en historial
            self.message_history.append(message)
            
            # Log
            logger.debug(
                f"Message from [{from_agent}] p={priority}: {clean_content[:100]}..."
                if len(clean_content) > 100
                else f"Message from [{from_agent}] p={priority}: {clean_content}"
            )
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    def _clean_content(self, content: str) -> str:
        """Limpia y formatea el contenido del mensaje."""
        try:
            # Eliminar formateo innecesario
            clean = (content
                    .replace("[", "")
                    .replace("]", "")
                    .replace("**", "")
                    .replace("*", "")
                    .strip())
            
            # Normalizar espacios
            clean = " ".join(clean.split())
            
            return clean
            
        except Exception as e:
            logger.error(f"Error cleaning content: {e}")
            return content

    async def get_messages(
        self,
        filter_agent: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene mensajes del sistema, opcionalmente filtrados.
        """
        try:
            messages = []
            
            # Obtener todos los mensajes de la cola
            while not self.message_queue.empty():
                msg = await self.message_queue.get()
                messages.append(msg)
            
            # Filtrar si es necesario
            if filter_agent:
                messages = [
                    msg for msg in messages
                    if msg["from_agent"] == filter_agent
                ]
            
            # Ordenar por prioridad y timestamp
            messages.sort(key=lambda x: (x["priority"], x["timestamp"]))
            
            # Limitar si es necesario
            if limit:
                messages = messages[:limit]
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []

    async def generate_dialogue(
        self,
        task: str,
        participants: List[str]
    ) -> List[Dict[str, str]]:
        """
        Genera un diálogo simulado entre agentes.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Tarea: {task}\n"
                        f"Participantes: {', '.join(participants)}\n"
                        "Genera un diálogo colaborativo profesional entre los participantes."
                    )
                },
                {
                    "role": "user",
                    "content": task
                }
            ]
            
            response = await self.llm.agenerate([messages])
            text_response = response.generations[0][0].message.content
            
            # Parsear diálogo
            dialogue = []
            current_speaker = None
            current_content = []
            
            for line in text_response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    # Nuevo hablante
                    if current_speaker and current_content:
                        dialogue.append({
                            "speaker": current_speaker,
                            "content": " ".join(current_content)
                        })
                    parts = line.split(':', 1)
                    current_speaker = parts[0].strip()
                    current_content = (
                        [parts[1].strip()] if len(parts) > 1 else []
                    )
                else:
                    if current_speaker:
                        current_content.append(line)
            
            # Agregar último diálogo
            if current_speaker and current_content:
                dialogue.append({
                    "speaker": current_speaker,
                    "content": " ".join(current_content)
                })
            
            return dialogue
            
        except Exception as e:
            logger.error(f"Error generating dialogue: {e}")
            return []

    async def broadcast_message(
        self,
        content: str,
        exclude_agents: Optional[List[str]] = None
    ) -> None:
        """
        Envía un mensaje a todos los agentes activos.
        """
        try:
            exclude = set(exclude_agents or [])
            for agent in self.active_agents:
                if agent not in exclude:
                    await self.send_message(
                        "System",
                        f"[Broadcast] {content}",
                        priority=0
                    )
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

    async def cleanup(self) -> None:
        """
        Limpia recursos y realiza tareas de finalización.
        """
        try:
            # Vaciar cola de mensajes
            while not self.message_queue.empty():
                await self.message_queue.get()
            
            # Limpiar estado
            self.active_agents.clear()
            self.conversation_context.clear()
            
            # Log final
            logger.info("Communication system cleaned up")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen de la conversación.
        """
        try:
            return {
                "total_messages": len(self.message_history),
                "active_agents": list(self.active_agents),
                "last_message": (
                    self.message_history[-1] if self.message_history else None
                ),
                "conversation_context": self.conversation_context
            }
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return {}