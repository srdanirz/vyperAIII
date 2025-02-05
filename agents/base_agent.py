import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Clase base para todos los agentes, que provee:
      - Atributos estándar (task, openai_api_key, metadata, shared_data)
      - Lógica de ejecución con temporización y manejo de errores
    """
    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        self.task = task
        self.openai_api_key = openai_api_key
        self.metadata = metadata or {}
        self.shared_data = shared_data or {}
        self.execution_start: Optional[datetime] = None
        self.execution_end: Optional[datetime] = None
        
    async def execute(self) -> Dict[str, Any]:
        """
        Ejecuta la lógica propia del agente en el método interno _execute().
        Gestiona el tiempo de ejecución y capturas de errores.
        """
        try:
            self.execution_start = datetime.now()
            result = await self._execute()

            # Aseguramos que el resultado sea un diccionario
            if not isinstance(result, dict):
                result = {"result": result}
            
            self.execution_end = datetime.now()
            execution_time = (self.execution_end - self.execution_start).total_seconds()
            
            return {
                **result,
                "metadata": {
                    **self.metadata,
                    "execution_time": execution_time,
                    "execution_start": self.execution_start.isoformat(),
                    "execution_end": self.execution_end.isoformat(),
                },
                "agent_type": self.__class__.__name__
            }
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "metadata": self.metadata,
                "agent_type": self.__class__.__name__
            }

    async def _execute(self) -> Dict[str, Any]:
        """Método que cada agente concreto debe implementar."""
        raise NotImplementedError("Each agent must implement _execute().")
