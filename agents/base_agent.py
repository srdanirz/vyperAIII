# agents/base_agent.py
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents with standard initialization"""
    def __init__(self, task: str, openai_api_key: str, metadata: Optional[Dict[str, Any]] = None, partial_data: Optional[Dict[str, Any]] = None):
        self.task = task
        self.openai_api_key = openai_api_key
        self.metadata = metadata or {}
        self.partial_data = partial_data or {}
        self.execution_start = None
        self.execution_end = None
        
    async def execute(self) -> Dict[str, Any]:
        """Execute agent task with lifecycle management"""
        try:
            self.execution_start = datetime.now()
            result = await self._execute()
            self.execution_end = datetime.now()
            
            if not isinstance(result, dict):
                result = {"result": result}
            
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
            logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            return {
                "error": str(e),
                "metadata": self.metadata,
                "agent_type": self.__class__.__name__
            }
            
    async def _execute(self) -> Dict[str, Any]:
        """To be implemented by each agent"""
        raise NotImplementedError("Each agent must implement _execute()")