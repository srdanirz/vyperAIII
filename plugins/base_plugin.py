# plugins/base_plugin.py

import logging
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from functools import wraps

logger = logging.getLogger(__name__)

def hook(hook_name: str) -> Callable:
    """Decorador para marcar una función como hook."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        setattr(wrapper, '_hook', hook_name)
        return wrapper
    return decorator

class BasePlugin(ABC):
    """
    Clase base para plugins.
    
    Define la interfaz que deben implementar todos los plugins:
    - setup: Configuración inicial
    - execute: Ejecución principal
    - cleanup: Limpieza
    """
    
    def __init__(self, name: str):
        self.name = name
        self.config: Dict[str, Any] = {}
        self.status = "initialized"
        self.metrics: Dict[str, Any] = {
            "executions": 0,
            "errors": 0,
            "average_time": 0
        }

    @abstractmethod
    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Configura el plugin.
        
        Args:
            config: Configuración del plugin
        """
        self.config = config
        self.status = "setup"

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Ejecuta la funcionalidad principal del plugin.
        
        Returns:
            Resultado de la ejecución
        """
        self.metrics["executions"] += 1
        self.status = "executing"

    @abstractmethod
    async def cleanup(self) -> None:
        """Limpia recursos del plugin."""
        self.status = "cleaned"

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del plugin."""
        return {
            "name": self.name,
            "status": self.status,
            "config": self.config,
            "metrics": self.metrics
        }

    def _update_metrics(self, execution_time: float) -> None:
        """Actualiza métricas del plugin."""
        self.metrics["average_time"] = (
            (self.metrics["average_time"] * self.metrics["executions"] + execution_time) /
            (self.metrics["executions"] + 1)
        )

    def _log_error(self, error: Exception) -> None:
        """Registra un error del plugin."""
        self.metrics["errors"] += 1
        logger.error(f"Plugin {self.name} error: {error}")

# Ejemplo de uso:
"""
class MyPlugin(BasePlugin):
    def __init__(self):
        super().__init__("my_plugin")
    
    async def setup(self, config: Dict[str, Any]) -> None:
        await super().setup(config)
        # Configuración específica
    
    @hook("pre_process")
    async def pre_process(self, data: Any) -> Any:
        # Hook de preprocesamiento
        return processed_data
    
    async def execute(self, *args, **kwargs) -> Any:
        await super().execute(*args, **kwargs)
        # Lógica principal
        return result
    
    @hook("post_process")
    async def post_process(self, result: Any) -> Any:
        # Hook de postprocesamiento
        return processed_result
    
    async def cleanup(self) -> None:
        await super().cleanup()
        # Limpieza específica
"""
