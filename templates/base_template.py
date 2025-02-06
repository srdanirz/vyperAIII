# templates/base_template.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseTemplate(ABC):
    """Clase base para todos los templates"""
    
    def __init__(self, template_name: str, config: Dict[str, Any]):
        self.template_name = template_name
        self.config = config
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "template_version": "1.0.0",
            "last_modified": datetime.now().isoformat()
        }

    @abstractmethod
    async def apply(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica el template al contenido"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Valida que el template tenga toda la información necesaria"""
        pass