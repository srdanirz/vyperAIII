# config.py

from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
import logging

logger = logging.getLogger(__name__)

def get_config() -> Optional[Dict[str, Any]]:
    """
    Obtiene la configuración del sistema.
    """
    try:
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
                
        logger.warning("No se encontró archivo de configuración")
        return None
        
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return None