import logging
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from capsolver import Capsolver
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)
load_dotenv()

class CapsolverAgent(BaseAgent):
    """
    Agente para resolver CAPTCHAs usando el servicio de Capsolver.
    """
    def __init__(self, task: str, openai_api_key: str, shared_data: Dict[str, Any] = None, metadata: dict = None):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.capsolver_key = os.getenv("CAPSOLVER_API_KEY")

    async def _execute(self) -> Dict[str, Any]:
        """Ejecuta la resolución de CAPTCHA."""
        if not self.capsolver_key:
            return {
                "error": "Missing CAPSOLVER_API_KEY environment variable",
                "metadata": self.metadata
            }
        try:
            config = self._parse_task()
            if "error" in config:
                config["metadata"] = self.metadata
                return config
            solution = await self._solve_captcha(config)
            solution["metadata"] = self.metadata
            return solution

        except Exception as e:
            logger.error(f"Error in CapsolverAgent: {e}", exc_info=True)
            return {
                "error": str(e),
                "metadata": self.metadata
            }

    def _parse_task(self) -> Dict[str, Any]:
        """Parsea la configuración de CAPTCHA desde la task."""
        try:
            cfg = json.loads(self.task)
            if "type" not in cfg:
                return {"error": "Missing 'type' in CAPTCHA configuration"}
            return cfg
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in task configuration"}

    async def _solve_captcha(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve el CAPTCHA usando Capsolver."""
        client = Capsolver(api_key=self.capsolver_key)
        try:
            task_obj = self._prepare_task(config)
            result = client.solve(task_obj)
            return {
                "status": "success",
                "solution": result.get("solution", {}),
                "timing": {
                    "start_time": result.get("startTime"),
                    "end_time": result.get("endTime")
                }
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def _prepare_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara el objeto de task para Capsolver."""
        task = {
            "type": config["type"],
            "websiteURL": config.get("websiteURL", ""),
            "websiteKey": config.get("websiteKey", "")
        }
        optional_params = ["proxy", "userAgent", "cookies", "imageBase64", "minScore"]
        for param in optional_params:
            if param in config:
                task[param] = config[param]
        return task
