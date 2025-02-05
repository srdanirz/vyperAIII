# agents/capsolver_agent.py
import logging
import os
import json
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from capsolver import Capsolver
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)
load_dotenv()

class CapsolverAgent(BaseAgent):
    """
    Agent for handling CAPTCHA solving tasks
    """
    def __init__(self, task: str, openai_api_key: str, partial_data: dict = None, metadata: dict = None):
        super().__init__(task, metadata)
        self.openai_api_key = openai_api_key
        self.partial_data = partial_data or {}
        self.capsolver_key = os.getenv("CAPSOLVER_API_KEY")

    async def execute(self) -> Dict[str, Any]:
        """Execute CAPTCHA solving task"""
        if not self.capsolver_key:
            return {
                "error": "Missing CAPSOLVER_API_KEY environment variable",
                "metadata": self.metadata
            }

        try:
            captcha_config = self._parse_task()
            if "error" in captcha_config:
                captcha_config["metadata"] = self.metadata
                return captcha_config

            solution = await self._solve_captcha(captcha_config)
            solution["metadata"] = self.metadata
            return solution

        except Exception as e:
            logger.error(f"Error in CapsolverAgent: {e}")
            return {
                "error": str(e),
                "metadata": self.metadata
            }

    def _parse_task(self) -> Dict[str, Any]:
        """Parse the task configuration"""
        try:
            config = json.loads(self.task)
            if "type" not in config:
                return {"error": "Missing 'type' in CAPTCHA configuration"}
            return config
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in task configuration"}

    async def _solve_captcha(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the CAPTCHA using Capsolver"""
        client = Capsolver(api_key=self.capsolver_key)
        
        try:
            task = self._prepare_task(config)
            result = client.solve(task)
            
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
        """Prepare the task for Capsolver"""
        task = {
            "type": config["type"],
            "websiteURL": config.get("websiteURL", ""),
            "websiteKey": config.get("websiteKey", "")
        }

        # Add optional parameters if present
        optional_params = ["proxy", "userAgent", "cookies", "imageBase64", "minScore"]
        for param in optional_params:
            if param in config:
                task[param] = config[param]

        return task