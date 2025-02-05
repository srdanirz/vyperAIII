# coordination_agent.py
import logging
import asyncio
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CoordinationAgent(BaseAgent):
    """
    Agent specialized in coordinating complex workflows, managing dependencies,
    and providing final synthesized responses
    """
    def __init__(self, task: str, openai_api_key: str, partial_data: Dict[str, Any] = None, metadata: dict = None):
        super().__init__(task, metadata)
        self.openai_api_key = openai_api_key
        self.partial_data = partial_data or {}
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.2,
            max_tokens=2000
        )

    async def execute(self) -> Dict[str, Any]:
        """Coordinate workflow and synthesize final response"""
        try:
            # Extract relevant information first
            simple_query_info = self._get_agent_info('simple_query')
            research_info = self._get_agent_info('research')
            validation_info = self._get_agent_info('validation')

            # Procesar tareas en paralelo
            tasks = [
                self._generate_concise_response(),
                self._generate_detailed_response(),
                self._extract_key_findings(),
                self._generate_collaboration_dialogue()
            ]

            # Ejecutar todas las tareas en paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extraer resultados
            concise_response, detailed_response, key_findings, final_dialogue = results

            # Verificar errores
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in coordination task: {result}")

            return {
                "final_response": concise_response if not isinstance(concise_response, Exception) else "Error generating concise response",
                "detailed_response": detailed_response if not isinstance(detailed_response, Exception) else "Error generating detailed response",
                "key_findings": key_findings if not isinstance(key_findings, Exception) else {"error": "Error extracting key findings"},
                "collaboration_dialogue": final_dialogue if not isinstance(final_dialogue, Exception) else {"error": "Error generating dialogue"},
                "metadata": self.metadata,
                "source_info": {
                    "simple_query": simple_query_info,
                    "research": research_info,
                    "validation": validation_info
                }
            }

        except Exception as e:
            logger.error(f"Error in CoordinationAgent: {e}")
            return {
                "error": f"Coordination error: {str(e)}",
                "metadata": self.metadata
            }

    def _analyze_current_state(self) -> Dict[str, Any]:
        """Analyze current state of all collected information"""
        completed_results = {}
        error_results = {}
        
        for key, data in self.partial_data.items():
            if isinstance(data, dict):
                if "error" not in data:
                    completed_results[key] = data
                else:
                    error_results[key] = data

        return {
            "completed_tasks": list(completed_results.keys()),
            "error_tasks": list(error_results.keys()),
            "total_tasks": len(self.partial_data),
            "success_rate": len(completed_results) / len(self.partial_data) if self.partial_data else 0
        }

    def _get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Extract information from a specific agent type"""
        for key, value in self.partial_data.items():
            if agent_type in key.lower() and isinstance(value, dict):
                return value
        return {}

    async def _generate_collaboration_dialogue(self) -> Dict[str, Any]:
        """Generate a dialogue showing how agents collaborated"""
        messages = [
            {
                "role": "system",
                "content": """You are a coordinator synthesizing a conversation between AI agents. 
                Create a natural dialogue that shows how the agents collaborated and built upon each 
                other's findings to reach the final conclusion."""
            },
            {
                "role": "user",
                "content": f"Based on these results and interactions:\n{str(self.partial_data)}\n\nCreate a dialogue showing how the agents collaborated."
            }
        ]

        try:
            response = await self.llm.agenerate([messages])
            return {
                "dialogue": response.generations[0][0].message.content,
                "participants": self._get_participating_agents(),
                "timestamp": self.metadata.get("timestamp", "unknown")
            }
        except Exception as e:
            logger.error(f"Error generating dialogue: {e}")
            return {"error": str(e)}

    async def _generate_detailed_response(self) -> str:
        """Generate a detailed, comprehensive response"""
        messages = [
            {
                "role": "system",
                "content": """Create a detailed, well-structured response that comprehensively 
                explains the topic. Include relevant details, examples, and context while maintaining 
                clarity and accuracy."""
            },
            {
                "role": "user",
                "content": f"Based on all collected information:\n{str(self.partial_data)}\n\nProvide a detailed explanation."
            }
        ]

        try:
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating detailed response: {e}")
            return "Error generating detailed response"

    async def _generate_concise_response(self) -> str:
        """Generate a clear, concise response"""
        messages = [
            {
                "role": "system",
                "content": """Create a clear, concise response that captures the essential 
                information. Focus on the most important points while ensuring the response 
                is easy to understand."""
            },
            {
                "role": "user",
                "content": f"Based on this information:\n{str(self.partial_data)}\n\nProvide a clear, concise explanation."
            }
        ]

        try:
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating concise response: {e}")
            return "Error generating concise response"

    async def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings and insights"""
        messages = [
            {
                "role": "system",
                "content": """Analyze the collected information and extract key findings, 
                main points, and important insights. Structure the response in a clear, 
                organized manner."""
            },
            {
                "role": "user",
                "content": f"Extract key findings from:\n{str(self.partial_data)}"
            }
        ]

        try:
            response = await self.llm.agenerate([messages])
            return {
                "key_points": response.generations[0][0].message.content.strip().split("\n"),
                "timestamp": self.metadata.get("timestamp", "unknown")
            }
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return {"error": str(e)}

    def _get_participating_agents(self) -> list:
        """Get list of agents that participated in the task"""
        agents = set()
        for key in self.partial_data.keys():
            agent = key.split('_')[0]
            if agent:
                agents.add(agent)
        return list(agents)