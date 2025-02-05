# agents/research_agent.py
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Agent specialized in deep research and information synthesis from multiple sources
    """
    def __init__(self, task: str, openai_api_key: str, partial_data: Dict[str, Any] = None, metadata: dict = None):
        super().__init__(task, metadata)
        self.openai_api_key = openai_api_key
        self.partial_data = partial_data or {}
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",  # Using GPT-4 for better research capabilities
            temperature=0.3
        )

    async def execute(self) -> Dict[str, Any]:
        try:
            # Analyze existing data from other agents
            context = self._prepare_research_context()
            
            messages = [
                {"role": "system", "content": "You are a research specialist focused on synthesizing information and identifying gaps in knowledge."},
                {"role": "user", "content": f"Task: {self.task}\nContext: {context}"}
            ]
            
            response = await self.llm.agenerate([messages])
            return {
                "research_findings": response.generations[0][0].message.content,
                "metadata": self.metadata
            }
            
        except Exception as e:
            return {
                "error": f"Research error: {str(e)}",
                "metadata": self.metadata
            }

    def _prepare_research_context(self) -> str:
        """
        Prepare research context from partial data
        """
        context_parts = []
        for key, data in self.partial_data.items():
            if isinstance(data, dict) and "error" not in data:
                context_parts.append(f"{key}: {str(data)}")
        return "\n".join(context_parts)