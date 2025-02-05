# agents/analysis_agent.py
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    def __init__(self, task: str, openai_api_key: str, metadata: Optional[Dict[str, Any]] = None, partial_data: Optional[Dict[str, Any]] = None):
        super().__init__(task, openai_api_key, metadata, partial_data)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4",
            temperature=0
        )

    async def _execute(self) -> Dict[str, Any]:
        """Execute analysis task"""
        try:
            # Get data for analysis
            data_to_analyze = self._get_analysis_data()
            
            # Perform analysis
            analysis_result = await self._analyze_data(data_to_analyze)
            
            # Determine complexity
            is_complex = len(data_to_analyze) > 100 if isinstance(data_to_analyze, str) else False
            
            return {
                "result": analysis_result,
                "type": "complex" if is_complex else "simple",
                "data_analyzed": True,
                "partial_results": self.partial_data
            }

        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise

    def _get_analysis_data(self) -> Any:
        """Get data to analyze from task or partial data"""
        if not self.partial_data:
            return self.task

        # Check for existing results
        for k, v in self.partial_data.items():
            if isinstance(v, dict):
                if "result" in v:
                    return v["result"]
                elif "response" in v:
                    return v["response"]
                elif "content" in v:
                    return v["content"]

        return self.task

    async def _analyze_data(self, data: Any) -> str:
        """Analyze the data using LLM"""
        messages = [
            {"role": "system", "content": """You are an expert analyst AI. Analyze the provided information 
            and provide clear, actionable insights. Focus on key points and provide concrete conclusions."""},
            {"role": "user", "content": f"""Task: {self.task}
            
Data to analyze:
{data}

Provide a thorough analysis."""}
        ]

        response = await self.llm.agenerate([messages])
        return response.generations[0][0].message.content
