import logging
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RefinerAgent(BaseAgent):
    """
    Agent for refining queries and improving search strategies
    """
    def __init__(self, task: str, openai_api_key: str, partial_data: dict = None, metadata: dict = None):
        super().__init__(task, metadata)
        self.openai_api_key = openai_api_key
        self.partial_data = partial_data or {}
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4",
            temperature=0
        )

    async def execute(self) -> Dict[str, Any]:
        """Refine search strategy and optimize query"""
        try:
            refinement_result = await self._generate_refinement_plan()
            refinement_result["metadata"] = self.metadata
            return refinement_result
        except Exception as e:
            logger.error(f"Error in RefinerAgent: {e}")
            return {
                "error": str(e),
                "metadata": self.metadata
            }

    async def _generate_refinement_plan(self) -> Dict[str, Any]:
        """Generate a detailed refinement plan"""
        prompt = self._create_refinement_prompt()
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": """You are an expert in search optimization and information retrieval.
                                Always return responses in valid JSON format.
                                If you include any explanations, include them within the JSON structure."""
                },
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.agenerate([messages])
            content = response.generations[0][0].message.content.strip()
            
            # Attempt to extract JSON from the response if it contains additional text
            try:
                # Try to find JSON-like content between curly braces
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    return json.loads(json_content)
                
                # If no JSON-like content found, try parsing the whole response
                return json.loads(content)
                
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                logger.error(f"Problematic content: {content}")
                return self._create_backup_plan(error_details=str(json_error))
                
        except Exception as e:
            logger.error(f"Error generating refinement plan: {e}")
            return self._create_backup_plan(error_details=str(e))

    def _create_refinement_prompt(self) -> str:
        """Create the refinement prompt with clear JSON structure requirements"""
        return f"""
        Analyze this task and provide a detailed search plan:
        "{self.task}"

        Return ONLY a JSON object with the following structure, no additional text:
        {{
            "search_type": "type of search required",
            "sources": [
                {{
                    "url": "source domain",
                    "rationale": "why this source is authoritative"
                }}
            ],
            "data_points": ["specific data to collect"],
            "search_constraints": ["important limitations or filters"],
            "expected_format": "expected format for results",
            "analysis_notes": "any additional analysis or explanation"
        }}
        """

    def _create_backup_plan(self, error_details: str = None) -> Dict[str, Any]:
        """Create a backup plan with error tracking when the main plan fails"""
        backup_plan = {
            "sources": ["General search engines"],
            "data_points": ["Basic information about the topic"],
            "search_type": "general",
            "search_constraints": ["Focus on recent and reliable sources"],
            "expected_format": "text summary",
            "error_recovery": {
                "triggered": True,
                "timestamp": self.metadata.get("timestamp", "unknown"),
                "error_details": error_details
            }
        }
        
        if self.partial_data:
            backup_plan["partial_data_used"] = True
            backup_plan["available_data"] = list(self.partial_data.keys())
            
        return backup_plan