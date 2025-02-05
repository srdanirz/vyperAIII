# agents/data_processing_agent.py
import logging
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent
import pandas as pd
import numpy as np
import io

logger = logging.getLogger(__name__)

class DataProcessingAgent(BaseAgent):
    """
    Agent specialized in data processing and analysis
    """

    def __init__(self, task: str, openai_api_key: str, metadata: Optional[Dict[str, Any]] = None, partial_data: Optional[Dict[str, Any]] = None):
        super().__init__(task, openai_api_key, metadata, partial_data)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.7
        )

    async def _execute(self) -> Dict[str, Any]:
        """Process and transform data or content as needed"""
        try:
            # Analyze what type of processing is needed
            processing_plan = await self._analyze_processing_needs()
            
            # Get data to process
            input_data = self._get_input_data()
            
            # Process data according to plan
            processed_result = await self._process_data(input_data, processing_plan)
            
            return {
                "result": processed_result,
                "processing_type": processing_plan["type"],
                "format": processing_plan.get("output_format"),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Error in DataProcessingAgent: {e}")
            raise

    async def _analyze_processing_needs(self) -> Dict[str, Any]:
        """Analyze what kind of processing is needed based on task and data"""
        messages = [
            {"role": "system", "content": """You are a data processing expert. Analyze the task and determine:
            1. What kind of processing is needed
            2. What should be the output format
            3. What transformations are required
            Be specific but flexible."""},
            {"role": "user", "content": f"Task: {self.task}\nAvailable data: {list(self.partial_data.keys()) if self.partial_data else 'None'}"}
        ]

        response = await self.llm.agenerate([messages])
        processing_analysis = response.generations[0][0].message.content

        # Convert analysis to structured plan
        plan = await self._structure_processing_plan(processing_analysis)
        return plan

    async def _structure_processing_plan(self, analysis: str) -> Dict[str, Any]:
        """Convert analysis into structured plan"""
        messages = [
            {"role": "system", "content": """Convert the analysis into a structured plan with:
            {
                "type": "processing type",
                "output_format": "desired format",
                "transformations": ["list of needed transformations"],
                "requirements": {"specific requirements"}
            }"""},
            {"role": "user", "content": analysis}
        ]

        response = await self.llm.agenerate([messages])
        return eval(response.generations[0][0].message.content)

    def _get_input_data(self) -> Any:
        """Get data to process from context or partial_data"""
        if not self.partial_data:
            return None

        # Look for processable data in partial_data
        for key, value in self.partial_data.items():
            if isinstance(value, (dict, list)):
                return value
            elif isinstance(value, str) and len(value) > 0:
                return value

        return None

    async def _process_data(self, data: Any, plan: Dict[str, Any]) -> Any:
        """Process data according to plan"""
        if data is None:
            # Generate new content if no input data
            return await self._generate_content(plan)

        # Convert data to appropriate format if needed
        if isinstance(data, str):
            try:
                # Try to parse as JSON
                import json
                data = json.loads(data)
            except json.JSONDecodeError:
                # If not JSON, might be CSV
                try:
                    import pandas as pd
                    import io
                    data = pd.read_csv(io.StringIO(data))
                except:
                    # If not CSV, keep as text
                    pass

        # Apply transformations according to plan
        for transform in plan.get("transformations", []):
            data = await self._apply_transformation(data, transform)

        return data

    async def _apply_transformation(self, data: Any, transformation: str) -> Any:
        """Apply specific transformation to data"""
        if isinstance(data, pd.DataFrame):
            if transformation == "clean":
                return data.dropna().drop_duplicates()
            elif transformation == "summarize":
                return data.describe()
            
        elif isinstance(data, (dict, list)):
            if transformation == "structure":
                return pd.DataFrame(data)
            elif transformation == "format":
                return self._format_data(data)
            
        elif isinstance(data, str):
            if transformation == "extract_key_points":
                return await self._extract_key_points(data)
            elif transformation == "organize":
                return await self._organize_content(data)

        return data

    async def _generate_content(self, plan: Dict[str, Any]) -> Any:
        """Generate new content based on plan"""
        messages = [
            {"role": "system", "content": f"""Generate content according to this plan:
            {plan}
            Make it detailed and well-structured."""},
            {"role": "user", "content": self.task}
        ]

        response = await self.llm.agenerate([messages])
        return response.generations[0][0].message.content

    async def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        messages = [
            {"role": "system", "content": "Extract the key points from this text. Return them as a list."},
            {"role": "user", "content": text}
        ]

        response = await self.llm.agenerate([messages])
        return response.generations[0][0].message.content.split("\n")

    async def _organize_content(self, content: str) -> Dict[str, Any]:
        """Organize content into coherent structure"""
        messages = [
            {"role": "system", "content": "Organize this content into a clear structure with sections and subsections."},
            {"role": "user", "content": content}
        ]

        response = await self.llm.agenerate([messages])
        return eval(response.generations[0][0].message.content)

    def _format_data(self, data: Any) -> Dict[str, Any]:
        """Format data structure consistently"""
        if isinstance(data, dict):
            return {
                key: self._format_data(value) if isinstance(value, (dict, list)) else value
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self._format_data(item) if isinstance(item, (dict, list)) else item
                for item in data
            ]
        return data