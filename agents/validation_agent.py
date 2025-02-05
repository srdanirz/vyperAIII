# agents/validation_agent.py
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """Agent specialized in validating results and ensuring quality control"""
    def __init__(self, task: str, openai_api_key: str, partial_data: Dict[str, Any] = None, metadata: dict = None):
        super().__init__(task, openai_api_key, metadata)
        self.partial_data = partial_data or {}
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",  # Using 3.5 for efficiency
            temperature=0
        )

    async def execute(self) -> Dict[str, Any]:
        """Validate results in partial_data"""
        try:
            validation_results = {}
            
            # Process each piece of data separately
            for key, data in self.partial_data.items():
                validation = await self._validate_result(key, self._prepare_data_for_validation(data))
                validation_results[key] = validation
            
            return {
                "validation_summary": validation_results,
                "overall_status": self._get_overall_status(validation_results),
                "metadata": self.metadata
            }
        
        except Exception as e:
            logger.error(f"Error in ValidationAgent: {e}")
            return {
                "error": f"Validation error: {str(e)}",
                "metadata": self.metadata
            }

    def _prepare_data_for_validation(self, data: Any) -> Any:
        """Prepare data for validation by reducing size if needed"""
        if isinstance(data, dict):
            if "content" in data and isinstance(data["content"], str) and len(data["content"]) > 1000:
                # For base64 content, just verify it's valid base64
                if self._is_base64(data["content"]):
                    return {
                        "format": data.get("format", "unknown"),
                        "size": len(data["content"]),
                        "type": "binary_content"
                    }
                # For text content, truncate
                return {
                    "content_preview": data["content"][:1000] + "...",
                    "format": data.get("format"),
                    "total_length": len(data["content"])
                }
            return {k: self._prepare_data_for_validation(v) for k, v in data.items()}
        elif isinstance(data, list) and len(data) > 10:
            return data[:10] + ["... and {} more items".format(len(data) - 10)]
        return data

    def _is_base64(self, s: str) -> bool:
        """Check if a string is base64 encoded"""
        import base64
        try:
            # Check if string contains only valid base64 characters
            if not all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in s):
                return False
            # Try to decode
            base64.b64decode(s)
            return True
        except:
            return False

    async def _validate_result(self, key: str, data: Any) -> Dict[str, Any]:
        """Validate a single result using LLM"""
        try:
            messages = [
                {"role": "system", "content": """Analyze this content and validate:
                1. Format and structure correctness
                2. Content completeness
                3. Data consistency
                Be concise in your validation."""},
                {"role": "user", "content": f"Validate this {key} result: {str(data)}"}
            ]
            
            response = await self.llm.agenerate([messages])
            validation_text = response.generations[0][0].message.content
            
            return {
                "valid": self._determine_validity(validation_text),
                "validation_notes": validation_text
            }
        except Exception as e:
            logger.error(f"Error validating {key}: {e}")
            return {
                "valid": False,
                "validation_notes": f"Error during validation: {str(e)}"
            }

    def _determine_validity(self, validation_text: str) -> bool:
        """Determine if content is valid based on validation text"""
        validation_text = validation_text.lower()
        invalid_indicators = ["error", "invalid", "missing", "incomplete", "incorrect"]
        return not any(indicator in validation_text for indicator in invalid_indicators)

    def _get_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        if not validation_results:
            return "NO_DATA"
        
        all_valid = all(v.get("valid", False) for v in validation_results.values())
        return "PASSED" if all_valid else "FAILED"