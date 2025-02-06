# agents/validation_agent.py

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from llm_factory import get_llm
from config import get_config

logger = logging.getLogger(__name__)

class ValidationCriteria:
    """Define criterios de validación para diferentes tipos de contenido"""

    @staticmethod
    def get_criteria(content_type: str) -> Dict[str, Any]:
        base_criteria = {
            "clarity": {
                "weight": 0.2,
                "description": "Claridad y comprensión del contenido",
                "min_score": 7.0
            },
            "accuracy": {
                "weight": 0.25,
                "description": "Precisión y veracidad de la información",
                "min_score": 8.0
            },
            "coherence": {
                "weight": 0.15,
                "description": "Coherencia y estructura lógica",
                "min_score": 7.5
            },
            "relevance": {
                "weight": 0.2,
                "description": "Relevancia para el objetivo",
                "min_score": 7.5
            },
            "completeness": {
                "weight": 0.2,
                "description": "Completitud de la información",
                "min_score": 7.0
            }
        }

        specific_criteria = {
            "powerpoint": {
                "visual_appeal": {
                    "weight": 0.15,
                    "description": "Atractivo visual y diseño",
                    "min_score": 7.0
                },
                "slide_density": {
                    "weight": 0.1,
                    "description": "Densidad apropiada de información por slide",
                    "min_score": 7.0
                }
            },
            "document": {
                "formatting": {
                    "weight": 0.15,
                    "description": "Formato y estilo consistente",
                    "min_score": 7.5
                },
                "readability": {
                    "weight": 0.15,
                    "description": "Facilidad de lectura",
                    "min_score": 7.5
                }
            }
        }

        if content_type in specific_criteria:
            return {**base_criteria, **specific_criteria[content_type]}
        return base_criteria

class ValidationAgent(BaseAgent):
    """
    Agente avanzado de validación que asegura la calidad del contenido
    y proporciona retroalimentación detallada para mejoras.
    """

    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.config = get_config()
        engine_mode = self.metadata.get("engine_mode", "openai")
        self.llm = get_llm(
            engine_mode,
            openai_api_key,
            model=self.config["api"]["openai"]["model"],
            temperature=0.2  # Bajo para consistencia
        )

    async def _execute(self) -> Dict[str, Any]:
        """Ejecuta la validación completa del contenido"""
        try:
            # Identificar tipo de contenido
            content_type = self._determine_content_type()
            
            # Obtener criterios de validación
            validation_criteria = ValidationCriteria.get_criteria(content_type)
            
            # Realizar validaciones
            validation_results = await self._perform_validations(
                content_type,
                validation_criteria
            )
            
            # Generar recomendaciones de mejora
            improvement_suggestions = await self._generate_improvements(
                validation_results,
                content_type
            )
            
            # Calcular score final
            final_score = self._calculate_final_score(validation_results)
            
            # Determinar estado general
            overall_status = "PASSED" if final_score >= 7.5 else "NEEDS_IMPROVEMENT"
            
            return {
                "validation_summary": {
                    "content_type": content_type,
                    "overall_status": overall_status,
                    "final_score": final_score,
                    "timestamp": datetime.now().isoformat(),
                    "detailed_results": validation_results,
                    "improvement_suggestions": improvement_suggestions,
                    "validation_metadata": {
                        "criteria_used": validation_criteria,
                        "validator_version": "2.0.0"
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in ValidationAgent: {e}")
            return {"error": str(e)}

    def _determine_content_type(self) -> str:
        """Determina el tipo de contenido a validar"""
        for agent_data in self.shared_data.values():
            if isinstance(agent_data, dict):
                if "content_type" in agent_data:
                    return agent_data["content_type"]
                if "type" in agent_data:
                    return agent_data["type"]
        return "generic"

    async def _perform_validations(
        self,
        content_type: str,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Realiza validaciones detalladas del contenido"""
        validation_results = {}
        content_to_validate = self._get_content_to_validate()

        for criterion_name, criterion_info in criteria.items():
            messages = [
                {
                    "role": "system",
                    "content": f"""Eres un experto en validación de {criterion_name}.
                    Evalúa el contenido según este criterio: {criterion_info['description']}.
                    Proporciona una puntuación de 0 a 10 y una justificación detallada."""
                },
                {
                    "role": "user",
                    "content": f"Evalúa este contenido:\n{json.dumps(content_to_validate, indent=2)}"
                }
            ]

            response = await self.llm.agenerate([messages])
            result = self._parse_validation_response(response.generations[0][0].message.content)
            validation_results[criterion_name] = result

        return validation_results

    def _get_content_to_validate(self) -> Dict[str, Any]:
        """Obtiene el contenido a validar del shared_data"""
        content = {}
        for agent_data in self.shared_data.values():
            if isinstance(agent_data, dict):
                if "content" in agent_data:
                    content.update(agent_data["content"])
                elif "result" in agent_data:
                    content.update(agent_data["result"])
        return content

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parsea la respuesta de validación del LLM"""
        try:
            # Extrae score y justificación del texto
            lines = response.split('\n')
            score = None
            justification = []

            for line in lines:
                if 'score' in line.lower() or 'puntuación' in line.lower():
                    # Busca un número entre 0 y 10
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        score = float(numbers[0])
                elif line.strip():
                    justification.append(line.strip())

            return {
                "score": score if score is not None else 5.0,  # Default medio si no se encuentra
                "justification": ' '.join(justification),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error parsing validation response: {e}")
            return {
                "score": 5.0,
                "justification": "Error parsing validation response",
                "error": str(e)
            }

    async def _generate_improvements(
        self,
        validation_results: Dict[str, Any],
        content_type: str
    ) -> List[Dict[str, Any]]:
        """Genera sugerencias de mejora basadas en los resultados"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""Eres un experto en mejora de contenido de tipo {content_type}.
                    Analiza los resultados de validación y sugiere mejoras específicas y accionables.
                    Prioriza las sugerencias por impacto potencial."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Basado en estos resultados de validación:
                    {json.dumps(validation_results, indent=2)}
                    
                    Genera sugerencias de mejora en formato JSON con:
                    - priority: (high|medium|low)
                    - aspect: área a mejorar
                    - suggestion: descripción específica
                    - expected_impact: impacto esperado
                    """
                }
            ]

            response = await self.llm.agenerate([messages])
            suggestions = json.loads(response.generations[0][0].message.content)
            
            return sorted(
                suggestions,
                key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]
            )

        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            return [{
                "priority": "high",
                "aspect": "error_handling",
                "suggestion": "Error generating improvements",
                "expected_impact": "N/A"
            }]

    def _calculate_final_score(self, validation_results: Dict[str, Any]) -> float:
        """Calcula el score final ponderado"""
        try:
            total_weight = 0
            weighted_sum = 0

            for criterion, result in validation_results.items():
                weight = ValidationCriteria.get_criteria(
                    self._determine_content_type()
                )[criterion]["weight"]
                score = result["score"]
                
                weighted_sum += weight * score
                total_weight += weight

            return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating final score: {e}")
            return 0.0

    async def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del agente"""
        return {
            "agent_type": "ValidationAgent",
            "task": self.task,
            "metadata": self.metadata,
            "validation_criteria": ValidationCriteria.get_criteria(
                self._determine_content_type()
            )
        }