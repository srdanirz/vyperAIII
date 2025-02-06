# audit/decision_explainer.py

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import shap
import lime.lime_text
from llm_factory import get_llm

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Estructura para resultados de explicación."""
    summary: str
    importance_scores: Dict[str, float]
    contributing_factors: List[Dict[str, Any]]
    confidence_analysis: Dict[str, Any]
    metadata: Dict[str, Any]

class DecisionExplainer:
    """
    Sistema avanzado de explicabilidad para decisiones de IA.
    
    Características:
    - Explicaciones en lenguaje natural
    - Análisis de importancia de features
    - Análisis de confianza
    - Detección de factores críticos
    """
    
    def __init__(self, api_key: str, engine_mode: str = "openai"):
        self.api_key = api_key
        self.engine_mode = engine_mode
        self.llm = get_llm(engine_mode, api_key)
        
        # Inicializar componentes
        self._initialize_explainers()
        
        # Cache de explicaciones
        self.explanation_cache: Dict[str, ExplanationResult] = {}
        
        # Historial y métricas
        self.history: List[Dict[str, Any]] = []
        self.metrics = {
            "total_explanations": 0,
            "cache_hits": 0,
            "average_time": 0.0
        }

    def _initialize_explainers(self) -> None:
        """Inicializa componentes de explicación."""
        self.explainers = {
            "shap": self._create_shap_explainer(),
            "lime": self._create_lime_explainer()
        }

    def _create_shap_explainer(self) -> Any:
        """Crea explainer SHAP."""
        try:
            return shap.TreeExplainer if self.engine_mode == "deepseek" else shap.KernelExplainer
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            return None

    def _create_lime_explainer(self) -> Any:
        """Crea explainer LIME."""
        try:
            return lime.lime_text.LimeTextExplainer(
                class_names=["negative", "neutral", "positive"],
                verbose=False
            )
        except Exception as e:
            logger.error(f"Error creating LIME explainer: {e}")
            return None

    async def explain_decision(
        self,
        decision: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ExplanationResult:
        """
        Genera una explicación completa de una decisión.
        
        Args:
            decision: Resultado de la decisión del modelo
            input_data: Datos de entrada originales
            context: Contexto adicional opcional
        
        Returns:
            ExplanationResult con la explicación completa
        """
        try:
            start_time = datetime.now()
            
            # Verificar caché
            cache_key = self._generate_cache_key(decision, input_data)
            if cached := self.explanation_cache.get(cache_key):
                self.metrics["cache_hits"] += 1
                return cached

            # 1. Generar explicación narrativa
            narrative = await self._generate_narrative(decision, input_data, context)

            # 2. Analizar importancia de features
            importance_scores = await self._analyze_feature_importance(decision, input_data)

            # 3. Identificar factores contribuyentes
            factors = await self._identify_contributing_factors(decision, input_data)

            # 4. Analizar confianza
            confidence = await self._analyze_confidence(decision)

            # Crear resultado
            result = ExplanationResult(
                summary=narrative,
                importance_scores=importance_scores,
                contributing_factors=factors,
                confidence_analysis=confidence,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "engine_mode": self.engine_mode,
                    "execution_time": (datetime.now() - start_time).total_seconds()
                }
            )

            # Actualizar caché y métricas
            self._update_cache_and_metrics(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise

    async def _generate_narrative(
        self,
        decision: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Genera explicación narrativa usando LLM."""
        try:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Generate a clear and concise explanation of an AI decision. "
                        "Focus on key factors and their impact. "
                        "Use natural language and avoid technical jargon."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    Decision: {json.dumps(decision, indent=2)}
                    Input Data: {json.dumps(input_data, indent=2)}
                    Context: {json.dumps(context, indent=2) if context else 'None'}
                    
                    Explain this decision, highlighting:
                    1. Main factors that influenced the decision
                    2. How these factors were weighted
                    3. Level of confidence and why
                    4. Any important context or caveats
                    """
                }
            ]

            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].message.content

        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return "Error generating explanation"

    async def _analyze_feature_importance(
        self,
        decision: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analiza importancia de features usando SHAP y LIME."""
        try:
            importance_scores = {}

            # SHAP analysis if available
            if self.explainers["shap"]:
                shap_values = await self._get_shap_values(decision, input_data)
                for feature, value in shap_values.items():
                    importance_scores[f"shap_{feature}"] = float(value)

            # LIME analysis if available
            if self.explainers["lime"]:
                lime_values = await self._get_lime_values(decision, input_data)
                for feature, value in lime_values.items():
                    importance_scores[f"lime_{feature}"] = float(value)

            return importance_scores

        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {}

    async def _identify_contributing_factors(
        self,
        decision: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identifica factores que contribuyeron a la decisión."""
        try:
            factors = []

            # Analizar cada feature
            for feature, value in input_data.items():
                impact = await self._analyze_feature_impact(
                    feature,
                    value,
                    decision
                )
                
                if impact["significance"] > 0.1:  # Umbral configurable
                    factors.append({
                        "feature": feature,
                        "value": value,
                        "impact": impact["score"],
                        "direction": impact["direction"],
                        "confidence": impact["confidence"]
                    })

            # Ordenar por impacto
            return sorted(
                factors,
                key=lambda x: abs(x["impact"]),
                reverse=True
            )

        except Exception as e:
            logger.error(f"Error identifying contributing factors: {e}")
            return []

    async def _analyze_confidence(
        self,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza nivel de confianza de la decisión."""
        try:
            # Extraer probabilidades si existen
            probabilities = decision.get("probabilities", [])
            if not probabilities:
                return {
                    "confidence_score": 0.0,
                    "reliability": "unknown"
                }

            # Calcular métricas de confianza
            max_prob = float(max(probabilities))
            entropy = float(-sum(p * np.log(p) for p in probabilities if p > 0))

            return {
                "confidence_score": max_prob,
                "entropy": entropy,
                "reliability": self._assess_reliability(max_prob, entropy),
                "distribution": {
                    "max_probability": max_prob,
                    "entropy": entropy,
                    "probability_distribution": [float(p) for p in probabilities]
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing confidence: {e}")
            return {
                "confidence_score": 0.0,
                "reliability": "error"
            }

    def _assess_reliability(
        self,
        confidence: float,
        entropy: float
    ) -> str:
        """Evalúa confiabilidad basada en métricas."""
        if confidence > 0.9 and entropy < 0.5:
            return "high"
        elif confidence > 0.7 and entropy < 1.0:
            return "medium"
        else:
            return "low"

    async def _analyze_feature_impact(
        self,
        feature: str,
        value: Any,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza el impacto de un feature específico."""
        try:
            # Calcular significancia
            significance = await self._calculate_significance(
                feature,
                value,
                decision
            )

            # Determinar dirección del impacto
            direction = "positive" if significance > 0 else "negative"

            # Calcular confianza
            confidence = min(abs(significance) * 2, 1.0)

            return {
                "significance": abs(significance),
                "score": float(significance),
                "direction": direction,
                "confidence": float(confidence)
            }

        except Exception as e:
            logger.error(f"Error analyzing feature impact: {e}")
            return {
                "significance": 0.0,
                "score": 0.0,
                "direction": "unknown",
                "confidence": 0.0
            }

    def _generate_cache_key(
        self,
        decision: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> str:
        """Genera clave única para caché."""
        try:
            combined = {
                "decision": decision,
                "input": input_data
            }
            return hashlib.md5(
                json.dumps(combined, sort_keys=True).encode()
            ).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return datetime.now().isoformat()

    def _update_cache_and_metrics(
        self,
        cache_key: str,
        result: ExplanationResult
    ) -> None:
        """Actualiza caché y métricas."""
        try:
            # Actualizar caché
            self.explanation_cache[cache_key] = result

            # Actualizar métricas
            self.metrics["total_explanations"] += 1
            execution_time = result.metadata["execution_time"]
            self.metrics["average_time"] = (
                (self.metrics["average_time"] * (self.metrics["total_explanations"] - 1) +
                execution_time) / self.metrics["total_explanations"]
            )

            # Añadir a historial
            self.history.append(asdict(result))

        except Exception as e:
            logger.error(f"Error updating cache and metrics: {e}")

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Guardar historial
            history_path = Path("explanation_history.json")
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)

            # Limpiar recursos
            self.explanation_cache.clear()
            self.explainers.clear()

            logger.info("Decision Explainer cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema."""
        return {
            "active_explainers": list(self.explainers.keys()),
            "cache_size": len(self.explanation_cache),
            "history_size": len(self.history),
            "metrics": self.metrics
        }