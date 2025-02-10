import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)

@dataclass
class InteractionPattern:
    """Patrón de interacción detectado."""
    pattern_id: str
    features: Dict[str, float]
    success_rate: float
    frequency: int
    last_used: datetime
    context_type: str

class AdaptiveSystem:
    """
    Sistema de autoadaptación que evoluciona basado en interacciones.
    
    Características:
    - Detección automática de patrones de comunicación
    - Adaptación dinámica de respuestas
    - Optimización continua de estrategias
    - Personalización profunda
    """
    
    def __init__(self):
        # Patrones detectados
        self.patterns: Dict[str, InteractionPattern] = {}
        
        # Estado actual del sistema
        self.current_state = {
            "interaction_mode": "normal",
            "complexity_level": 5,
            "personalization_depth": 3,
            "response_style": "balanced"
        }
        
        # Métricas de aprendizaje
        self.learning_metrics = {
            "patterns_detected": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0,
            "average_success_rate": 0.0
        }
        
        # Iniciar procesos de adaptación
        self._adaptation_task = asyncio.create_task(self._adapt_continuously())

    async def process_interaction(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesa una interacción y adapta el sistema.
        
        Args:
            input_data: Datos de la interacción
            context: Contexto adicional
            
        Returns:
            Dict con respuesta adaptada y metadatos
        """
        try:
            # Extraer características
            features = await self._extract_features(input_data, context)
            
            # Detectar patrones
            patterns = await self._detect_patterns(features)
            
            # Adaptar sistema
            adaptations = await self._adapt_system(patterns, features)
            
            # Generar respuesta adaptada
            response = await self._generate_adaptive_response(
                input_data,
                adaptations,
                context
            )
            
            # Actualizar métricas
            await self._update_metrics(response["success"])
            
            return {
                "response": response["content"],
                "adaptations_applied": adaptations,
                "confidence": response["confidence"],
                "metadata": {
                    "patterns_matched": len(patterns),
                    "adaptation_level": response["adaptation_level"],
                    "personalization_score": response["personalization_score"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return {"error": str(e)}

    async def _extract_features(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extrae características relevantes de la interacción."""
        features = {}
        
        try:
            # Análisis lingüístico
            text = input_data.get("text", "")
            features.update({
                "text_length": len(text),
                "complexity": self._analyze_complexity(text),
                "formality": self._analyze_formality(text),
                "sentiment": self._analyze_sentiment(text)
            })
            
            # Análisis de contexto
            if context:
                features.update({
                    "context_relevance": self._analyze_context_relevance(context),
                    "interaction_history": self._analyze_history(context),
                    "user_expertise": self._estimate_user_expertise(context)
                })
            
            # Análisis de comportamiento
            features.update({
                "interaction_frequency": self._calculate_frequency(context),
                "pattern_consistency": self._analyze_consistency(context),
                "adaptation_responsiveness": self._analyze_responsiveness(context)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}

    async def _detect_patterns(
        self,
        features: Dict[str, float]
    ) -> List[InteractionPattern]:
        """Detecta patrones en las características."""
        try:
            # Convertir características a vector
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Agrupar con patrones existentes
            all_patterns = np.array([
                list(p.features.values())
                for p in self.patterns.values()
            ])
            
            if len(all_patterns) > 0:
                # Usar DBSCAN para clustering
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(all_patterns)
                
                # Encontrar cluster más cercano
                distances = np.linalg.norm(all_patterns - feature_vector, axis=1)
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < 0.5:  # Umbral de similitud
                    pattern_id = list(self.patterns.keys())[closest_idx]
                    return [self.patterns[pattern_id]]
            
            # Crear nuevo patrón
            new_pattern = InteractionPattern(
                pattern_id=f"pat_{len(self.patterns)}",
                features=features,
                success_rate=1.0,
                frequency=1,
                last_used=datetime.now(),
                context_type="new"
            )
            
            self.patterns[new_pattern.pattern_id] = new_pattern
            self.learning_metrics["patterns_detected"] += 1
            
            return [new_pattern]
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    async def _adapt_system(
        self,
        patterns: List[InteractionPattern],
        current_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Adapta el sistema basado en patrones detectados."""
        adaptations = []
        
        try:
            for pattern in patterns:
                # Calcular diferencias
                feature_diff = {
                    k: current_features.get(k, 0) - pattern.features.get(k, 0)
                    for k in set(current_features) | set(pattern.features)
                }
                
                # Adaptar complejidad
                if abs(feature_diff.get("complexity", 0)) > 0.2:
                    new_level = max(1, min(10, 
                        self.current_state["complexity_level"] + 
                        feature_diff["complexity"] * 2
                    ))
                    adaptations.append({
                        "type": "complexity",
                        "from": self.current_state["complexity_level"],
                        "to": new_level,
                        "confidence": pattern.success_rate
                    })
                    self.current_state["complexity_level"] = new_level
                
                # Adaptar estilo
                if abs(feature_diff.get("formality", 0)) > 0.3:
                    new_style = "formal" if feature_diff["formality"] > 0 else "casual"
                    adaptations.append({
                        "type": "style",
                        "from": self.current_state["response_style"],
                        "to": new_style,
                        "confidence": pattern.success_rate
                    })
                    self.current_state["response_style"] = new_style
                
                # Adaptar personalización
                if pattern.success_rate > 0.8:
                    new_depth = min(10, 
                        self.current_state["personalization_depth"] + 1
                    )
                    adaptations.append({
                        "type": "personalization",
                        "from": self.current_state["personalization_depth"],
                        "to": new_depth,
                        "confidence": pattern.success_rate
                    })
                    self.current_state["personalization_depth"] = new_depth
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Error adapting system: {e}")
            return []

    async def _generate_adaptive_response(
        self,
        input_data: Dict[str, Any],
        adaptations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Genera una respuesta adaptada."""
        try:
            # Aplicar adaptaciones
            response_config = {
                "complexity": self.current_state["complexity_level"],
                "style": self.current_state["response_style"],
                "personalization_depth": self.current_state["personalization_depth"]
            }
            
            # Generar respuesta base
            base_response = await self._generate_base_response(
                input_data,
                response_config
            )
            
            # Personalizar respuesta
            personalized = await self._personalize_response(
                base_response,
                context,
                response_config
            )
            
            # Calcular métricas
            adaptation_level = len(adaptations) / 3  # Normalizado 0-1
            personalization_score = min(1.0, 
                self.current_state["personalization_depth"] / 10
            )
            
            # Calcular confianza
            confidence = np.mean([
                adapt["confidence"] for adapt in adaptations
            ]) if adaptations else 0.8
            
            return {
                "content": personalized,
                "success": True,
                "confidence": confidence,
                "adaptation_level": adaptation_level,
                "personalization_score": personalization_score
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "content": "Error generating response",
                "success": False,
                "confidence": 0.0,
                "adaptation_level": 0.0,
                "personalization_score": 0.0
            }

    async def _adapt_continuously(self) -> None:
        """Proceso continuo de adaptación del sistema."""
        try:
            while True:
                # Limpiar patrones antiguos
                await self._cleanup_patterns()
                
                # Optimizar patrones
                await self._optimize_patterns()
                
                # Actualizar métricas
                await self._update_learning_metrics()
                
                await asyncio.sleep(3600)  # Cada hora
                
        except Exception as e:
            logger.error(f"Error in continuous adaptation: {e}")
        finally:
            if self._adaptation_task and not self._adaptation_task.cancelled():
                self._adaptation_task.cancel()

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Cancelar tareas
            if self._adaptation_task:
                self._adaptation_task.cancel()
                try:
                    await self._adaptation_task
                except asyncio.CancelledError:
                    pass
            
            # Guardar patrones
            await self._save_patterns()
            
            # Limpiar estado
            self.patterns.clear()
            self.current_state = {
                "interaction_mode": "normal",
                "complexity_level": 5,
                "personalization_depth": 3,
                "response_style": "balanced"
            }
            
            logger.info("Adaptive system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema."""
        return {
            "current_state": self.current_state,
            "active_patterns": len(self.patterns),
            "learning_metrics": self.learning_metrics,
            "adaptation_status": {
                "last_adaptation": self._adaptation_task and not self._adaptation_task.cancelled(),
                "system_confidence": np.mean([p.success_rate for p in self.patterns.values()]) if self.patterns else 0.0
            }
        }