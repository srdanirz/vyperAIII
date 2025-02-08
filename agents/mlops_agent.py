# agents/mlops_agent.py

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error
)
import optuna
from prometheus_client import Counter, Gauge, Histogram

from .base_agent import BaseAgent
from core.llm import get_llm

logger = logging.getLogger(__name__)

class MLOpsAgent(BaseAgent):
    """
    Agente especializado en ML Operations, monitoreo y optimización.
    
    Capacidades:
    - Monitoreo de rendimiento de modelos
    - Detección de drift y degradación
    - Optimización automática de hiperparámetros
    - A/B testing de modelos
    - Gestión de versiones y experimentos
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__("mlops", api_key, metadata, shared_data)
        self.engine_mode = engine_mode
        
        # Métricas Prometheus
        self._setup_metrics()
        
        # Estado del sistema
        self.system_metrics = {
            "model_performance": {},
            "resource_usage": {},
            "error_rates": {},
            "latency_stats": {}
        }
        
        # Configuración de optimización
        self.optimization_config = {
            "target_metric": "latency",
            "optimization_metric": "mean_response_time",
            "constraints": {
                "max_error_rate": 0.05,
                "min_accuracy": 0.95
            }
        }
        
        # Historial de experimentos
        self.experiment_history = []

    def _setup_metrics(self) -> None:
        """Configura métricas Prometheus."""
        self.metrics = {
            "request_count": Counter(
                "vyper_request_total",
                "Total number of requests processed"
            ),
            "response_time": Histogram(
                "vyper_response_time_seconds",
                "Response time in seconds",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            ),
            "error_rate": Gauge(
                "vyper_error_rate",
                "Current error rate"
            ),
            "model_accuracy": Gauge(
                "vyper_model_accuracy",
                "Current model accuracy",
                ["model_name"]
            )
        }

    async def monitor_performance(
        self,
        model_name: str,
        predictions: List[Any],
        ground_truth: List[Any],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Monitorea el rendimiento de un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            predictions: Predicciones del modelo
            ground_truth: Valores reales
            metrics: Métricas a calcular
        """
        try:
            if not metrics:
                metrics = ["accuracy", "precision", "recall", "latency"]
            
            results = {}
            
            # Calcular métricas
            if "accuracy" in metrics:
                acc = accuracy_score(ground_truth, predictions)
                results["accuracy"] = float(acc)
                self.metrics["model_accuracy"].labels(model_name).set(acc)
                
            if "precision" in metrics:
                results["precision"] = float(
                    precision_score(ground_truth, predictions, average='weighted')
                )
                
            if "recall" in metrics:
                results["recall"] = float(
                    recall_score(ground_truth, predictions, average='weighted')
                )
                
            # Actualizar métricas del sistema
            self.system_metrics["model_performance"][model_name] = {
                "current_metrics": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Detectar degradación
            drift_detected = await self._detect_performance_drift(
                model_name,
                results
            )
            
            if drift_detected:
                await self._trigger_optimization(model_name)
            
            return {
                "model_name": model_name,
                "metrics": results,
                "drift_detected": drift_detected,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            raise

    async def optimize_model(
        self,
        model_name: str,
        optimization_objective: str = "latency",
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Optimiza los hiperparámetros de un modelo.
        
        Args:
            model_name: Nombre del modelo
            optimization_objective: Métrica a optimizar
            n_trials: Número de pruebas de optimización
        """
        try:
            # Crear estudio Optuna
            study = optuna.create_study(
                direction="minimize" if optimization_objective == "latency" else "maximize"
            )
            
            # Función objetivo para optimización
            def objective(trial):
                params = {
                    "temperature": trial.suggest_float("temperature", 0.0, 1.0),
                    "max_tokens": trial.suggest_int("max_tokens", 100, 4000),
                    "top_p": trial.suggest_float("top_p", 0.1, 1.0)
                }
                
                # Evaluar configuración
                return self._evaluate_config(model_name, params)
            
            # Ejecutar optimización
            study.optimize(objective, n_trials=n_trials)
            
            # Guardar resultados
            best_params = study.best_params
            best_value = study.best_value
            
            # Actualizar configuración
            await self._update_model_config(model_name, best_params)
            
            return {
                "model_name": model_name,
                "best_parameters": best_params,
                "best_value": float(best_value),
                "optimization_history": [
                    {
                        "trial": i,
                        "value": float(trial.value),
                        "params": trial.params
                    }
                    for i, trial in enumerate(study.trials)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            raise

    async def run_ab_test(
        self,
        model_a: str,
        model_b: str,
        test_data: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Ejecuta un test A/B entre dos modelos.
        
        Args:
            model_a: Primer modelo
            model_b: Segundo modelo
            test_data: Datos de prueba
            metrics: Métricas a comparar
        """
        try:
            results_a = await self._evaluate_model(model_a, test_data)
            results_b = await self._evaluate_model(model_b, test_data)
            
            # Calcular significancia estadística
            significance = await self._calculate_significance(
                results_a["metrics"],
                results_b["metrics"]
            )
            
            # Determinar ganador
            winner = model_a if significance["winner"] == "a" else model_b
            
            # Registrar experimento
            experiment = {
                "type": "ab_test",
                "models": {
                    "a": model_a,
                    "b": model_b
                },
                "results": {
                    "model_a": results_a,
                    "model_b": results_b
                },
                "significance": significance,
                "winner": winner,
                "timestamp": datetime.now().isoformat()
            }
            
            self.experiment_history.append(experiment)
            
            return experiment
            
        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            raise

    async def track_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Registra y trackea un experimento.
        
        Args:
            experiment_name: Nombre del experimento
            config: Configuración del experimento
            metrics: Métricas obtenidas
        """
        try:
            experiment = {
                "name": experiment_name,
                "config": config,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            # Guardar experimento
            self.experiment_history.append(experiment)
            
            # Actualizar métricas
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics[f"experiment_{metric_name}"].labels(
                        experiment_name
                    ).set(value)
            
            return experiment
            
        except Exception as e:
            logger.error(f"Error tracking experiment: {e}")
            raise

    async def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """
        Detecta drift en los datos.
        
        Args:
            reference_data: Datos de referencia
            current_data: Datos actuales
            features: Features a analizar
        """
        try:
            drift_metrics = {}
            
            for feature in features:
                # Calcular estadísticas
                ref_stats = reference_data[feature].describe()
                curr_stats = current_data[feature].describe()
                
                # Calcular KL divergence
                kl_div = self._calculate_kl_divergence(
                    reference_data[feature],
                    current_data[feature]
                )
                
                drift_metrics[feature] = {
                    "statistics_diff": {
                        stat: float(curr_stats[stat] - ref_stats[stat])
                        for stat in ["mean", "std", "25%", "50%", "75%"]
                    },
                    "kl_divergence": float(kl_div),
                    "drift_detected": kl_div > 0.1  # Umbral configurable
                }
            
            return {
                "drift_detected": any(
                    m["drift_detected"] for m in drift_metrics.values()
                ),
                "feature_metrics": drift_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            raise

    async def _detect_performance_drift(
        self,
        model_name: str,
        current_metrics: Dict[str, float]
    ) -> bool:
        """Detecta degradación en el rendimiento del modelo."""
        try:
            if model_name not in self.system_metrics["model_performance"]:
                return False
                
            historical = self.system_metrics["model_performance"][model_name]
            
            # Calcular cambios significativos
            for metric, current_value in current_metrics.items():
                if metric in historical["current_metrics"]:
                    historical_value = historical["current_metrics"][metric]
                    change = abs(current_value - historical_value) / historical_value
                    
                    # Umbral de 10% de cambio
                    if change > 0.1:
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Error detecting performance drift: {e}")
            return False

    async def _trigger_optimization(self, model_name: str) -> None:
        """Inicia optimización automática cuando se detecta degradación."""
        try:
            logger.info(f"Triggering optimization for {model_name}")
            
            # Iniciar optimización en background
            asyncio.create_task(
                self.optimize_model(
                    model_name,
                    self.optimization_config["target_metric"]
                )
            )
            
        except Exception as e:
            logger.error(f"Error triggering optimization: {e}")

    async def _evaluate_model(
        self,
        model_name: str,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evalúa un modelo con datos de prueba."""
        try:
            start_time = datetime.now()
            
            # Procesar datos de prueba
            predictions = []
            ground_truth = []
            latencies = []
            
            for sample in test_data:
                # Medir latencia
                pred_start = datetime.now()
                
                if self.engine_mode == "openai":
                    prediction = await self._predict_openai(model_name, sample)
                else:
                    prediction = await self._predict_deepseek(model_name, sample)
                    
                latency = (datetime.now() - pred_start).total_seconds()
                latencies.append(latency)
                
                predictions.append(prediction)
                ground_truth.append(sample["expected"])
            
            # Calcular métricas
            metrics = {
                "accuracy": float(accuracy_score(ground_truth, predictions)),
                "mean_latency": float(np.mean(latencies)),
                "p95_latency": float(np.percentile(latencies, 95)),
                "error_rate": 1.0 - float(accuracy_score(ground_truth, predictions))
            }
            
            return {
                "model_name": model_name,
                "metrics": metrics,
                "predictions": predictions,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def _calculate_kl_divergence(
        self,
        p: pd.Series,
        q: pd.Series,
        bins: int = 50
    ) -> float:
        """Calcula Kullback-Leibler divergence entre dos distribuciones."""
        try:
            # Calcular histogramas
            p_hist, _ = np.histogram(p, bins=bins, density=True)
            q_hist, _ = np.histogram(q, bins=bins, density=True)
            
            # Evitar divisiones por cero
            p_hist = np.clip(p_hist, 1e-10, None)
            q_hist = np.clip(q_hist, 1e-10, None)
            
            # Calcular KL divergence
            return float(np.sum(p_hist * np.log(p_hist / q_hist)))
            
        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return float('inf')

    async def _calculate_significance(
        self,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calcula significancia estadística entre resultados."""
        try:
            results = {}
            
            for metric in metrics_a:
                if metric in metrics_b:
                    # Calcular diferencia relativa
                    diff = abs(metrics_a[metric] - metrics_b[metric])
                    relative_diff = diff / metrics_a[metric]
                    
                    results[metric] = {
                        "absolute_difference": float(diff),
                        "relative_difference": float(relative_diff),
                        "significant": relative_diff > 0.05  # 5% threshold
                    }
            
            # Determinar ganador
            winner = "a"
            if sum(r["relative_difference"] for r in results.values()) < 0:
                winner = "b"
                
            return {
                "metrics": results,
                "winner": winner,
                "confidence": "high" if all(r["significant"] for r in results.values()) else "low"
            }
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            raise

    async def cleanup(self) -> None:
        """Limpia recursos y persiste datos importantes."""
        try:
            # Guardar historial de experimentos
            history_path = Path("experiment_history.json")
            with open(history_path, "w") as f:
                json.dump(self.experiment_history, f, indent=2)
            
            # Limpiar métricas
            self.system_metrics.clear()
            
            logger.info("MLOps Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del agente."""
        return {
            "agent_type": "MLOpsAgent",
            "engine_mode": self.engine_mode,
            "active_experiments": len(self.experiment_history),
            "system_metrics": self.system_metrics,
            "optimization_config": self.optimization_config
        }