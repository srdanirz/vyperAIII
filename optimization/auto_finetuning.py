# optimization/auto_finetuning.py

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    Trainer, TrainingArguments,
    AutoModelForCausalLM, AutoTokenizer
)

from .base_optimizer import BaseOptimizer
from llm_factory import get_llm

logger = logging.getLogger(__name__)

class AutoFineTuner(BaseOptimizer):
    """
    Sistema de fine-tuning automático para modelos de lenguaje.
    
    Características:
    - Detección automática de necesidad de fine-tuning
    - Preparación de datos de entrenamiento
    - Optimización de hiperparámetros
    - Evaluación continua
    - Gestión de versiones
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__("auto_finetuning", api_key, metadata)
        self.engine_mode = engine_mode
        
        # Configuración
        self.config = self._load_config()
        
        # Estado
        self.training_state = {
            "active_jobs": {},
            "model_versions": {},
            "performance_history": {}
        }
        
        # Métricas
        self.metrics = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "average_improvement": 0.0
        }

    def _load_config(self) -> Dict[str, Any]:
        """Carga configuración de fine-tuning."""
        try:
            config_path = Path(__file__).parent / "finetuning_config.yaml"
            if not config_path.exists():
                return self._get_default_config()
            
            with open(config_path) as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuración por defecto."""
        return {
            "training": {
                "batch_size": 8,
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "eval_steps": 100,
                "save_steps": 500
            },
            "optimization": {
                "metric": "loss",
                "patience": 3,
                "min_improvement": 0.01
            },
            "thresholds": {
                "min_samples": 1000,
                "max_samples": 50000,
                "min_performance": 0.7,
                "drift_threshold": 0.1
            }
        }

    async def check_finetuning_need(
        self,
        model_name: str,
        recent_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Verifica si un modelo necesita fine-tuning.
        
        Args:
            model_name: Nombre del modelo
            recent_performance: Métricas recientes
        """
        try:
            needs_finetuning = False
            reasons = []
            
            # Verificar degradación de rendimiento
            if self._detect_performance_degradation(
                model_name,
                recent_performance
            ):
                needs_finetuning = True
                reasons.append("performance_degradation")
            
            # Verificar drift de datos
            if await self._detect_data_drift(model_name):
                needs_finetuning = True
                reasons.append("data_drift")
            
            # Verificar tiempo desde último fine-tuning
            if self._check_time_threshold(model_name):
                needs_finetuning = True
                reasons.append("time_threshold")
            
            return {
                "needs_finetuning": needs_finetuning,
                "reasons": reasons,
                "current_performance": recent_performance,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking fine-tuning need: {e}")
            return {
                "needs_finetuning": False,
                "error": str(e)
            }

    async def prepare_training_data(
        self,
        data_sources: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Prepara datos para fine-tuning.
        
        Args:
            data_sources: Fuentes de datos
            validation_split: Proporción para validación
        """
        try:
            processed_data = []
            
            # Procesar cada fuente
            for source in data_sources:
                data = await self._process_data_source(source)
                processed_data.extend(data)
            
            # Validar cantidad de datos
            if len(processed_data) < self.config["thresholds"]["min_samples"]:
                raise ValueError("Insufficient training data")
            
            # Dividir datos
            train_data, val_data = train_test_split(
                processed_data,
                test_size=validation_split,
                random_state=42
            )
            
            return {
                "training_data": train_data,
                "validation_data": val_data,
                "metrics": {
                    "total_samples": len(processed_data),
                    "training_samples": len(train_data),
                    "validation_samples": len(val_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    async def start_finetuning(
        self,
        model_name: str,
        training_data: Dict[str, Any],
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Inicia proceso de fine-tuning.
        
        Args:
            model_name: Nombre del modelo
            training_data: Datos preparados
            custom_config: Configuración personalizada
        """
        try:
            # Crear ID de trabajo
            job_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Combinar configuraciones
            config = self.config["training"].copy()
            if custom_config:
                config.update(custom_config)
            
            # Registrar trabajo
            self.training_state["active_jobs"][job_id] = {
                "model_name": model_name,
                "status": "preparing",
                "config": config,
                "start_time": datetime.now().isoformat()
            }
            
            # Iniciar entrenamiento en background
            asyncio.create_task(
                self._run_finetuning(job_id, model_name, training_data, config)
            )
            
            return {
                "job_id": job_id,
                "model_name": model_name,
                "status": "started",
                "config": config
            }
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning: {e}")
            raise

    async def _run_finetuning(
        self,
        job_id: str,
        model_name: str,
        training_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> None:
        """Ejecuta el proceso de fine-tuning."""
        try:
            # Actualizar estado
            self.training_state["active_jobs"][job_id]["status"] = "training"
            
            # Preparar modelo y tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configurar entrenamiento
            training_args = TrainingArguments(
                output_dir=f"./models/{job_id}",
                num_train_epochs=config["num_epochs"],
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["batch_size"],
                warmup_steps=config["warmup_steps"],
                weight_decay=config["weight_decay"],
                logging_dir=f"./logs/{job_id}",
                logging_steps=config["eval_steps"],
                evaluation_strategy="steps",
                eval_steps=config["eval_steps"],
                save_steps=config["save_steps"],
                learning_rate=config["learning_rate"]
            )
            
            # Crear trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_data["training_data"],
                eval_dataset=training_data["validation_data"]
            )
            
            # Entrenar
            train_result = trainer.train()
            
            # Evaluar
            eval_result = trainer.evaluate()
            
            # Guardar modelo
            model_path = f"./models/{job_id}/final"
            trainer.save_model(model_path)
            
            # Actualizar estado y métricas
            self._update_training_results(
                job_id,
                model_name,
                train_result,
                eval_result
            )
            
        except Exception as e:
            logger.error(f"Error in fine-tuning job {job_id}: {e}")
            self._handle_training_error(job_id, str(e))

    def _update_training_results(
        self,
        job_id: str,
        model_name: str,
        train_result: Dict[str, Any],
        eval_result: Dict[str, Any]
    ) -> None:
        """Actualiza resultados de entrenamiento."""
        try:
            # Calcular métricas
            improvement = self._calculate_improvement(
                model_name,
                eval_result
            )
            
            # Actualizar estado
            job_info = self.training_state["active_jobs"][job_id]
            job_info.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "train_result": train_result,
                "eval_result": eval_result,
                "improvement": improvement
            })
            
            # Actualizar versión del modelo
            version_id = f"{model_name}_v{len(self.training_state['model_versions']) + 1}"
            self.training_state["model_versions"][version_id] = {
                "job_id": job_id,
                "base_model": model_name,
                "created_at": datetime.now().isoformat(),
                "metrics": eval_result
            }
            
            # Actualizar métricas globales
            self.metrics["total_jobs"] += 1
            self.metrics["successful_jobs"] += 1
            self.metrics["average_improvement"] = (
                self.metrics["average_improvement"] * (self.metrics["successful_jobs"] - 1) +
                improvement
            ) / self.metrics["successful_jobs"]
            
        except Exception as e:
            logger.error(f"Error updating training results: {e}")

    def _handle_training_error(self, job_id: str, error: str) -> None:
        """Maneja errores de entrenamiento."""
        try:
            # Actualizar estado del trabajo
            if job_id in self.training_state["active_jobs"]:
                self.training_state["active_jobs"][job_id].update({
                    "status": "failed",
                    "error": error,
                    "end_time": datetime.now().isoformat()
                })
            
            # Actualizar métricas
            self.metrics["total_jobs"] += 1
            self.metrics["failed_jobs"] += 1
            
        except Exception as e:
            logger.error(f"Error handling training error: {e}")

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Obtiene estado de un trabajo de fine-tuning.
        
        Args:
            job_id: ID del trabajo
        """
        try:
            if job_id not in self.training_state["active_jobs"]:
                return {"error": "Job not found"}
            
            return self.training_state["active_jobs"][job_id]
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Cancelar trabajos activos
            for job_id, job_info in self.training_state["active_jobs"].items():
                if job_info["status"] == "training":
                    logger.warning(f"Cancelling active job {job_id}")
                    job_info["status"] = "cancelled"
            
            # Limpiar estado
            self.training_state["active_jobs"].clear()
            
            logger.info("Auto Fine-tuning cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema."""
        return {
            "active_jobs": len([
                j for j in self.training_state["active_jobs"].values()
                if j["status"] == "training"
            ]),
            "model_versions": len(self.training_state["model_versions"]),
            "metrics": self.metrics
        }