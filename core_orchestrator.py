# core_orchestrator.py

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import json

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

# Nuevos imports
from edge.edge_manager import EdgeManager
from plugins.plugin_manager import PluginManager
from optimization.rl_optimizer import RLOptimizer
from optimization.prompt_adapter import PromptAdapter
from audit.blockchain_manager import BlockchainManager
from audit.decision_explainer import DecisionExplainer

# Agentes especializados
from agents.vision_agent import VisionAgent
from agents.audio_agent import AudioAgent
from agents.mlops_agent import MLOpsAgent
from agents.security_agent import SecurityAgent

logger = logging.getLogger(__name__)

class CoreOrchestrator:
    """
    Orquestador central del sistema con soporte para múltiples modelos y capacidades avanzadas.
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        edge_enabled: bool = True,
        blockchain_enabled: bool = True
    ):
        self.api_key = api_key
        self.engine_mode = engine_mode
        
        # Inicializar componentes principales
        self.edge_manager = EdgeManager() if edge_enabled else None
        self.plugin_manager = PluginManager()
        self.rl_optimizer = RLOptimizer(engine_mode)
        self.prompt_adapter = PromptAdapter(engine_mode)
        self.blockchain_manager = BlockchainManager() if blockchain_enabled else None
        self.decision_explainer = DecisionExplainer()
        
        # Configurar sistema de memoria y embeddings
        self._setup_memory_system()
        
        # Inicializar agentes especializados
        self.specialized_agents = {
            "vision": VisionAgent(api_key, engine_mode),
            "audio": AudioAgent(api_key, engine_mode),
            "mlops": MLOpsAgent(api_key, engine_mode),
            "security": SecurityAgent(api_key, engine_mode)
        }
        
        # Estado del sistema
        self.system_state = {
            "active_tasks": {},
            "resource_usage": {},
            "performance_metrics": {},
            "security_status": {}
        }

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una solicitud completa con el nuevo sistema mejorado.
        """
        try:
            # 1. Validación de seguridad inicial
            security_check = await self.specialized_agents["security"].validate_request(request)
            if not security_check["is_safe"]:
                return {"error": "Security validation failed", "details": security_check}

            # 2. Análisis y optimización de la tarea
            task_analysis = await self._analyze_task(request)
            optimized_prompt = await self.prompt_adapter.optimize(task_analysis)
            
            # 3. Determinar si usar edge computing
            if self.edge_manager and self._should_use_edge(task_analysis):
                return await self._process_on_edge(optimized_prompt)
            
            # 4. Ejecutar tarea principal
            result = await self._execute_task(optimized_prompt)
            
            # 5. Post-procesamiento y auditoría
            processed_result = await self._post_process_result(result)
            await self._audit_execution(request, processed_result)
            
            # 6. Actualizar estado y métricas
            await self._update_system_state(request, processed_result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in process_request: {e}")
            return {"error": str(e)}

    async def _analyze_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis avanzado de la tarea con optimización RL.
        """
        # Obtener recomendaciones del optimizador RL
        rl_suggestions = await self.rl_optimizer.get_suggestions(request)
        
        # Combinar con análisis de agentes especializados
        vision_analysis = await self.specialized_agents["vision"].analyze(request)
        audio_analysis = await self.specialized_agents["audio"].analyze(request)
        mlops_analysis = await self.specialized_agents["mlops"].analyze(request)
        
        return {
            "rl_suggestions": rl_suggestions,
            "specialized_analysis": {
                "vision": vision_analysis,
                "audio": audio_analysis,
                "mlops": mlops_analysis
            },
            "timestamp": datetime.now().isoformat()
        }

    def _should_use_edge(self, task_analysis: Dict[str, Any]) -> bool:
        """
        Determina si una tarea debería ejecutarse en edge.
        """
        if not self.edge_manager:
            return False
            
        criteria = {
            "data_size": task_analysis.get("data_size", 0) > 10_000_000,  # 10MB
            "latency_sensitive": task_analysis.get("latency_requirements", "low") == "high",
            "privacy_required": task_analysis.get("privacy_level", "low") == "high"
        }
        
        return any(criteria.values())

    async def _process_on_edge(self, optimized_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una tarea en edge computing.
        """
        try:
            # Seleccionar edge worker óptimo
            worker = await self.edge_manager.get_optimal_worker()
            
            # Ejecutar en edge
            result = await worker.execute(optimized_prompt)
            
            # Validar resultado
            if not await self._validate_edge_result(result):
                raise ValueError("Edge execution validation failed")
                
            return result
            
        except Exception as e:
            logger.error(f"Edge processing error: {e}")
            # Fallback a procesamiento normal
            return await self._execute_task(optimized_prompt)

    async def _execute_task(self, optimized_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una tarea con el motor seleccionado (OpenAI o DeepSeek).
        """
        try:
            # Aplicar plugins pre-ejecución
            await self.plugin_manager.apply_pre_execution_plugins(optimized_prompt)
            
            # Seleccionar motor y ejecutar
            if self.engine_mode == "deepseek":
                from deepseek_chat import DeepSeekChat
                executor = DeepSeekChat(api_key=self.api_key)
            else:
                from langchain_openai import ChatOpenAI
                executor = ChatOpenAI(api_key=self.api_key)
                
            response = await executor.agenerate([optimized_prompt])
            
            # Aplicar plugins post-ejecución
            result = await self.plugin_manager.apply_post_execution_plugins(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            raise

    async def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-procesa el resultado incluyendo explicabilidad.
        """
        try:
            # Generar explicaciones
            explanations = await self.decision_explainer.explain(result)
            
            # Agregar métricas y metadata
            processed_result = {
                **result,
                "explanations": explanations,
                "metrics": await self._calculate_metrics(result),
                "execution_metadata": {
                    "engine_mode": self.engine_mode,
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0.0"
                }
            }
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            return result

    async def _audit_execution(
        self,
        request: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Registra la ejecución en blockchain si está habilitado.
        """
        if self.blockchain_manager:
            await self.blockchain_manager.record_execution({
                "request_hash": self.blockchain_manager.hash_data(request),
                "result_hash": self.blockchain_manager.hash_data(result),
                "timestamp": datetime.now().isoformat(),
                "engine_mode": self.engine_mode
            })

    def _setup_memory_system(self) -> None:
        """
        Configura el sistema de memoria y embeddings.
        """
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        if self.engine_mode == "openai":
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        else:
            # Usar embeddings de DeepSeek
            from deepseek_chat import DeepSeekEmbeddings
            self.embeddings = DeepSeekEmbeddings(api_key=self.api_key)
            
        self.vector_store = Chroma(
            collection_name="system_memory",
            embedding_function=self.embeddings
        )

    async def _update_system_state(
        self,
        request: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Actualiza el estado del sistema y métricas.
        """
        try:
            # Actualizar métricas de MLOps
            mlops_metrics = await self.specialized_agents["mlops"].update_metrics(
                request, result
            )
            
            # Actualizar estado del sistema
            self.system_state.update({
                "last_update": datetime.now().isoformat(),
                "mlops_metrics": mlops_metrics,
                "active_tasks_count": len(self.system_state["active_tasks"]),
                "performance_metrics": await self._calculate_metrics(result)
            })
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")

    async def _calculate_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula métricas detalladas del sistema.
        """
        return {
            "response_time": result.get("execution_time", 0),
            "token_usage": result.get("token_usage", {}),
            "model_performance": await self.rl_optimizer.get_performance_metrics(),
            "edge_metrics": await self.edge_manager.get_metrics() if self.edge_manager else {},
            "resource_usage": self.system_state["resource_usage"]
        }

    async def cleanup(self) -> None:
        """
        Limpia recursos y finaliza procesos.
        """
        try:
            # Limpiar agentes especializados
            cleanup_tasks = [
                agent.cleanup() for agent in self.specialized_agents.values()
            ]
            await asyncio.gather(*cleanup_tasks)
            
            # Limpiar otros componentes
            if self.edge_manager:
                await self.edge_manager.cleanup()
            if self.blockchain_manager:
                await self.blockchain_manager.cleanup()
                
            # Limpiar memoria
            self.vector_store.delete_collection()
            
            logger.info("System cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
