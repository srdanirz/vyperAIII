# vyper_core.py

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Importar todos los componentes
from core_orchestrator import CoreOrchestrator
from agents.vision_agent import VisionAgent
from agents.audio_agent import AudioAgent
from agents.mlops_agent import MLOpsAgent
from agents.security_agent import SecurityAgent
from optimization.auto_finetuning import AutoFineTuner
from audit.decision_explainer import DecisionExplainer
from audit.blockchain_manager import BlockchainManager
from edge.edge_manager import EdgeManager
from monitoring.monitoring_manager import MonitoringManager
from plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)

class VyperAI:
    """
    Sistema central de VyperAI que coordina todos los componentes.
    
    Flujo de trabajo:
    1. Validación de seguridad
    2. Procesamiento en edge si es necesario
    3. Ejecución de agentes especializados
    4. Optimización y monitoreo
    5. Explicabilidad y auditoría
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        config_path: Optional[str] = None
    ):
        self.api_key = api_key
        self.engine_mode = engine_mode
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Inicializar componentes core
        self._initialize_core_components()
        
        # Estado del sistema
        self.system_state = {
            "status": "initializing",
            "active_tasks": {},
            "components_health": {}
        }
        
        # Iniciar monitoreo
        self._start_monitoring()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carga configuración del sistema."""
        try:
            if config_path:
                with open(config_path) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _initialize_core_components(self) -> None:
        """Inicializa todos los componentes principales."""
        try:
            # Core components
            self.orchestrator = CoreOrchestrator(self.api_key, self.engine_mode)
            self.plugin_manager = PluginManager()
            self.monitoring = MonitoringManager()
            
            # Specialized agents
            self.agents = {
                "vision": VisionAgent(self.api_key, self.engine_mode),
                "audio": AudioAgent(self.api_key, self.engine_mode),
                "mlops": MLOpsAgent(self.api_key, self.engine_mode),
                "security": SecurityAgent(self.api_key, self.engine_mode)
            }
            
            # Advanced features
            self.edge_manager = EdgeManager()
            self.auto_finetuner = AutoFineTuner(self.api_key, self.engine_mode)
            self.explainer = DecisionExplainer(self.api_key, self.engine_mode)
            self.blockchain = BlockchainManager()
            
            # Update system state
            self.system_state["status"] = "ready"
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.system_state["status"] = "error"
            raise

    def _start_monitoring(self) -> None:
        """Inicia tareas de monitoreo."""
        asyncio.create_task(self._monitor_system_health())
        asyncio.create_task(self._monitor_performance())

    async def process_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa una solicitud completa.
        
        Args:
            request: Solicitud con tipo y datos
        """
        try:
            # 1. Validación de seguridad
            security_check = await self.agents["security"].validate_request(request)
            if not security_check["is_safe"]:
                return {"error": "Security validation failed", "details": security_check}

            # 2. Verificar procesamiento edge
            if self._should_use_edge(request):
                return await self._process_on_edge(request)

            # 3. Procesamiento principal
            start_time = datetime.now()
            
            # Determinar tipo de procesamiento
            processing_type = self._determine_processing_type(request)
            
            # Ejecutar agentes necesarios
            results = await self._execute_required_agents(request, processing_type)
            
            # 4. Optimización y monitoreo
            await self._optimize_and_monitor(results)
            
            # 5. Explicabilidad y auditoría
            explanation = await self.explainer.explain_decision(results, request)
            await self.blockchain.record_action("process_request", {
                "request": request,
                "results": results,
                "explanation": explanation
            })
            
            # Preparar respuesta
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "results": results,
                "explanation": explanation,
                "execution_time": execution_time,
                "metadata": {
                    "processing_type": processing_type,
                    "engine_mode": self.engine_mode,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}

    def _determine_processing_type(
        self,
        request: Dict[str, Any]
    ) -> List[str]:
        """Determina qué tipos de procesamiento se necesitan."""
        processing_types = []
        
        # Verificar contenido visual
        if self._has_visual_content(request):
            processing_types.append("vision")
            
        # Verificar contenido de audio
        if self._has_audio_content(request):
            processing_types.append("audio")
            
        # Siempre incluir security
        processing_types.append("security")
        
        return processing_types

    async def _execute_required_agents(
        self,
        request: Dict[str, Any],
        processing_types: List[str]
    ) -> Dict[str, Any]:
        """Ejecuta los agentes necesarios."""
        results = {}
        tasks = []
        
        for proc_type in processing_types:
            if proc_type in self.agents:
                tasks.append(self._execute_agent(
                    self.agents[proc_type],
                    request
                ))
        
        if tasks:
            agent_results = await asyncio.gather(*tasks)
            for proc_type, result in zip(processing_types, agent_results):
                results[proc_type] = result
                
        return results

    async def _optimize_and_monitor(
        self,
        results: Dict[str, Any]
    ) -> None:
        """Realiza optimización y monitoreo."""
        try:
            # Verificar necesidad de fine-tuning
            if await self.auto_finetuner.check_finetuning_need(
                self.engine_mode,
                results
            ):
                asyncio.create_task(self._start_finetuning())
            
            # Actualizar métricas
            await self.monitoring.record_metrics(results)
            
        except Exception as e:
            logger.error(f"Error in optimization and monitoring: {e}")

    async def _monitor_system_health(self) -> None:
        """Monitorea salud del sistema."""
        while True:
            try:
                health_status = {}
                
                # Verificar componentes core
                health_status["core"] = await self._check_core_health()
                
                # Verificar agentes
                health_status["agents"] = await self._check_agents_health()
                
                # Verificar edge nodes
                health_status["edge"] = await self.edge_manager.get_status()
                
                # Actualizar estado
                self.system_state["components_health"] = health_status
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(60)

    async def _monitor_performance(self) -> None:
        """Monitorea rendimiento del sistema."""
        while True:
            try:
                # Recolectar métricas
                metrics = {
                    "orchestrator": await self.orchestrator.get_metrics(),
                    "agents": {
                        name: await agent.get_metrics()
                        for name, agent in self.agents.items()
                    },
                    "edge": await self.edge_manager.get_metrics(),
                    "blockchain": await self.blockchain.get_metrics()
                }
                
                # Actualizar monitoring
                await self.monitoring.update_metrics(metrics)
                
                await asyncio.sleep(30)  # Actualizar cada 30 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(30)

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Limpiar componentes core
            await self.orchestrator.cleanup()
            await self.plugin_manager.cleanup()
            await self.monitoring.cleanup()
            
            # Limpiar agentes
            for agent in self.agents.values():
                await agent.cleanup()
            
            # Limpiar características avanzadas
            await self.edge_manager.cleanup()
            await self.auto_finetuner.cleanup()
            await self.explainer.cleanup()
            await self.blockchain.cleanup()
            
            logger.info("VyperAI cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema."""
        return {
            "status": self.system_state["status"],
            "components": {
                "core": {
                    "orchestrator": await self.orchestrator.get_status(),
                    "plugins": await self.plugin_manager.get_status(),
                    "monitoring": await self.monitoring.get_status()
                },
                "agents": {
                    name: await agent.get_status()
                    for name, agent in self.agents.items()
                },
                "advanced": {
                    "edge": await self.edge_manager.get_status(),
                    "finetuning": await self.auto_finetuner.get_status(),
                    "explainer": await self.explainer.get_status(),
                    "blockchain": await self.blockchain.get_status()
                }
            },
            "active_tasks": len(self.system_state["active_tasks"]),
            "health": self.system_state["components_health"]
        }