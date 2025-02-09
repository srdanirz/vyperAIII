import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
from pathlib import Path
import json
import hashlib

from .errors import ProcessingError, handle_errors, ErrorBoundary
from .cache import CacheManager
from .managers.team_manager import TeamManager
from .managers.agent_manager import AgentManager
from agents.agent_communication import AgentCommunicationSystem
from monitoring.monitoring_manager import MonitoringManager
from optimization.auto_finetuning import AutoFineTuner
from audit.blockchain_manager import BlockchainManager
from audit.decision_explainer import DecisionExplainer
from edge.edge_manager import EdgeManager
from plugins.plugin_manager import PluginManager
from .interfaces import (
    RequestContext,
    ProcessingResult,
    EngineMode,
    ProcessingMode,
    Priority,
    ResourceUsage,
    PerformanceMetrics,
    create_request_context,
    create_processing_result
)
from config.config import get_config

logger = logging.getLogger(__name__)

class CoreOrchestrator:
    """
    Orquestador central con soporte para equipos dinámicos.
    
    Coordina:
    - Gestión de equipos dinámicos
    - Procesamiento distribuido
    - Optimización y monitoreo
    - Auditoría y seguridad
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        config_path: Optional[str] = None,
        enable_edge: bool = True,
        enable_blockchain: bool = True
    ):
        self.api_key = api_key
        self.engine_mode = EngineMode(engine_mode)
        
        # Cargar configuración
        self.config = get_config()
        
        # Cache system
        self.cache = CacheManager()
        
        # Inicializar componentes core
        self.team_manager = TeamManager(api_key, engine_mode)
        self.agent_manager = AgentManager(api_key, engine_mode)
        self.edge_manager = EdgeManager() if enable_edge else None
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        
        # Sistemas de soporte
        self.monitoring = MonitoringManager()
        self.optimizer = AutoFineTuner(api_key, engine_mode)
        self.blockchain = BlockchainManager() if enable_blockchain else None
        self.decision_explainer = DecisionExplainer(api_key, engine_mode)
        self.plugin_manager = PluginManager()
        
        # Estado del sistema
        self.system_state = {
            "status": "initializing",
            "active_teams": {},
            "pending_tasks": [],
            "resource_usage": ResourceUsage(),
            "performance_metrics": PerformanceMetrics(),
            "error_count": 0,
            "last_error": None
        }
        
        # Iniciar monitores
        self._start_monitoring()

    @handle_errors(error_types=(ProcessingError,))
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una solicitud completa usando equipos dinámicos.
        
        Args:
            request: Solicitud con tipo y datos
        """
        start_time = datetime.now()
        try:
            # Verificar caché
            cache_key = self._generate_cache_key(request)
            if cached := await self.cache.get(cache_key):
                self.system_state["performance_metrics"].cache_hits += 1
                return cached

            # Crear contexto de solicitud
            context = create_request_context(
                request_id=str(hash(cache_key)),
                mode=ProcessingMode(request.get("mode", "standard")),
                priority=Priority(request.get("priority", Priority.MEDIUM)),
                metadata=request.get("metadata", {}),
                engine_mode=self.engine_mode
            )

            async with ErrorBoundary(logger, "Error processing request"):
                # 1. Analizar solicitud y determinar equipos necesarios
                analysis = await self._analyze_request(request)
                required_teams = await self._determine_required_teams(analysis)
                
                # 2. Crear o asignar equipos
                teams = []
                for team_req in required_teams:
                    team = await self._get_or_create_team(
                        team_req["name"],
                        team_req["roles"],
                        team_req["objectives"]
                    )
                    teams.append(team)
                
                # 3. Distribuir trabajo
                tasks = await self._distribute_work(teams, request, analysis)
                
                # 4. Procesar en edge si es necesario
                if self.edge_manager and self._should_use_edge(analysis):
                    edge_results = await self._process_on_edge(tasks)
                    tasks = self._merge_edge_results(tasks, edge_results)
                
                # 5. Ejecutar y coordinar
                results = await self._execute_coordinated_tasks(teams, tasks)
                
                # 6. Post-procesar y optimizar
                processed_results = await self._post_process_results(results)
                
                # 7. Generar explicaciones
                explanations = await self.decision_explainer.explain_decisions(
                    request,
                    processed_results,
                    teams
                )
                
                # 8. Registrar en blockchain
                if self.blockchain:
                    await self._record_execution(request, processed_results, explanations)
                
                # 9. Actualizar métricas
                execution_time = (datetime.now() - start_time).total_seconds()
                await self._update_metrics(teams, processed_results, execution_time)

                # Crear resultado
                response = create_processing_result(
                    status="success",
                    data={
                        "results": processed_results,
                        "explanations": explanations,
                        "teams_involved": [t.id for t in teams]
                    },
                    metadata={
                        "request_id": context.request_id,
                        "timestamp": datetime.now().isoformat(),
                        "engine_mode": self.engine_mode,
                        "execution_time": execution_time,
                        "teams_performance": await self._get_teams_performance(teams)
                    }
                )

                # Guardar en caché
                await self.cache.set(cache_key, response)
                
                return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.system_state["error_count"] += 1
            self.system_state["last_error"] = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "request_type": request.get("type", "unknown")
            }
            logger.error(f"Error processing request: {e}")
            raise ProcessingError(str(e))