import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path

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
from core.adaptive.adaptive_system import AdaptiveSystem
from core.adaptive.adaptive_meta import AdaptiveMetaSystem
from plugins.plugin_manager import PluginManager
from .interfaces import (
    RequestContext,
    ProcessingResult,
    EngineMode,
    ProcessingMode,
    Priority,
    create_request_context,
    create_processing_result
)
from config.config import get_config

logger = logging.getLogger(__name__)

class CoreOrchestrator:
    """
    Orchestrator central con soporte para equipos dinámicos y manejo robusto de errores.
    
    Características:
    - Gestión de equipos dinámicos
    - Procesamiento distribuido
    - Optimización y monitoreo
    - Auditoría y seguridad
    - Manejo detallado de errores y recuperación
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
        
        # Crear logger específico
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        try:
            # Cargar configuración
            self.config = get_config()
            
            # Cache system
            self.cache = CacheManager()
            
            # Inicializar componentes core
            self.team_manager = TeamManager(api_key, engine_mode)
            self.agent_manager = AgentManager(api_key, engine_mode)
            self.edge_manager = EdgeManager() if enable_edge else None
            self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
            self.adaptive_system = AdaptiveSystem()
            self.meta_system = AdaptiveMetaSystem()
            
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
                "resource_usage": {},
                "performance_metrics": {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "average_response_time": 0.0
                },
                "error_count": 0,
                "last_error": None
            }
            
            # Recovery points
            self.recovery_points = {}
            
            self.logger.info("Core Orchestrator initialized successfully")
            self.system_state["status"] = "ready"
            
        except Exception as e:
            self.logger.error(f"Error initializing Core Orchestrator: {e}", exc_info=True)
            self.system_state["status"] = "error"
            raise

    @handle_errors(error_types=(ProcessingError,))
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una solicitud completa usando equipos dinámicos.
        
        Args:
            request: Solicitud con tipo y datos
            
        Returns:
            Dict con resultados y metadata
            
        Raises:
            ProcessingError: Si hay errores durante el procesamiento
        """
        start_time = datetime.now()
        request_id = None
        
        try:
            # Validar request
            if not self._validate_request(request):
                raise ProcessingError("Invalid request format")
            
            # Verificar caché
            cache_key = self._generate_cache_key(request)
            if cached := await self.cache.get(cache_key):
                self.logger.debug(f"Cache hit for key: {cache_key}")
                self.system_state["performance_metrics"]["cache_hits"] = \
                    self.system_state["performance_metrics"].get("cache_hits", 0) + 1
                return cached

            # Crear contexto de solicitud
            context = create_request_context(
                request_id=str(hash(cache_key)),
                mode=ProcessingMode(request.get("mode", "standard")),
                priority=Priority(request.get("priority", Priority.MEDIUM)),
                metadata=request.get("metadata", {}),
                engine_mode=self.engine_mode
            )
            request_id = context.request_id

            async with ErrorBoundary(self.logger, "Error processing request"):
                # 1. Analizar solicitud y determinar equipos necesarios
                self.logger.debug(f"Analyzing request: {request_id}")
                analysis = await self._analyze_request(request)
                required_teams = await self._determine_required_teams(analysis)
                
                # Crear recovery point
                self.recovery_points[request_id] = {
                    "stage": "analysis",
                    "data": {
                        "analysis": analysis,
                        "required_teams": required_teams
                    }
                }
                
                # 2. Crear o asignar equipos
                self.logger.debug(f"Creating teams for request: {request_id}")
                teams = []
                for team_req in required_teams:
                    team = await self._get_or_create_team(
                        team_req["name"],
                        team_req["roles"],
                        team_req["objectives"]
                    )
                    teams.append(team)
                
                # Actualizar recovery point
                self.recovery_points[request_id]["stage"] = "teams_created"
                self.recovery_points[request_id]["data"]["teams"] = teams
                
                # 3. Distribuir trabajo
                self.logger.debug(f"Distributing work for request: {request_id}")
                tasks = await self._distribute_work(teams, request, analysis)
                
                # 4. Procesar en edge si es necesario
                if self.edge_manager and self._should_use_edge(analysis):
                    self.logger.debug(f"Processing on edge for request: {request_id}")
                    edge_results = await self._process_on_edge(tasks)
                    tasks = self._merge_edge_results(tasks, edge_results)
                
                # 5. Ejecutar y coordinar
                self.logger.debug(f"Executing tasks for request: {request_id}")
                results = await self._execute_coordinated_tasks(teams, tasks)
                
                # 6. Post-procesar y optimizar
                self.logger.debug(f"Post-processing results for request: {request_id}")
                processed_results = await self._post_process_results(results)
                
                # 7. Generar explicaciones
                explanations = await self.decision_explainer.explain_decisions(
                    request,
                    processed_results,
                    teams
                )
                
                # 8. Registrar en blockchain si está habilitado
                if self.blockchain:
                    self.logger.debug(f"Recording to blockchain for request: {request_id}")
                    await self._record_execution(request, processed_results, explanations)
                
                # 9. Actualizar métricas
                execution_time = (datetime.now() - start_time).total_seconds()
                await self._update_metrics(teams, processed_results, execution_time)

                # Crear resultado final
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
                
                # Limpiar recovery point
                if request_id in self.recovery_points:
                    del self.recovery_points[request_id]
                
                return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.system_state["error_count"] += 1
            self.system_state["last_error"] = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "request_id": request_id,
                "request_type": request.get("type", "unknown")
            }
            
            # Log detallado del error
            self.logger.error(
                f"Error processing request {request_id}: {e}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "execution_time": execution_time,
                    "recovery_point": self.recovery_points.get(request_id)
                }
            )
            
            # Intentar recuperación si hay recovery point
            if request_id in self.recovery_points:
                try:
                    self.logger.info(f"Attempting recovery for request {request_id}")
                    recovered_result = await self._attempt_recovery(request_id)
                    if recovered_result:
                        return recovered_result
                except Exception as recovery_error:
                    self.logger.error(
                        f"Recovery failed for request {request_id}: {recovery_error}",
                        exc_info=True
                    )
            
            raise ProcessingError(f"Error processing request: {str(e)}")
            
        finally:
            # Cleanup y métricas finales
            if request_id in self.recovery_points:
                del self.recovery_points[request_id]
            
            self.system_state["performance_metrics"]["total_requests"] += 1
            if "error" not in locals():
                self.system_state["performance_metrics"]["successful_requests"] += 1
            else:
                self.system_state["performance_metrics"]["failed_requests"] += 1

    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """Valida formato y contenido de la solicitud."""
        try:
            required_fields = {"type", "content"}
            if not all(field in request for field in required_fields):
                self.logger.error(f"Missing required fields in request: {required_fields - request.keys()}")
                return False
            
            # Validar tipo
            if not isinstance(request["type"], str):
                self.logger.error("Request type must be string")
                return False
            
            # Validar contenido
            if not request["content"]:
                self.logger.error("Request content cannot be empty")
                return False
            
            # Validar metadata si existe
            if "metadata" in request and not isinstance(request["metadata"], dict):
                self.logger.error("Request metadata must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating request: {e}")
            return False

    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Genera una clave única para caché."""
        try:
            # Crear diccionario para hash
            cache_dict = {
                "type": request["type"],
                "content": request["content"],
                "engine_mode": self.engine_mode
            }
            
            # Añadir metadata relevante
            if "metadata" in request:
                relevant_meta = {
                    k: v for k, v in request["metadata"].items()
                    if k in ["priority", "mode", "requires_edge"]
                }
                if relevant_meta:
                    cache_dict["metadata"] = relevant_meta
            
            # Generar hash
            cache_str = json.dumps(cache_dict, sort_keys=True)
            return f"{self.engine_mode}_{hash(cache_str)}"
            
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return f"fallback_{datetime.now().timestamp()}"

    async def _attempt_recovery(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Intenta recuperar una solicitud fallida."""
        recovery_point = self.recovery_points[request_id]
        stage = recovery_point["stage"]
        data = recovery_point["data"]
        
        self.logger.info(f"Attempting recovery from stage: {stage}")
        
        try:
            if stage == "analysis":
                # Reintentar desde el análisis
                return await self._retry_from_analysis(data["analysis"], data["required_teams"])
                
            elif stage == "teams_created":
                # Reintentar desde la creación de equipos
                return await self._retry_from_teams(data["teams"], data["analysis"])
                
            return None
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return None

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Detener componentes en orden
            components = [
                (self.agent_manager, "Agent Manager"),
                (self.team_manager, "Team Manager"),
                (self.edge_manager, "Edge Manager"),
                (self.comm_system, "Communication System"),
                (self.cache, "Cache System"),
                (self.monitoring, "Monitoring System"),
                (self.blockchain, "Blockchain Manager"),
                (self.plugin_manager, "Plugin Manager")
            ]
            
            for component, name in components:
                if component:
                    try:
                        self.logger.debug(f"Cleaning up {name}")
                        await component.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up {name}: {e}")
            
            # Limpiar estado
            self.system_state["status"] = "cleaned"
            self.recovery_points.clear()
            
            self.logger.info("Core Orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del sistema."""
        try:
            return {
                "status": self.system_state["status"],
                "active_teams": len(self.system_state["active_teams"]),
                "pending_tasks": len(self.system_state["pending_tasks"]),
                "performance_metrics": self.system_state["performance_metrics"],
                "error_stats": {
                    "total_errors": self.system_state["error_count"],
                    "last_error": self.system_state["last_error"]
                },
                "components_status": {
                    "team_manager": await self.team_manager.get_status(),
                    "agent_manager": await self.agent_manager.get_status(),
                    "edge_manager": await self.edge_manager.get_status() if self.edge_manager else None,
                    "cache": await self.cache.get_status(),
                    "monitoring": await self.monitoring.get_status(),
                    "blockchain": await self.blockchain.get_status() if self.blockchain else None,
                    "plugins": await self.plugin_manager.get_status()
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza una solicitud para determinar necesidades."""
        try:
            start_time = datetime.now()
            self.logger.debug(f"Starting request analysis: {request['type']}")

            # Determinar tipo de procesamiento
            processing_type = self._determine_processing_type(request)
            
            # Estimar recursos necesarios
            resource_estimates = await self._estimate_resource_needs(request)
            
            # Determinar equipos necesarios
            team_requirements = await self._analyze_team_requirements(request)
            
            # Verificar requisitos especiales
            special_requirements = self._check_special_requirements(request)
            
            # Estimar complejidad
            complexity_analysis = await self._analyze_complexity(request)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "request_type": request["type"],
                "processing_type": processing_type,
                "resource_estimates": resource_estimates,
                "team_requirements": team_requirements,
                "special_requirements": special_requirements,
                "complexity_analysis": complexity_analysis,
                "analysis_time": analysis_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing request: {e}", exc_info=True)
            raise ProcessingError(f"Request analysis failed: {str(e)}")

    def _determine_processing_type(self, request: Dict[str, Any]) -> str:
        """Determina el tipo de procesamiento requerido."""
        try:
            request_type = request["type"].lower()
            content = request["content"]
            metadata = request.get("metadata", {})
            
            # Verificar procesamiento edge
            if metadata.get("requires_edge", False):
                return "edge"
                
            # Verificar procesamiento batch
            if isinstance(content, list) and len(content) > 10:
                return "batch"
                
            # Verificar streaming
            if metadata.get("stream", False):
                return "streaming"
                
            # Tipos específicos
            type_mapping = {
                "analysis": "analytical",
                "report": "document",
                "visualization": "visual",
                "calculation": "computational"
            }
            
            return type_mapping.get(request_type, "standard")
            
        except Exception as e:
            self.logger.error(f"Error determining processing type: {e}")
            return "standard"

    async def _estimate_resource_needs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Estima recursos necesarios para la solicitud."""
        try:
            # Estimar uso de CPU
            cpu_estimate = self._estimate_cpu_usage(request)
            
            # Estimar uso de memoria
            memory_estimate = self._estimate_memory_usage(request)
            
            # Estimar tiempo de procesamiento
            processing_time = self._estimate_processing_time(request)
            
            # Estimar uso de red
            network_usage = self._estimate_network_usage(request)
            
            return {
                "cpu_usage": cpu_estimate,
                "memory_usage": memory_estimate,
                "estimated_time": processing_time,
                "network_usage": network_usage,
                "total_cost": self._calculate_resource_cost(
                    cpu_estimate,
                    memory_estimate,
                    processing_time,
                    network_usage
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating resources: {e}")
            return {
                "cpu_usage": "unknown",
                "memory_usage": "unknown",
                "estimated_time": "unknown",
                "network_usage": "unknown",
                "total_cost": "unknown"
            }

    def _check_special_requirements(self, request: Dict[str, Any]) -> Dict[str, bool]:
        """Verifica requisitos especiales de la solicitud."""
        try:
            metadata = request.get("metadata", {})
            
            return {
                "requires_edge": metadata.get("requires_edge", False),
                "requires_blockchain": metadata.get("requires_blockchain", False),
                "requires_gpu": self._check_gpu_requirement(request),
                "requires_real_time": metadata.get("real_time", False),
                "requires_persistence": metadata.get("persistent", False),
                "requires_encryption": metadata.get("encrypted", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking special requirements: {e}")
            return {}

    def _check_gpu_requirement(self, request: Dict[str, Any]) -> bool:
        """Determina si la solicitud requiere GPU."""
        try:
            request_type = request["type"].lower()
            content = request["content"]
            
            # Tipos que típicamente requieren GPU
            gpu_intensive_types = {
                "image_processing",
                "video_processing",
                "machine_learning",
                "deep_learning",
                "3d_rendering"
            }
            
            if request_type in gpu_intensive_types:
                return True
                
            # Verificar contenido
            if isinstance(content, dict):
                if content.get("model_type") in ["neural_network", "deep_learning"]:
                    return True
                    
                if content.get("dataset_size", 0) > 1000000:  # 1M registros
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking GPU requirement: {e}")
            return False

    async def _analyze_complexity(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la complejidad de la solicitud."""
        try:
            # Análisis básico
            basic_complexity = self._calculate_basic_complexity(request)
            
            # Análisis de dependencias
            dependencies = await self._analyze_dependencies(request)
            
            # Análisis de riesgo
            risk_analysis = self._analyze_risk(basic_complexity, dependencies)
            
            return {
                "basic_complexity": basic_complexity,
                "dependencies": dependencies,
                "risk_level": risk_analysis["risk_level"],
                "risk_factors": risk_analysis["risk_factors"],
                "estimated_completion_time": self._estimate_completion_time(
                    basic_complexity,
                    dependencies
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing complexity: {e}")
            return {
                "basic_complexity": "unknown",
                "dependencies": [],
                "risk_level": "high",
                "risk_factors": ["analysis_failed"],
                "estimated_completion_time": None
            }

    def _calculate_basic_complexity(self, request: Dict[str, Any]) -> str:
        """Calcula complejidad básica de la solicitud."""
        try:
            score = 0
            content = request["content"]
            
            # Longitud del contenido
            if isinstance(content, str):
                score += len(content) / 1000  # 1 punto por cada 1000 caracteres
            elif isinstance(content, dict):
                score += len(json.dumps(content)) / 1000
            elif isinstance(content, list):
                score += len(content) * 0.5  # 0.5 puntos por elemento
            
            # Ajustar por tipo
            type_multipliers = {
                "analysis": 1.5,
                "report": 1.2,
                "visualization": 1.3,
                "calculation": 1.4
            }
            score *= type_multipliers.get(request["type"].lower(), 1.0)
            
            # Clasificar
            if score < 5:
                return "low"
            elif score < 15:
                return "medium"
            else:
                return "high"
                
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {e}")
            return "unknown"

    async def _analyze_dependencies(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analiza dependencias de la solicitud."""
        try:
            dependencies = []
            content = request["content"]
            
            # Analizar contenido
            if isinstance(content, dict):
                # Verificar referencias a otros recursos
                if "references" in content:
                    for ref in content["references"]:
                        dependencies.append({
                            "type": "resource",
                            "resource_id": ref,
                            "required": True
                        })
                
                # Verificar dependencias de datos
                if "data_sources" in content:
                    for source in content["data_sources"]:
                        dependencies.append({
                            "type": "data",
                            "source": source,
                            "required": True
                        })
            
            # Verificar dependencias de plugins
            required_plugins = await self.plugin_manager.get_required_plugins(request)
            for plugin in required_plugins:
                dependencies.append({
                    "type": "plugin",
                    "plugin_id": plugin,
                    "required": True
                })
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {e}")
            return []

    def _analyze_risk(
        self,
        complexity: str,
        dependencies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analiza riesgos basados en complejidad y dependencias."""
        try:
            risk_factors = []
            risk_level = "low"
            
            # Evaluar complejidad
            if complexity == "high":
                risk_factors.append("high_complexity")
                risk_level = "medium"
            
            # Evaluar dependencias
            required_deps = [d for d in dependencies if d["required"]]
            if len(required_deps) > 5:
                risk_factors.append("many_dependencies")
                risk_level = "high"
            
            # Verificar dependencias externas
            external_deps = [
                d for d in dependencies
                if d["type"] in ["resource", "data"]
            ]
            if external_deps:
                risk_factors.append("external_dependencies")
                risk_level = max(risk_level, "medium")
            
            return {
                "risk_level": risk_level,
                "risk_factors": risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk: {e}")
            return {
                "risk_level": "high",
                "risk_factors": ["analysis_failed"]
            }

    def _estimate_completion_time(
        self,
        complexity: str,
        dependencies: List[Dict[str, Any]]
    ) -> float:
        """Estima tiempo de completado en segundos."""
        try:
            # Tiempo base por complejidad
            base_times = {
                "low": 30,
                "medium": 120,
                "high": 300,
                "unknown": 600
            }
            total_time = base_times[complexity]
            
            # Añadir tiempo por dependencias
            dep_time = len(dependencies) * 30  # 30 segundos por dependencia
            
            # Ajustar por tipo de dependencias
            for dep in dependencies:
                if dep["type"] == "data":
                    dep_time += 60  # 1 minuto extra por dependencia de datos
                elif dep["type"] == "resource":
                    dep_time += 30  # 30 segundos extra por recurso
            
            return total_time + dep_time
            
        except Exception as e:
            self.logger.error(f"Error estimating completion time: {e}")
            return 600  # 10 minutos por defecto