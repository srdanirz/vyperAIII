import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
from pathlib import Path
import json

from .errors import ProcessingError, handle_errors, ErrorBoundary
from .cache import CacheManager
from .managers.team_manager import TeamManager, TeamRole
from agents.agent_communication import AgentCommunicationSystem
from monitoring.monitoring_manager import MonitoringManager
from optimization.auto_finetuning import AutoFineTuner
from audit.blockchain_manager import BlockchainManager
from audit.decision_explainer import DecisionExplainer
from edge.edge_manager import EdgeManager
from plugins.plugin_manager import PluginManager
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
        self.engine_mode = engine_mode
        
        # Cargar configuración
        self.config = get_config()
        
        # Cache system
        self.cache = CacheManager()
        
        # Inicializar componentes core
        self.team_manager = TeamManager(api_key, engine_mode)
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
            "resource_usage": {},
            "performance_metrics": {},
            "error_count": 0,
            "last_error": None
        }
        
        # Métricas
        self.metrics = {
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "edge_processing_count": 0
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
                self.metrics["cache_hits"] += 1
                return cached

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
                    self.metrics["edge_processing_count"] += 1
                
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

                response = {
                    "status": "success",
                    "results": processed_results,
                    "explanations": explanations,
                    "teams_involved": [t.id for t in teams],
                    "execution_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "engine_mode": self.engine_mode,
                        "execution_time": execution_time,
                        "teams_performance": await self._get_teams_performance(teams)
                    }
                }

                # Guardar en caché
                await self.cache.set(cache_key, response)
                
                self.metrics["requests_processed"] += 1
                self.metrics["successful_requests"] += 1
                self._update_average_time(execution_time)
                
                return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics["failed_requests"] += 1
            self.system_state["error_count"] += 1
            self.system_state["last_error"] = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "request_type": request.get("type", "unknown")
            }
            logger.error(f"Error processing request: {e}")
            raise ProcessingError(str(e))

    @handle_errors()
    async def _analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza una solicitud para determinar necesidades."""
        try:
            # Determinar tipo de solicitud
            request_type = self._determine_request_type(request)
            
            # Analizar complejidad y requisitos
            complexity_analysis = await self._analyze_complexity(request)
            requirements = await self._extract_requirements(request)
            
            # Determinar capacidades necesarias
            capabilities = await self._determine_required_capabilities(
                request_type,
                complexity_analysis
            )
            
            # Analizar dependencias
            dependencies = await self._analyze_dependencies(requirements)
            
            return {
                "request_type": request_type,
                "complexity": complexity_analysis,
                "requirements": requirements,
                "capabilities": capabilities,
                "dependencies": dependencies,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            raise ProcessingError("Request analysis failed", {"error": str(e)})

    def _determine_request_type(self, request: Dict[str, Any]) -> str:
        """Determina el tipo de solicitud."""
        if "type" in request:
            return request["type"]
            
        # Inferir tipo basado en contenido
        content = request.get("content", "").lower()
        if any(word in content for word in ["analyze", "study", "investigate"]):
            return "analysis"
        elif any(word in content for word in ["create", "generate", "make"]):
            return "generation"
        elif any(word in content for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        else:
            return "general"

    async def _analyze_complexity(self, request: Dict[str, Any]) -> Dict[str, float]:
        """Analiza la complejidad de la solicitud."""
        try:
            content = request.get("content", "")
            return {
                "level": self._calculate_complexity_level(content),
                "factors": {
                    "length": len(content) / 1000,  # Normalizado por 1000 caracteres
                    "dependencies": len(request.get("dependencies", [])) / 10,
                    "constraints": len(request.get("constraints", [])) / 5
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return {"level": 0.5, "factors": {}}

    def _calculate_complexity_level(self, content: str) -> float:
        """Calcula nivel de complejidad entre 0 y 1."""
        # Factores de complejidad
        factors = {
            "length": min(len(content) / 1000, 1.0),
            "special_chars": len([c for c in content if not c.isalnum()]) / len(content),
            "words": len(content.split()) / 100
        }
        return min(sum(factors.values()) / len(factors), 1.0)

    async def _extract_requirements(self, request: Dict[str, Any]) -> List[str]:
        """Extrae requerimientos de la solicitud."""
        try:
            requirements = []
            
            # Requerimientos explícitos
            if "requirements" in request:
                requirements.extend(request["requirements"])
            
            # Inferir requerimientos del contenido
            content = request.get("content", "").lower()
            for keyword, req in self.config.get("requirement_keywords", {}).items():
                if keyword in content:
                    requirements.append(req)
            
            return list(set(requirements))  # Eliminar duplicados
            
        except Exception as e:
            logger.error(f"Error extracting requirements: {e}")
            return []

    async def _determine_required_capabilities(
        self,
        request_type: str,
        complexity: Dict[str, Any]
    ) -> List[str]:
        """Determina capacidades necesarias para la solicitud."""
        try:
            base_capabilities = self.config.get("base_capabilities", {}).get(request_type, [])
            additional_capabilities = []
            
            # Agregar capacidades basadas en complejidad
            if complexity["level"] > 0.7:
                additional_capabilities.extend(
                    self.config.get("complex_capabilities", [])
                )
            
            return list(set(base_capabilities + additional_capabilities))
            
        except Exception as e:
            logger.error(f"Error determining capabilities: {e}")
            return []

    async def _analyze_dependencies(self, requirements: List[str]) -> Dict[str, List[str]]:
        """Analiza dependencias entre requerimientos."""
        try:
            dependency_graph = {}
            for req in requirements:
                dependencies = []
                for other_req in requirements:
                    if other_req != req and self._is_dependent(req, other_req):
                        dependencies.append(other_req)
                dependency_graph[req] = dependencies
            return dependency_graph
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            return {}

    def _is_dependent(self, req1: str, req2: str) -> bool:
        """Determina si un requerimiento depende de otro."""
        dependency_rules = self.config.get("dependency_rules", {})
        return req2 in dependency_rules.get(req1, [])

    async def _get_teams_performance(self, teams: List[Any]) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento de los equipos."""
        try:
            performance = {}
            for team in teams:
                if hasattr(team, "get_performance"):
                    perf = await team.get_performance()
                else:
                    perf = {
                        "tasks_completed": len(team.active_tasks),
                        "success_rate": 0.0,
                        "average_time": 0.0
                    }
                performance[team.id] = perf
            return performance
        except Exception as e:
            logger.error(f"Error getting teams performance: {e}")
            return {}

    def _should_use_edge(self, analysis: Dict[str, Any]) -> bool:
        """Determina si usar procesamiento edge."""
        if not self.edge_manager:
            return False
            
        try:
            # Criterios para uso de edge
            return any([
                analysis["complexity"]["level"] > 0.8,
                len(analysis.get("requirements", [])) > 5,
                "real_time" in analysis.get("requirements", []),
                analysis.get("request_type") in ["streaming", "real_time"]
            ])
        except Exception as e:
            logger.error(f"Error determining edge usage: {e}")
            return False

    async def _process_on_edge(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Procesa tareas en nodos edge."""
        try:
            if not self.edge_manager:
                raise ProcessingError("Edge processing not available")
                
            # Agrupar tareas por tipo
            task_groups = self._group_tasks_by_type(tasks)
            
            results = {}
            for task_type, group_tasks in task_groups.items():
                result = await self.edge_manager.process_tasks(
                    task_type,
                    group_tasks
                )
                results[task_type] = result
                
            return results
            
        except Exception as e:
            logger.error(f"Error in edge processing: {e}")
            raise ProcessingError("Edge processing failed", {"error": str(e)})

    def _group_tasks_by_type(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Agrupa tareas por tipo para procesamiento eficiente."""
        groups = {}
        for task in tasks:
            task_type = task.get("type", "default")
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(task)
        return groups

    def _merge_edge_results(
        self,
        tasks: List[Dict[str, Any]],
        edge_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combina resultados de edge con tareas originales."""
        try:
            merged_tasks = []
            for task in tasks:
                task_type = task.get("type", "default")
                if task_type in edge_results:
                    task["edge_result"] = edge_results[task_type]
                merged_tasks.append(task)
            return merged_tasks
        except Exception as e:
            logger.error(f"Error merging edge results: {e}")
            return tasks

    async def _record_execution(
        self,
        request: Dict[str, Any],
        results: Dict[str, Any],
        explanations: Dict[str, Any]
    ) -> None:
        """Registra la ejecución en blockchain."""
        try:
            if not self.blockchain:
                return
                
            record = {
                "timestamp": datetime.now().isoformat(),
                "request": self._sanitize_for_blockchain(request),
                "results": self._sanitize_for_blockchain(results),
                "explanations": self._sanitize_for_blockchain(explanations)
            }
            
            await self.blockchain.record_action(
                "process_request",
                record
            )
            
        except Exception as e:
            logger.error(f"Error recording to blockchain: {e}")

    def _sanitize_for_blockchain(self, data: Any) -> Any:
        """Prepara datos para almacenamiento en blockchain."""
        if isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_for_blockchain(item) for item in data]
        elif isinstance(data, dict):
            return {
                str(k): self._sanitize_for_blockchain(v)
                for k, v in data.items()
                if not k.startswith('_')
            }
        else:
            return str(data)

    async def _update_metrics(
        self,
        teams: List[Any],
        results: Dict[str, Any],
        execution_time: float
    ) -> None:
        """Actualiza métricas del sistema."""
        try:
            # Actualizar métricas de equipos
            for team in teams:
                team_metrics = await team.get_metrics()
                self.system_state["active_teams"][team.id] = {
                    "status": team.status,
                    "metrics": team_metrics
                }
            
            # Actualizar métricas de rendimiento
            self.system_state["performance_metrics"].update({
                "last_execution_time": execution_time,
                "average_execution_time": self._calculate_average_time(execution_time),
                "active_tasks": len(self.system_state["pending_tasks"]),
                "completed_tasks": self.metrics["requests_processed"]
            })
            
            # Actualizar uso de recursos
            self.system_state["resource_usage"] = await self._get_resource_usage()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _calculate_average_time(self, new_time: float) -> float:
        """Calcula tiempo promedio de ejecución."""
        try:
            if self.metrics["requests_processed"] == 0:
                return new_time
                
            current_avg = self.metrics["average_processing_time"]
            total = self.metrics["requests_processed"]
            
            return (current_avg * total + new_time) / (total + 1)
            
        except Exception as e:
            logger.error(f"Error calculating average time: {e}")
            return 0.0

    async def _get_resource_usage(self) -> Dict[str, float]:
        """Obtiene uso actual de recursos."""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "active_threads": len(psutil.Process().threads())
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {}

    def _start_monitoring(self) -> None:
        """Inicia tareas de monitoreo."""
        asyncio.create_task(self._monitor_system_health())
        asyncio.create_task(self._monitor_teams_performance())
        asyncio.create_task(self._monitor_resource_usage())

    async def _monitor_system_health(self) -> None:
        """Monitorea salud general del sistema."""
        while True:
            try:
                # Verificar componentes
                health_status = {
                    "teams": await self.team_manager.get_health_status(),
                    "edge": await self.edge_manager.get_health_status() if self.edge_manager else None,
                    "plugins": await self.plugin_manager.get_health_status(),
                    "cache": await self.cache.get_stats()
                }
                
                # Verificar métricas críticas
                if self._check_critical_metrics():
                    await self._handle_critical_situation()
                
                # Actualizar estado
                self.system_state["health"] = health_status
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(60)

    async def _monitor_teams_performance(self) -> None:
        """Monitorea rendimiento de los equipos."""
        while True:
            try:
                # Obtener métricas de equipos
                for team_id, team_info in self.system_state["active_teams"].items():
                    team = await self.team_manager.get_team(team_id)
                    if team:
                        metrics = await team.get_metrics()
                        team_info["metrics"] = metrics
                        
                        # Verificar rendimiento
                        if self._check_team_performance(metrics):
                            await self._optimize_team(team)
                
                await asyncio.sleep(30)  # Verificar cada 30 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring teams: {e}")
                await asyncio.sleep(30)

    async def _monitor_resource_usage(self) -> None:
        """Monitorea uso de recursos."""
        while True:
            try:
                usage = await self._get_resource_usage()
                self.system_state["resource_usage"] = usage
                
                # Verificar límites
                if self._check_resource_limits(usage):
                    await self._optimize_resources()
                
                await asyncio.sleep(15)  # Verificar cada 15 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(15)

    def _check_critical_metrics(self) -> bool:
        """Verifica si hay métricas en estado crítico."""
        try:
            return any([
                self.system_state["error_count"] > self.config.get("max_errors", 100),
                self.metrics["failed_requests"] / max(self.metrics["requests_processed"], 1) > 0.2,
                self.system_state["resource_usage"].get("cpu_percent", 0) > 90,
                self.system_state["resource_usage"].get("memory_percent", 0) > 90
            ])
        except Exception as e:
            logger.error(f"Error checking critical metrics: {e}")
            return False

    async def _handle_critical_situation(self) -> None:
        """Maneja situaciones críticas del sistema."""
        try:
            # Pausar nuevas tareas
            self.system_state["status"] = "degraded"
            
            # Notificar situación
            await self.monitoring.send_alert(
                "critical",
                "System in critical state",
                self.system_state
            )
            
            # Intentar recuperación
            await self._attempt_recovery()
            
        except Exception as e:
            logger.error(f"Error handling critical situation: {e}")

    async def _attempt_recovery(self) -> None:
        """Intenta recuperar el sistema de estado crítico."""
        try:
            # Limpiar caché
            await self.cache.cleanup()
            
            # Reiniciar equipos problemáticos
            problem_teams = self._identify_problem_teams()
            for team_id in problem_teams:
                await self.team_manager.reset_team(team_id)
            
            # Liberar recursos
            await self._optimize_resources()
            
            # Verificar recuperación
            if not self._check_critical_metrics():
                self.system_state["status"] = "ready"
                logger.info("System recovered from critical state")
            
        except Exception as e:
            logger.error(f"Error attempting recovery: {e}")

    def _identify_problem_teams(self) -> List[str]:
        """Identifica equipos con problemas de rendimiento."""
        try:
            problem_teams = []
            for team_id, team_info in self.system_state["active_teams"].items():
                metrics = team_info.get("metrics", {})
                if metrics.get("error_rate", 0) > 0.2 or metrics.get("latency", 0) > 5000:
                    problem_teams.append(team_id)
            return problem_teams
        except Exception as e:
            logger.error(f"Error identifying problem teams: {e}")
            return []

    def _check_team_performance(self, metrics: Dict[str, Any]) -> bool:
        """Verifica si el rendimiento del equipo necesita optimización."""
        try:
            return any([
                metrics.get("error_rate", 0) > 0.1,
                metrics.get("latency", 0) > 2000,
                metrics.get("success_rate", 1.0) < 0.8
            ])
        except Exception as e:
            logger.error(f"Error checking team performance: {e}")
            return False

    async def _optimize_team(self, team: Any) -> None:
        """Optimiza un equipo con bajo rendimiento."""
        try:
            # Verificar carga
            if team.load > 0.8:
                await self.team_manager.redistribute_tasks(team.id)
            
            # Verificar recursos
            if team.resource_usage > 0.8:
                await team.optimize_resources()
            
            # Verificar errores
            if team.error_rate > 0.2:
                await team.reset_state()
            
        except Exception as e:
            logger.error(f"Error optimizing team: {e}")

    def _check_resource_limits(self, usage: Dict[str, float]) -> bool:
        """Verifica si el uso de recursos excede límites."""
        try:
            return any([
                usage.get("cpu_percent", 0) > 80,
                usage.get("memory_percent", 0) > 80,
                usage.get("disk_percent", 0) > 90
            ])
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return False

    async def _optimize_resources(self) -> None:
        """Optimiza uso de recursos del sistema."""
        try:
            # Limpiar caché si necesario
            if await self.cache.get_size() > self.config.get("cache.max_size"):
                await self.cache.cleanup()
            
            # Liberar recursos no utilizados
            for team in self.system_state["active_teams"].values():
                if not team.is_active():
                    await team.cleanup()
            
            # Optimizar almacenamiento
            await self._cleanup_old_files()
            
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")

    async def _cleanup_old_files(self) -> None:
        """Limpia archivos temporales antiguos."""
        try:
            temp_dir = Path("temp")
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    if self._is_old_file(file):
                        file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")

    def _is_old_file(self, file: Path) -> bool:
        """Determina si un archivo es antiguo."""
        try:
            age = datetime.now().timestamp() - file.stat().st_mtime
            return age > self.config.get("max_file_age", 86400)  # 24 horas
        except Exception as e:
            logger.error(f"Error checking file age: {e}")
            return False

    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Genera una clave única para cachear la respuesta."""
        request_str = json.dumps(request, sort_keys=True)
        return f"request_{hash(request_str)}"

    def _update_average_time(self, execution_time: float) -> None:
        """Actualiza el tiempo promedio de procesamiento."""
        try:
            current_avg = self.metrics["average_processing_time"]
            total_requests = self.metrics["requests_processed"]
            
            new_avg = (current_avg * total_requests + execution_time) / (total_requests + 1)
            self.metrics["average_processing_time"] = new_avg
            
        except Exception as e:
            logger.error(f"Error updating average time: {e}")

    async def cleanup(self) -> None:
        """Limpia recursos y finaliza procesos."""
        try:
            # Limpiar componentes core
            await self.team_manager.cleanup()
            if self.edge_manager:
                await self.edge_manager.cleanup()
            await self.comm_system.cleanup()
            
            # Limpiar sistemas de soporte
            await self.monitoring.cleanup()
            await self.optimizer.cleanup()
            if self.blockchain:
                await self.blockchain.cleanup()
            await self.plugin_manager.cleanup()
            
            # Limpiar caché
            await self.cache.cleanup()
            
            # Limpiar archivos temporales
            await self._cleanup_old_files()
            
            logger.info("Core Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise ProcessingError("Cleanup failed", {"error": str(e)})