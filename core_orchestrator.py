import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import json

from dynamic_team_manager import DynamicTeamManager, TeamRole
from agents.agent_communication import AgentCommunicationSystem
from monitoring.monitoring_manager import MonitoringManager
from optimization.auto_finetuning import AutoFineTuner
from audit.blockchain_manager import BlockchainManager
from audit.decision_explainer import DecisionExplainer
from edge.edge_manager import EdgeManager
from plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)

class CoreOrchestrator:
    """
    Orquestador central mejorado con soporte para equipos dinámicos.
    
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
        self.config = self._load_config(config_path)
        
        # Inicializar componentes core
        self.team_manager = DynamicTeamManager(api_key, engine_mode)
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
            "performance_metrics": {}
        }
        
        # Iniciar monitores
        self._start_monitoring()

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una solicitud completa usando equipos dinámicos.
        
        Args:
            request: Solicitud con tipo y datos
        """
        try:
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
            await self._update_metrics(teams, processed_results)
            
            return {
                "status": "success",
                "results": processed_results,
                "explanations": explanations,
                "teams_involved": [t.id for t in teams],
                "execution_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "engine_mode": self.engine_mode,
                    "teams_performance": await self._get_teams_performance(teams)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}

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
            raise

    async def _determine_required_teams(
        self,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Determina los equipos necesarios basado en el análisis."""
        try:
            required_teams = []
            
            # Equipo principal basado en tipo de solicitud
            main_team = {
                "name": f"Team_{analysis['request_type']}",
                "roles": self._get_roles_for_type(analysis['request_type']),
                "objectives": [
                    f"Handle {analysis['request_type']} request",
                    "Ensure quality and efficiency",
                    "Collaborate with support teams"
                ]
            }
            required_teams.append(main_team)
            
            # Equipos de soporte basados en complejidad
            if analysis['complexity']['level'] > 0.7:
                support_team = {
                    "name": "Support_Team",
                    "roles": self._get_support_roles(analysis),
                    "objectives": ["Provide specialized support", "Handle complex aspects"]
                }
                required_teams.append(support_team)
            
            # Equipos especializados basados en capacidades
            for capability in analysis['capabilities']:
                if self._needs_specialized_team(capability):
                    specialized_team = {
                        "name": f"Specialized_{capability}",
                        "roles": self._get_specialized_roles(capability),
                        "objectives": [f"Handle {capability} aspects"]
                    }
                    required_teams.append(specialized_team)
            
            return required_teams
            
        except Exception as e:
            logger.error(f"Error determining required teams: {e}")
            raise

    async def _get_or_create_team(
        self,
        name: str,
        roles: List[TeamRole],
        objectives: List[str]
    ) -> Any:
        """Obtiene un equipo existente o crea uno nuevo."""
        try:
            # Buscar equipo existente adecuado
            existing_team = await self.team_manager.find_suitable_team(
                roles,
                objectives
            )
            
            if existing_team:
                # Verificar disponibilidad
                if await self.team_manager.is_team_available(existing_team.id):
                    return existing_team
            
            # Crear nuevo equipo
            return await self.team_manager.create_team(
                name=name,
                roles=roles,
                objectives=objectives,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "creator": "CoreOrchestrator"
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting/creating team: {e}")
            raise

    async def _distribute_work(
        self,
        teams: List[Any],
        request: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Distribuye el trabajo entre los equipos."""
        try:
            tasks = []
            
            # Crear plan de distribución
            distribution_plan = await self._create_distribution_plan(
                teams,
                analysis
            )
            
            # Asignar tareas según el plan
            for team in teams:
                team_tasks = await self.team_manager.assign_task(
                    team.id,
                    {
                        "type": "request_processing",
                        "data": request,
                        "analysis": analysis,
                        "plan": distribution_plan[team.id]
                    }
                )
                tasks.extend(team_tasks)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error distributing work: {e}")
            raise

    async def _execute_coordinated_tasks(
        self,
        teams: List[Any],
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ejecuta tareas de forma coordinada entre equipos."""
        try:
            results = []
            task_groups = self._group_related_tasks(tasks)
            
            for group in task_groups:
                # Ejecutar tareas del grupo en paralelo
                group_results = await asyncio.gather(*[
                    self.team_manager.execute_task(task["team_id"], task["task_id"])
                    for task in group
                ])
                
                # Sincronizar resultados
                synchronized_results = await self._synchronize_results(
                    teams,
                    group_results
                )
                
                results.extend(synchronized_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing coordinated tasks: {e}")
            raise

    async def _post_process_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Post-procesa los resultados."""
        try:
            # Combinar resultados
            combined_results = self._combine_results(results)
            
            # Aplicar optimizaciones
            optimized_results = await self.optimizer.optimize_results(combined_results)
            
            # Validar calidad
            validated_results = await self._validate_results(optimized_results)
            
            return validated_results
            
        except Exception as e:
            logger.error(f"Error post-processing results: {e}")
            raise

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
            
            logger.info("Core Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carga configuración del sistema."""
        try:
            if config_path:
                with open(config_path) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
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
                    "plugins": await self.plugin_manager.get_health_status()
                }
                
                # Actualizar estado
                self.system_state["health"] = health_status
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(60)