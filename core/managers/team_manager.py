import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import uuid

from ..errors import ErrorBoundary, handle_errors, ProcessingError
from ..cache import CacheManager
from ..interfaces import (
    Team,
    TeamMember,
    RequestContext,
    ProcessingResult,
    Priority
)
from agents.agent_communication import AgentCommunicationSystem
from monitoring.monitoring_manager import MonitoringManager
from config.config import get_config

logger = logging.getLogger(__name__)

class TeamManager:
    """
    Gestor de equipos dinámicos con gestión de recursos y manejo de errores.
    """
    
    def __init__(self, api_key: str, engine_mode: str = "openai"):
        self.api_key = api_key
        self.engine_mode = engine_mode
        self.config = get_config()
        self.cache = CacheManager()
        
        # Sistemas de soporte
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        self.monitoring = MonitoringManager()
        
        # Estado y equipos
        self.teams: Dict[str, Team] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Métricas
        self.performance_metrics = {
            "teams_created": 0,
            "active_teams": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0.0
        }

    @handle_errors()
    async def process_request(
        self,
        request: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Procesa una solicitud usando equipos dinámicos.
        
        Args:
            request: Solicitud a procesar
            
        Returns:
            ProcessingResult con resultados y metadatos
        """
        start_time = datetime.now()
        
        try:
            # Crear contexto
            context = RequestContext(
                request_id=str(uuid.uuid4()),
                metadata=request.get("metadata", {}),
                priority=Priority(request.get("priority", Priority.MEDIUM))
            )
            
            # Analizar requerimientos
            analysis = await self._analyze_requirements(request)
            
            # Crear o asignar equipos
            teams = await self._create_teams(analysis, context)
            
            # Procesar con equipos
            results = await self._process_with_teams(teams, request, context)
            
            # Actualizar métricas
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(True, execution_time)
            
            return ProcessingResult(
                status="success",
                data=results,
                metadata={
                    "request_id": context.request_id,
                    "execution_time": execution_time,
                    "teams": [team.id for team in teams]
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self._update_metrics(False, (datetime.now() - start_time).total_seconds())
            raise ProcessingError("Request processing failed", {"error": str(e)})

    async def _analyze_requirements(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza requerimientos de la solicitud."""
        try:
            # Determinar roles necesarios
            roles_needed = []
            for role, info in self.config.get("agent_roles", {}).items():
                if any(skill in request.get("requirements", []) for skill in info.get("skills", [])):
                    roles_needed.append(role)
            
            if not roles_needed:
                roles_needed = ["research", "analysis", "validation"]
            
            return {
                "roles": roles_needed,
                "priority": request.get("priority", Priority.MEDIUM),
                "metadata": request.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {e}")
            raise ProcessingError("Failed to analyze requirements", {"error": str(e)})

    async def _create_teams(
        self,
        analysis: Dict[str, Any],
        context: RequestContext
    ) -> List[Team]:
        """Crea o asigna equipos basados en el análisis."""
        try:
            teams = []
            
            # Intentar reutilizar equipos existentes
            for role in analysis["roles"]:
                existing_team = await self._find_suitable_team(role, context)
                if existing_team:
                    teams.append(existing_team)
                    continue
                
                # Crear nuevo equipo si no hay uno disponible
                new_team = await self._create_new_team(role, context)
                teams.append(new_team)
            
            self.performance_metrics["teams_created"] += len(teams)
            self.performance_metrics["active_teams"] = len(self.teams)
            
            return teams
            
        except Exception as e:
            logger.error(f"Error creating teams: {e}")
            raise ProcessingError("Failed to create teams", {"error": str(e)})

    async def _find_suitable_team(
        self,
        role: str,
        context: RequestContext
    ) -> Optional[Team]:
        """Busca un equipo existente adecuado."""
        for team in self.teams.values():
            if (
                # Verificar si el equipo tiene el rol necesario
                any(member.role == role for member in team.members.values()) and
                # Verificar disponibilidad
                len(team.active_tasks) < self.config.get("teams.max_tasks", 5) and
                # Verificar prioridad
                context.priority.value <= Priority.MEDIUM
            ):
                return team
        return None

    async def _create_new_team(
        self,
        role: str,
        context: RequestContext
    ) -> Team:
        """Crea un nuevo equipo."""
        team_id = str(uuid.uuid4())
        
        # Crear miembros del equipo
        members = {}
        role_config = self.config.get(f"agent_roles.{role}")
        
        # Miembro principal
        member_id = str(uuid.uuid4())
        members[member_id] = TeamMember(
            id=member_id,
            role=role,
            capabilities=role_config.get("skills", [])
        )
        
        # Miembros de soporte si es necesario
        if context.priority >= Priority.HIGH:
            support_roles = role_config.get("support_roles", [])
            for support_role in support_roles:
                support_id = str(uuid.uuid4())
                members[support_id] = TeamMember(
                    id=support_id,
                    role=support_role,
                    capabilities=self.config.get(f"agent_roles.{support_role}.skills", [])
                )
        
        # Crear y registrar equipo
        team = Team(
            id=team_id,
            name=f"Team-{role}-{team_id[:8]}",
            members=members,
            objectives=[
                f"Handle {role} tasks",
                "Ensure quality and efficiency",
                "Collaborate with other teams"
            ]
        )
        
        self.teams[team_id] = team
        return team

    async def _process_with_teams(
        self,
        teams: List[Team],
        request: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Procesa una solicitud usando los equipos asignados."""
        try:
            results = {}
            tasks = []
            
            # Distribuir trabajo entre equipos
            for team in teams:
                task = asyncio.create_task(
                    self._execute_team_task(team, request, context)
                )
                tasks.append((team.id, task))
            
            # Esperar resultados
            for team_id, task in tasks:
                try:
                    result = await task
                    results[team_id] = result
                    self.performance_metrics["completed_tasks"] += 1
                except Exception as e:
                    logger.error(f"Error in team {team_id}: {e}")
                    results[team_id] = {"error": str(e)}
                    self.performance_metrics["failed_tasks"] += 1
            
            return {
                "team_results": results,
                "metadata": {
                    "teams_involved": len(teams),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing with teams: {e}")
            raise ProcessingError("Team processing failed", {"error": str(e)})

    async def _execute_team_task(
        self,
        team: Team,
        request: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Ejecuta una tarea con un equipo específico."""
        try:
            task_id = str(uuid.uuid4())
            
            # Registrar tarea activa
            self.active_tasks[task_id] = {
                "team_id": team.id,
                "status": "running",
                "start_time": datetime.now().isoformat()
            }
            team.active_tasks.add(task_id)
            
            # Ejecutar con cada miembro
            member_results = {}
            for member_id, member in team.members.items():
                result = await self.comm_system.send_message(
                    from_agent=member.role,
                    content=request.get("prompt", ""),
                    metadata={
                        **context.metadata,
                        "team_id": team.id,
                        "member_id": member_id
                    }
                )
                member_results[member_id] = result
            
            # Limpiar estado
            team.active_tasks.remove(task_id)
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "member_results": member_results,
                "metadata": {
                    "task_id": task_id,
                    "team_id": team.id
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing team task: {e}")
            if task_id in team.active_tasks:
                team.active_tasks.remove(task_id)
            raise

    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """Actualiza métricas de rendimiento."""
        total = self.performance_metrics["completed_tasks"] + self.performance_metrics["failed_tasks"]
        
        if total > 0:
            prev_avg = self.performance_metrics["average_completion_time"]
            new_avg = (prev_avg * (total - 1) + execution_time) / total
            self.performance_metrics["average_completion_time"] = new_avg

    async def cleanup(self) -> None:
        """Limpia recursos del gestor."""
        try:
            await self.comm_system.cleanup()
            
            for team in self.teams.values():
                team.active_tasks.clear()
                
            self.teams.clear()
            self.active_tasks.clear()
            
            await self.cache.cleanup()
            logger.info("Team manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del gestor."""
        return {
            "teams": {
                "total": len(self.teams),
                "active": len([t for t in self.teams.values() if t.active_tasks])
            },
            "tasks": {
                "active": len(self.active_tasks),
                "completed": self.performance_metrics["completed_tasks"],
                "failed": self.performance_metrics["failed_tasks"]
            },
            "performance": self.performance_metrics,
            "cache_status": await self.cache.get_status()
        }