from typing import Dict, List, Set, Optional, Any, Union
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import uuid

from agents.agent_communication import AgentCommunicationSystem
from agents.base_agent import BaseAgent
from monitoring.monitoring_manager import MonitoringManager
from audit.blockchain_manager import BlockchainManager
from optimization.auto_finetuning import AutoFineTuner

logger = logging.getLogger(__name__)

class TeamRole(Enum):
    """Roles disponibles para los agentes en equipos."""
    
    # Roles Creativos
    CREATIVE_DIRECTOR = "creative_director"
    VISUAL_EXPERT = "visual_expert"
    CONTENT_STRATEGIST = "content_strategist"
    NARRATIVE_DESIGNER = "narrative_designer"
    MUSIC_PRODUCER = "music_producer"
    
    # Roles de Negocio
    BUSINESS_STRATEGIST = "business_strategist"
    MARKET_ANALYST = "market_analyst"
    FINANCIAL_ADVISOR = "financial_advisor"
    SALES_DIRECTOR = "sales_director"
    OPERATIONS_EXPERT = "operations_expert"
    
    # Roles Técnicos
    TECH_ARCHITECT = "tech_architect"
    DATA_SCIENTIST = "data_scientist"
    SECURITY_SPECIALIST = "security_specialist"
    ML_ENGINEER = "ml_engineer"
    DEVOPS_EXPERT = "devops_expert"
    
    # Roles de Investigación
    RESEARCH_DIRECTOR = "research_director"
    TREND_ANALYST = "trend_analyst"
    DOMAIN_EXPERT = "domain_expert"
    INNOVATION_SCOUT = "innovation_scout"
    
    # Roles de Productividad
    PRODUCTIVITY_COACH = "productivity_coach"
    PROJECT_MANAGER = "project_manager"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    QUALITY_ASSURANCE = "quality_assurance"

@dataclass
class TeamMember:
    """Representa un miembro del equipo con su rol y capacidades."""
    
    id: str
    role: TeamRole
    agent: BaseAgent
    capabilities: Set[str] = field(default_factory=set)
    active_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    specializations: List[str] = field(default_factory=list)
    collaboration_score: float = 0.0
    last_active: datetime = field(default_factory=datetime.now)

@dataclass
class Team:
    """Representa un equipo completo con sus miembros y objetivos."""
    
    id: str
    name: str
    members: Dict[str, TeamMember] = field(default_factory=dict)
    objectives: List[str] = field(default_factory=list)
    active_projects: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    creation_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicTeamManager:
    """
    Gestor avanzado de equipos dinámicos de agentes IA.
    
    Características:
    - Creación y gestión de equipos multidisciplinarios
    - Optimización automática de composición de equipos
    - Sistema de colaboración inter-equipo
    - Monitoreo de rendimiento y adaptación
    - Auditoría de decisiones y acciones
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai",
        config_path: Optional[str] = None,
        enable_blockchain: bool = True,
        enable_monitoring: bool = True
    ):
        self.api_key = api_key
        self.engine_mode = engine_mode
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Sistemas core
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        self.monitoring = MonitoringManager() if enable_monitoring else None
        self.blockchain = BlockchainManager() if enable_blockchain else None
        self.optimizer = AutoFineTuner(api_key, engine_mode)
        
        # Estado del sistema
        self.teams: Dict[str, Team] = {}
        self.global_objectives: List[str] = []
        self.active_collaborations: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Iniciar monitores
        self._start_monitoring()

    async def create_team(
        self,
        name: str,
        roles: List[TeamRole],
        objectives: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Team:
        """
        Crea un nuevo equipo con roles y objetivos específicos.
        
        Args:
            name: Nombre del equipo
            roles: Roles requeridos
            objectives: Objetivos del equipo
            metadata: Metadatos adicionales
        """
        try:
            team_id = str(uuid.uuid4())
            
            # Crear miembros del equipo
            members = {}
            for role in roles:
                member_id = str(uuid.uuid4())
                agent = await self._create_agent_for_role(role)
                members[member_id] = TeamMember(
                    id=member_id,
                    role=role,
                    agent=agent,
                    capabilities=self._get_role_capabilities(role),
                    specializations=self._get_role_specializations(role)
                )
            
            # Crear equipo
            team = Team(
                id=team_id,
                name=name,
                members=members,
                objectives=objectives,
                metadata=metadata or {}
            )
            
            # Registrar equipo
            self.teams[team_id] = team
            
            # Inicializar colaboración
            await self._initialize_team_collaboration(team)
            
            # Auditar creación
            if self.blockchain:
                await self.blockchain.record_action(
                    "create_team",
                    {
                        "team_id": team_id,
                        "name": name,
                        "roles": [r.value for r in roles],
                        "objectives": objectives
                    }
                )
            
            logger.info(f"Created new team: {name} ({team_id})")
            return team
            
        except Exception as e:
            logger.error(f"Error creating team: {e}")
            raise

    async def assign_task(
        self,
        team_id: str,
        task: Dict[str, Any],
        priority: int = 1
    ) -> Dict[str, Any]:
        """
        Asigna una tarea a un equipo.
        
        Args:
            team_id: ID del equipo
            task: Descripción de la tarea
            priority: Prioridad de la tarea
        """
        try:
            team = self.teams.get(team_id)
            if not team:
                raise ValueError(f"Team not found: {team_id}")
            
            # Analizar tarea
            analysis = await self._analyze_task(task)
            
            # Identificar mejores miembros para la tarea
            assigned_members = await self._assign_members_to_task(team, analysis)
            
            # Crear plan de ejecución
            execution_plan = await self._create_execution_plan(
                assigned_members,
                analysis
            )
            
            # Registrar tarea
            task_id = str(uuid.uuid4())
            team.active_projects[task_id] = {
                "task": task,
                "analysis": analysis,
                "assigned_members": assigned_members,
                "execution_plan": execution_plan,
                "status": "assigned",
                "priority": priority,
                "created_at": datetime.now().isoformat()
            }
            
            # Iniciar ejecución
            asyncio.create_task(
                self._execute_task(team_id, task_id)
            )
            
            return {
                "task_id": task_id,
                "team_id": team_id,
                "assigned_members": [m.id for m in assigned_members],
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            raise

    async def optimize_team(
        self,
        team_id: str,
        optimization_goals: List[str]
    ) -> Dict[str, Any]:
        """
        Optimiza la composición y configuración de un equipo.
        
        Args:
            team_id: ID del equipo
            optimization_goals: Objetivos de optimización
        """
        try:
            team = self.teams.get(team_id)
            if not team:
                raise ValueError(f"Team not found: {team_id}")
            
            # Analizar rendimiento actual
            performance_analysis = await self._analyze_team_performance(team)
            
            # Generar recomendaciones
            recommendations = await self.optimizer.get_optimization_recommendations(
                performance_analysis,
                optimization_goals
            )
            
            # Aplicar cambios recomendados
            changes_made = await self._apply_team_optimizations(
                team,
                recommendations
            )
            
            # Registrar optimización
            if self.blockchain:
                await self.blockchain.record_action(
                    "optimize_team",
                    {
                        "team_id": team_id,
                        "goals": optimization_goals,
                        "changes": changes_made
                    }
                )
            
            return {
                "team_id": team_id,
                "optimizations_applied": changes_made,
                "new_performance_metrics": await self._get_team_metrics(team)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing team: {e}")
            raise

    async def _analyze_team_performance(self, team: Team) -> Dict[str, Any]:
        """Analiza el rendimiento de un equipo."""
        try:
            # Recolectar métricas
            member_metrics = {}
            for member_id, member in team.members.items():
                member_metrics[member_id] = {
                    "task_completion_rate": self._calculate_completion_rate(member),
                    "collaboration_score": member.collaboration_score,
                    "specialization_alignment": self._calculate_specialization_alignment(member),
                    "response_time": self._calculate_response_time(member)
                }
            
            # Analizar patrones
            patterns = await self._analyze_performance_patterns(member_metrics)
            
            # Calcular métricas globales
            global_metrics = {
                "team_efficiency": sum(m["task_completion_rate"] for m in member_metrics.values()) / len(member_metrics),
                "collaboration_index": sum(m["collaboration_score"] for m in member_metrics.values()) / len(member_metrics),
                "specialization_coverage": self._calculate_specialization_coverage(team),
                "response_time_avg": sum(m["response_time"] for m in member_metrics.values()) / len(member_metrics)
            }
            
            return {
                "member_metrics": member_metrics,
                "global_metrics": global_metrics,
                "patterns": patterns,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing team performance: {e}")
            raise

    async def _apply_team_optimizations(
        self,
        team: Team,
        recommendations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Aplica optimizaciones recomendadas al equipo."""
        try:
            changes = []
            
            # Procesar cada recomendación
            for rec in recommendations.get("recommendations", []):
                if rec["type"] == "add_member":
                    # Añadir nuevo miembro
                    new_member = await self._create_team_member(
                        rec["role"],
                        rec.get("specializations", [])
                    )
                    team.members[new_member.id] = new_member
                    changes.append({
                        "type": "member_added",
                        "member_id": new_member.id,
                        "role": new_member.role.value
                    })
                    
                elif rec["type"] == "remove_member":
                    # Remover miembro
                    member_id = rec["member_id"]
                    if member_id in team.members:
                        del team.members[member_id]
                        changes.append({
                            "type": "member_removed",
                            "member_id": member_id
                        })
                        
                elif rec["type"] == "adjust_role":
                    # Ajustar rol de un miembro
                    member = team.members.get(rec["member_id"])
                    if member:
                        old_role = member.role
                        member.role = TeamRole(rec["new_role"])
                        changes.append({
                            "type": "role_adjusted",
                            "member_id": member.id,
                            "old_role": old_role.value,
                            "new_role": member.role.value
                        })
                        
                elif rec["type"] == "enhance_capabilities":
                    # Mejorar capacidades
                    member = team.members.get(rec["member_id"])
                    if member:
                        new_capabilities = set(rec["capabilities"])
                        member.capabilities.update(new_capabilities)
                        changes.append({
                            "type": "capabilities_enhanced",
                            "member_id": member.id,
                            "new_capabilities": list(new_capabilities)
                        })
            
            # Actualizar timestamp
            team.last_modified = datetime.now()
            
            return changes
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
            raise

    def _calculate_completion_rate(self, member: TeamMember) -> float:
        """Calcula tasa de completitud de tareas de un miembro."""
        try:
            if not member.performance_metrics.get("total_tasks", 0):
                return 0.0
                
            completed = member.performance_metrics.get("completed_tasks", 0)
            total = member.performance_metrics["total_tasks"]
            
            return completed / total
            
        except Exception as e:
            logger.error(f"Error calculating completion rate: {e}")
            return 0.0

    def _calculate_specialization_alignment(self, member: TeamMember) -> float:
        """Calcula alineación entre tareas y especializaciones."""
        try:
            if not member.active_tasks or not member.specializations:
                return 0.0
                
            alignment_scores = []
            for task in member.active_tasks:
                task_keywords = self._extract_task_keywords(task)
                spec_matches = sum(
                    1 for spec in member.specializations
                    if any(keyword in spec.lower() for keyword in task_keywords)
                )
                alignment_scores.append(spec_matches / len(member.specializations))
            
            return sum(alignment_scores) / len(alignment_scores)
            
        except Exception as e:
            logger.error(f"Error calculating specialization alignment: {e}")
            return 0.0

    def _calculate_response_time(self, member: TeamMember) -> float:
        """Calcula tiempo promedio de respuesta."""
        try:
            response_times = member.performance_metrics.get("response_times", [])
            if not response_times:
                return 0.0
            return sum(response_times) / len(response_times)
        except Exception as e:
            logger.error(f"Error calculating response time: {e}")
            return 0.0

    def _calculate_specialization_coverage(self, team: Team) -> float:
        """Calcula cobertura de especializaciones del equipo."""
        try:
            required_specs = set()
            for objective in team.objectives:
                required_specs.update(self._extract_required_specializations(objective))
            
            team_specs = set()
            for member in team.members.values():
                team_specs.update(member.specializations)
            
            if not required_specs:
                return 1.0
                
            return len(team_specs.intersection(required_specs)) / len(required_specs)
            
        except Exception as e:
            logger.error(f"Error calculating specialization coverage: {e}")
            return 0.0

    async def _analyze_performance_patterns(
        self,
        metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analiza patrones en métricas de rendimiento."""
        try:
            patterns = {
                "high_performers": [],
                "needs_improvement": [],
                "collaboration_opportunities": [],
                "skill_gaps": []
            }
            
            # Identificar alto rendimiento
            avg_completion = sum(m["task_completion_rate"] for m in metrics.values()) / len(metrics)
            for member_id, member_metrics in metrics.items():
                if member_metrics["task_completion_rate"] > avg_completion * 1.2:
                    patterns["high_performers"].append({
                        "member_id": member_id,
                        "metrics": member_metrics
                    })
                elif member_metrics["task_completion_rate"] < avg_completion * 0.8:
                    patterns["needs_improvement"].append({
                        "member_id": member_id,
                        "metrics": member_metrics
                    })
            
            # Identificar oportunidades de colaboración
            for m1_id, m1_metrics in metrics.items():
                for m2_id, m2_metrics in metrics.items():
                    if m1_id != m2_id:
                        collaboration_potential = self._calculate_collaboration_potential(
                            m1_metrics, m2_metrics
                        )
                        if collaboration_potential > 0.8:
                            patterns["collaboration_opportunities"].append({
                                "members": [m1_id, m2_id],
                                "potential": collaboration_potential
                            })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")
            return {}

    async def _execute_task(self, team_id: str, task_id: str) -> None:
        """Ejecuta una tarea asignada."""
        try:
            team = self.teams[team_id]
            task_info = team.active_projects[task_id]
            
            # Notificar inicio
            await self.comm_system.broadcast_message(
                f"Starting task {task_id} execution",
                team_id=team_id
            )
            
            # Ejecutar pasos del plan
            results = []
            for step in task_info["execution_plan"]["steps"]:
                step_result = await self._execute_step(
                    team_id,
                    task_id,
                    step
                )
                results.append(step_result)
                
                if step_result.get("status") == "failed":
                    await self._handle_step_failure(
                        team_id,
                        task_id,
                        step,
                        step_result
                    )
                    break
            
            # Actualizar estado
            task_info["status"] = "completed" if all(
                r["status"] == "success" for r in results
            ) else "failed"
            
            task_info["completion_time"] = datetime.now().isoformat()
            task_info["results"] = results
            
            # Notificar finalización
            await self.comm_system.broadcast_message(
                f"Task {task_id} {task_info['status']}",
                team_id=team_id
            )
            
            # Actualizar métricas
            self._update_performance_metrics(team_id, task_id, results)
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            if task_id in team.active_projects:
                team.active_projects[task_id]["status"] = "failed"
                team.active_projects[task_id]["error"] = str(e)

    async def _execute_step(
        self,
        team_id: str,
        task_id: str,
        step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta un paso individual de una tarea."""
        try:
            team = self.teams[team_id]
            member = team.members[step["assigned_to"]]
            
            # Preparar contexto
            context = {
                "team_id": team_id,
                "task_id": task_id,
                "step_id": step["id"],
                "previous_results": step.get("dependencies_results", {}),
                "global_context": team.active_projects[task_id].get("context", {})
            }
            
            # Ejecutar agente
            start_time = datetime.now()
            result = await member.agent.execute({
                "action": step["action"],
                "parameters": step["parameters"],
                "context": context
            })
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Registrar métricas
            self._record_execution_metrics(
                member,
                execution_time,
                result.get("status", "unknown")
            )
            
            return {
                "step_id": step["id"],
                "member_id": member.id,
                "result": result,
                "execution_time": execution_time,
                "status": "success" if "error" not in result else "failed"
            }
            
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            return {
                "step_id": step["id"],
                "status": "failed",
                "error": str(e)
            }

    def _record_execution_metrics(
        self,
        member: TeamMember,
        execution_time: float,
        status: str
    ) -> None:
        """Registra métricas de ejecución de un miembro."""
        try:
            if "response_times" not in member.performance_metrics:
                member.performance_metrics["response_times"] = []
                
            member.performance_metrics["response_times"].append(execution_time)
            
            if status == "success":
                member.performance_metrics["completed_tasks"] = (
                    member.performance_metrics.get("completed_tasks", 0) + 1
                )
            
            member.performance_metrics["total_tasks"] = (
                member.performance_metrics.get("total_tasks", 0) + 1
            )
            
            # Limitar histórico de tiempos
            if len(member.performance_metrics["response_times"]) > 100:
                member.performance_metrics["response_times"] = (
                    member.performance_metrics["response_times"][-100:]
                )
                
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")

    async def get_team_status(self, team_id: str) -> Dict[str, Any]:
        """Obtiene estado detallado de un equipo."""
        try:
            team = self.teams.get(team_id)
            if not team:
                raise ValueError(f"Team not found: {team_id}")
                
            return {
                "id": team.id,
                "name": team.name,
                "members": {
                    member_id: {
                        "role": member.role.value,
                        "active_tasks": len(member.active_tasks),
                        "performance": member.performance_metrics,
                        "last_active": member.last_active.isoformat()
                    }
                    for member_id, member in team.members.items()
                },
                "active_projects": len(team.active_projects),
                "performance_metrics": await self._get_team_metrics(team),
                "objectives_status": await self._get_objectives_status(team),
                "last_modified": team.last_modified.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting team status: {e}")
            raise

    async def cleanup(self) -> None:
        """Limpia recursos y finaliza procesos."""
        try:
            # Limpiar comunicación
            await self.comm_system.cleanup()
            
            # Limpiar blockchain si está activo
            if self.blockchain:
                await self.blockchain.cleanup()
                
            # Limpiar monitoreo si está activo
            if self.monitoring:
                await self.monitoring.cleanup()
                
            # Limpiar optimizer
            await self.optimizer.cleanup()
            
            # Limpiar equipos
            for team in self.teams.values():
                for member in team.members.values():
                    await member.agent.cleanup()
            
            self.teams.clear()
            
            logger.info("Dynamic Team Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")