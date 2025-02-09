import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import uuid

from ..errors import ProcessingError, handle_errors, ErrorBoundary
from ..cache import CacheManager
from ..interfaces import (
    Team,
    TeamMember,
    RequestContext,
    EngineMode,
    Priority,
    create_team,
    create_team_member
)
from agents.agent_communication import AgentCommunicationSystem
from monitoring.monitoring_manager import MonitoringManager
from config.config import get_config

logger = logging.getLogger(__name__)

class TeamManager:
    """
    Manager for dynamic team creation and management.
    """
    
    def __init__(
        self,
        api_key: str,
        engine_mode: str = "openai"
    ):
        self.api_key = api_key
        self.engine_mode = EngineMode(engine_mode)
        self.config = get_config()
        
        # Communication and monitoring
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        self.monitoring = MonitoringManager()
        
        # Team management
        self.active_teams: Dict[str, Team] = {}
        self.team_assignments: Dict[str, List[str]] = {}  # team_id -> task_ids
        
        # Cache for team states
        self.cache = CacheManager()
        
        # Performance tracking
        self.metrics = {
            "teams_created": 0,
            "active_teams": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0.0
        }

    async def create_team(
        self,
        name: str,
        roles: List[str],
        objectives: List[str],
        context: Optional[RequestContext] = None
    ) -> Team:
        """Create a new specialized team."""
        try:
            # Generate team ID
            team_id = f"team_{uuid.uuid4().hex[:8]}"
            
            # Create team members
            members: Dict[str, TeamMember] = {}
            for role in roles:
                member_id = f"member_{uuid.uuid4().hex[:8]}"
                capabilities = self._get_role_capabilities(role)
                members[member_id] = create_team_member(
                    member_id=member_id,
                    role=role,
                    capabilities=capabilities
                )
            
            # Create team
            team = create_team(
                team_id=team_id,
                name=name,
                members=members,
                objectives=objectives
            )
            
            # Initialize team state
            self.active_teams[team_id] = team
            self.team_assignments[team_id] = []
            
            # Update metrics
            self.metrics["teams_created"] += 1
            self.metrics["active_teams"] = len(self.active_teams)
            
            logger.info(f"Created new team: {team_id} ({name})")
            return team
            
        except Exception as e:
            logger.error(f"Error creating team: {e}")
            raise ProcessingError(f"Team creation failed: {str(e)}")

    async def assign_task(
        self,
        team_id: str,
        task: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> Dict[str, Any]:
        """Assign a task to a team."""
        try:
            team = self._get_team(team_id)
            
            # Validate team capacity
            if not self._check_team_capacity(team):
                raise ProcessingError("Team at maximum capacity")
            
            # Prepare task
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            task_record = {
                "id": task_id,
                "team_id": team_id,
                "priority": priority,
                "data": task,
                "status": "assigned",
                "assigned_at": datetime.now().isoformat()
            }
            
            # Assign to team
            self.team_assignments[team_id].append(task_id)
            
            # Execute task
            result = await self._execute_team_task(team, task_record)
            
            # Update metrics
            self._update_metrics(team_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            raise ProcessingError(f"Task assignment failed: {str(e)}")

    def _get_team(self, team_id: str) -> Team:
        """Get team by ID with validation."""
        team = self.active_teams.get(team_id)
        if not team:
            raise ProcessingError(f"Team not found: {team_id}")
        return team

    def _check_team_capacity(self, team: Team) -> bool:
        """Check if team has capacity for new tasks."""
        max_tasks = self.config.get("teams.max_tasks_per_team", 10)
        return len(team.active_tasks) < max_tasks

    async def _execute_team_task(
        self,
        team: Team,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task with a team."""
        try:
            start_time = datetime.now()
            
            # Distribute task to members
            member_results = {}
            for member_id, member in team.members.items():
                result = await self.comm_system.send_message(
                    from_agent=member.role,
                    content=task["data"],
                    metadata={
                        "team_id": team.id,
                        "member_id": member_id,
                        "task_id": task["id"]
                    }
                )
                member_results[member_id] = result
            
            # Process results
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "task_id": task["id"],
                "status": "completed",
                "results": member_results,
                "execution_time": execution_time,
                "metrics": {
                    "members_participated": len(member_results),
                    "average_response_time": execution_time / len(member_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing team task: {e}")
            raise

    def _get_role_capabilities(self, role: str) -> List[str]:
        """Get capabilities for a specific role."""
        role_config = self.config.get("agent_roles", {}).get(role, {})
        return role_config.get("skills", [])

    def _update_metrics(
        self,
        team_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Update performance metrics."""
        try:
            self.metrics["completed_tasks"] += 1
            if result.get("status") == "completed":
                # Update success metrics
                current_avg = self.metrics["average_completion_time"]
                total_tasks = self.metrics["completed_tasks"]
                new_time = result["execution_time"]
                
                self.metrics["average_completion_time"] = (
                    (current_avg * (total_tasks - 1) + new_time) / total_tasks
                )
            else:
                self.metrics["failed_tasks"] += 1
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def get_team_status(self, team_id: str) -> Dict[str, Any]:
        """Get current status of a team."""
        team = self._get_team(team_id)
        return {
            "id": team.id,
            "name": team.name,
            "active_tasks": len(team.active_tasks),
            "members": len(team.members),
            "metrics": {
                "tasks_completed": len(self.team_assignments[team_id]),
                "last_active": team.metadata.get("last_active")
            }
        }

    async def dissolve_team(self, team_id: str) -> None:
        """Dissolve a team and clean up its resources."""
        try:
            team = self._get_team(team_id)
            
            # Cancel active tasks
            active_tasks = list(team.active_tasks)
            for task_id in active_tasks:
                await self._cancel_task(team_id, task_id)
            
            # Remove team
            del self.active_teams[team_id]
            del self.team_assignments[team_id]
            
            # Update metrics
            self.metrics["active_teams"] = len(self.active_teams)
            
            logger.info(f"Dissolved team: {team_id}")
            
        except Exception as e:
            logger.error(f"Error dissolving team: {e}")
            raise

    async def _cancel_task(
        self,
        team_id: str,
        task_id: str
    ) -> None:
        """Cancel a task and update records."""
        try:
            team = self._get_team(team_id)
            team.active_tasks.discard(task_id)
            
            if task_id in self.team_assignments[team_id]:
                self.team_assignments[team_id].remove(task_id)
                
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")

    async def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            # Dissolve all teams
            for team_id in list(self.active_teams.keys()):
                await self.dissolve_team(team_id)
            
            # Clean up communication system
            await self.comm_system.cleanup()
            
            # Clear cache
            await self.cache.cleanup()
            
            # Reset metrics
            self.metrics = {
                "teams_created": 0,
                "active_teams": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "average_completion_time": 0.0
            }
            
            logger.info("Team manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the manager."""
        return {
            "active_teams": len(self.active_teams),
            "total_tasks": sum(len(tasks) for tasks in self.team_assignments.values()),
            "metrics": self.metrics,
            "health": await self.get_health_status()
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the manager."""
        try:
            return {
                "status": "healthy" if self._check_health() else "degraded",
                "teams": {
                    "total": len(self.active_teams),
                    "overloaded": self._count_overloaded_teams()
                },
                "tasks": {
                    "total": sum(len(tasks) for tasks in self.team_assignments.values()),
                    "completed": self.metrics["completed_tasks"],
                    "failed": self.metrics["failed_tasks"]
                }
            }
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {"status": "error", "message": str(e)}

    def _check_health(self) -> bool:
        """Check if manager is healthy."""
        try:
            # Check error rate
            total_tasks = self.metrics["completed_tasks"] + self.metrics["failed_tasks"]
            if total_tasks > 0:
                error_rate = self.metrics["failed_tasks"] / total_tasks
                if error_rate > 0.2:  # More than 20% failure rate
                    return False
            
            # Check team load
            if self._count_overloaded_teams() > len(self.active_teams) * 0.3:  # 30% teams overloaded
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return False

    def _count_overloaded_teams(self) -> int:
        """Count number of overloaded teams."""
        max_tasks = self.config.get("teams.max_tasks_per_team", 10)
        return sum(
            1 for team in self.active_teams.values()
            if len(team.active_tasks) >= max_tasks * 0.8  # 80% of max capacity
        )