import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from config.config import get_config
from team_interfaces import TeamRequest, TeamResponse, Team, TeamMember, validate_config
from agents.agent_communication import AgentCommunicationSystem
from monitoring.monitoring_manager import MonitoringManager

logger = logging.getLogger(__name__)

class DynamicTeamManager:
    def __init__(self, api_key: str, engine_mode: str = "openai"):
        self.api_key = api_key
        self.engine_mode = engine_mode
        self.config = get_config()
        
        if not validate_config(self.config.get_all()):
            raise ValueError("Invalid configuration")
            
        self.teams: Dict[str, Team] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        self.monitoring = MonitoringManager()
        
    async def process_request(
        self, 
        request: Union[Dict[str, Any], TeamRequest]
    ) -> Dict[str, Any]:
        try:
            if isinstance(request, dict):
                request = TeamRequest(**request)
                
            start_time = datetime.now()
            
            analysis = await self._analyze_requirements(request)
            teams = await self._create_teams(analysis)
            
            task_id = str(uuid.uuid4())
            self.active_tasks[task_id] = {
                "request": request,
                "teams": [t.id for t in teams],
                "status": "running",
                "start_time": start_time
            }
            
            results = await self._execute_task(task_id, teams, request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            await self._update_metrics(task_id, results, execution_time)
            
            response = TeamResponse(
                status="success",
                result=results,
                execution_time=execution_time,
                team_ids=[t.id for t in teams],
                metadata={
                    "task_id": task_id,
                    "engine_mode": self.engine_mode
                }
            )
            
            return response.__dict__
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": 0,
                "team_ids": []
            }
            
    async def _analyze_requirements(self, request: TeamRequest) -> Dict[str, Any]:
        try:
            roles_needed = []
            for role, info in self.config.get("agent_roles", {}).items():
                if any(skill in request.requirements for skill in info.get("skills", [])):
                    roles_needed.append(role)
                    
            if not roles_needed:
                roles_needed = ["research", "analysis", "validation"]
                
            return {
                "roles": roles_needed,
                "priority": request.priority,
                "metadata": request.metadata
            }
        except Exception as e:
            logger.error(f"Error analyzing requirements: {e}")
            raise
            
    async def _create_teams(self, analysis: Dict[str, Any]) -> List[Team]:
        try:
            teams = []
            team_id = str(uuid.uuid4())
            
            members = {}
            for role in analysis["roles"]:
                member_id = str(uuid.uuid4())
                role_config = self.config.get(f"agent_roles.{role}")
                
                members[member_id] = TeamMember(
                    id=member_id,
                    role=role,
                    capabilities=role_config.get("skills", [])
                )
                
            team = Team(
                id=team_id,
                name=f"Team-{team_id[:8]}",
                members=members,
                objectives=["Process request and generate response"]
            )
            
            self.teams[team_id] = team
            teams.append(team)
            
            return teams
            
        except Exception as e:
            logger.error(f"Error creating teams: {e}")
            raise
            
    async def _execute_task(
        self,
        task_id: str,
        teams: List[Team],
        request: TeamRequest
    ) -> Dict[str, Any]:
        try:
            results = {}
            for team in teams:
                team_result = await self._process_with_team(team, request)
                results[team.id] = team_result
                
            return self._combine_results(results)
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            raise
            
    async def _process_with_team(
        self,
        team: Team,
        request: TeamRequest
    ) -> Dict[str, Any]:
        try:
            results = {}
            for member_id, member in team.members.items():
                result = await self.comm_system.send_message(
                    from_agent=member.role,
                    content=request.prompt,
                    metadata=request.metadata
                )
                results[member_id] = result
                
            return {
                "status": "success",
                "team_id": team.id,
                "member_results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing with team {team.id}: {e}")
            raise
            
    def _combine_results(self, team_results: Dict[str, Any]) -> Dict[str, Any]:
        try:
            combined = {
                "summary": "",
                "details": team_results
            }
            
            summaries = []
            for team_id, result in team_results.items():
                if isinstance(result, dict) and "member_results" in result:
                    for member_result in result["member_results"].values():
                        if isinstance(member_result, str):
                            summaries.append(member_result)
                            
            combined["summary"] = "\n\n".join(summaries)
            return combined
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return {"error": str(e)}
            
    async def _update_metrics(
        self,
        task_id: str,
        results: Dict[str, Any],
        execution_time: float
    ) -> None:
        try:
            await self.monitoring.record_metrics({
                "task_id": task_id,
                "execution_time": execution_time,
                "teams_involved": len(results.get("details", {})),
                "status": "success" if "error" not in results else "error"
            })
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    async def cleanup(self) -> None:
        try:
            await self.comm_system.cleanup()
            await self.monitoring.cleanup()
            
            for team in self.teams.values():
                for member in team.members.values():
                    member.active_tasks.clear()
                    
            self.teams.clear()
            self.active_tasks.clear()
            
            logger.info("Team manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")