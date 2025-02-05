import logging
import asyncio  
from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from agents.browser_agent import BrowserAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.analysis_agent import AnalysisAgent
from agents.data_processing_agent import DataProcessingAgent
from agents.validation_agent import ValidationAgent
from agents.research_agent import ResearchAgent
from agents.agent_communication import AgentCommunicationSystem

logger = logging.getLogger(__name__)

class AgentManager:
    MODEL_CONFIG = {
        "orchestration": "gpt-4",
        "content": "gpt-4",
        "validation": "gpt-3.5-turbo",
        "research": "gpt-3.5-turbo",
        "summary": "gpt-3.5-turbo"
    }
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.comm_system = AgentCommunicationSystem(openai_api_key)
        self.cache = {}
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0
        }
        self.agent_registry = {
            'browser': BrowserAgent,
            'analysis': AnalysisAgent,
            'contentgeneration': ContentGenerationAgent,
            'research': ResearchAgent,
            'dataprocessing': DataProcessingAgent,
            'validation': ValidationAgent
        }

    async def execute(self, task: str, metadata: Dict[str, Any], workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = datetime.now()
        context = self._initialize_context(task, metadata)
        
        try:
            # Group tasks by dependency level
            dependency_groups = self._group_tasks_by_dependencies(workflow)
            
            # Process each group in parallel
            for group in dependency_groups:
                tasks = []
                for step in group:
                    if self._check_cache(step):
                        context["partial_results"][step["agent"]] = self._get_cached_result(step)
                    else:
                        tasks.append(self._process_step(step, context))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for step, result in zip(group, results):
                        if isinstance(result, Exception):
                            await self._handle_step_failure(step, {"error": str(result)}, context)
                        else:
                            context["partial_results"][step["agent"]] = result
                            self._cache_result(step, result)

            final_result = await self._compile_final_result(context)
            self._update_stats(True, start_time)
            return final_result

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            self._update_stats(False, start_time)
            return {
                "error": str(e),
                "partial_results": context.get("partial_results", {}),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

    def _initialize_context(self, task: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task": task,
            "metadata": metadata,
            "partial_results": {},
            "active_agents": set(),
            "completed_steps": [],
            "agent_interactions": [],
            "errors": []
        }

    def _group_tasks_by_dependencies(self, workflow: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups = []
        remaining = workflow.copy()
        completed = set()

        while remaining:
            current_group = []
            for task in remaining:
                if all(dep in completed for dep in task.get("requires", [])):
                    current_group.append(task)
            
            if not current_group:
                current_group = [remaining[0]]  # Handle potential cycles
            
            groups.append(current_group)
            completed.update(task["agent"] for task in current_group)
            remaining = [task for task in remaining if task not in current_group]

        return groups

    async def _process_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Create agent
            agent = self._create_agent(
                step.get("agent_type") or step.get("agent"),
                step.get("task", context["task"]),
                context["metadata"],
                self._get_required_results(step.get("requires", []), context)
            )

            if not agent:
                raise Exception(f"Could not create agent for type: {step.get('agent_type')}")

            # Execute agent
            return await self._execute_agent_with_tracking(agent, step, context)

        except Exception as e:
            raise Exception(f"Step processing error: {str(e)}")

    def _create_agent(self, agent_type: str, task: str, metadata: Dict[str, Any], partial_data: Dict[str, Any]) -> Optional[BaseAgent]:
        try:
            if not agent_type:
                return None

            agent_type = agent_type.lower().replace('_', '').replace('-', '')
            agent_class = self.agent_registry.get(agent_type)
            
            if not agent_class:
                logger.warning(f"Unknown agent type {agent_type}, defaulting to Research")
                agent_class = ResearchAgent

            return agent_class(
                task=task,
                openai_api_key=self.openai_api_key,
                metadata=metadata,
                partial_data=partial_data
            )

        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return None

    async def _execute_agent_with_tracking(self, agent: BaseAgent, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = step.get("agent_type", step.get("agent"))
        try:
            await self.comm_system.send_message(agent_id, f"Starting task: {step.get('task')}")
            context["active_agents"].add(agent_id)
            
            result = await agent.execute()
            
            context["agent_interactions"].append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_id,
                "task": step.get("task"),
                "success": "error" not in result,
                "collaborators": step.get("requires", [])
            })
            
            await self.comm_system.send_message(
                agent_id,
                "Task completed" if "error" not in result else f"Error: {result.get('error')}"
            )
            
            return result
            
        finally:
            context["active_agents"].discard(agent_id)

    def _check_cache(self, task: Dict[str, Any]) -> bool:
        cache_key = f"{task['agent']}_{hash(str(task))}"
        return cache_key in self.cache

    def _get_cached_result(self, task: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"{task['agent']}_{hash(str(task))}"
        return self.cache[cache_key]

    def _cache_result(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        cache_key = f"{task['agent']}_{hash(str(task))}"
        self.cache[cache_key] = result

    def _get_required_results(self, required_agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            agent: context["partial_results"][agent]
            for agent in required_agents
            if agent in context["partial_results"]
        }

    async def _handle_step_failure(self, step: Dict[str, Any], error_result: Dict[str, Any], context: Dict[str, Any]) -> None:
        await self.comm_system.send_message(
            "System",
            f"Step failed: {step.get('agent_type', step.get('agent'))}\nError: {error_result.get('error')}"
        )
        
        context["errors"].append({
            "step": step,
            "error": error_result.get("error"),
            "timestamp": datetime.now().isoformat()
        })

    async def _compile_final_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "result": context["partial_results"],
            "metadata": {
                **context["metadata"],
                "completion_time": datetime.now().isoformat(),
                "execution_stats": self.execution_stats,
                "agent_interactions": context["agent_interactions"],
                "errors": context["errors"]
            }
        }

    def _update_stats(self, success: bool, start_time: datetime) -> None:
        duration = (datetime.now() - start_time).total_seconds()
        self.execution_stats["total_executions"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
            
        total_prev = (self.execution_stats["average_duration"] * 
                     (self.execution_stats["total_executions"] - 1))
        self.execution_stats["average_duration"] = (total_prev + duration) / self.execution_stats["total_executions"]