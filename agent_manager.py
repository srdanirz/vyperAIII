# agent_manager.py

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from agents.agent_communication import AgentCommunicationSystem
from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Orquestador que maneja la ejecución de agentes y sus dependencias.
    """
    def __init__(self, api_key: str, engine_mode: str = "openai"):
        self.api_key = api_key
        self.engine_mode = engine_mode
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        
        # Cache simple para resultados
        self.cache = {}
        
        # Estadísticas de ejecución
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "agent_performance": {}
        }
        
        # Registro de agentes disponibles
        self.agent_registry = {
            "research": ResearchAgent,
            "analysis": AnalysisAgent,
            "contentgeneration": ContentGenerationAgent,
            "validation": ValidationAgent
        }

    async def execute(
        self,
        task: str,
        metadata: Dict[str, Any],
        workflow: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ejecuta un workflow completo de agentes.
        """
        start_time = datetime.now()
        context = {
            "task": task,
            "metadata": metadata,
            "partial_results": {},
            "active_agents": set(),
            "completed_steps": [],
            "agent_interactions": [],
            "errors": []
        }

        try:
            # Agrupar por dependencias
            dependency_groups = self._group_by_dependencies(workflow)
            
            # Ejecutar cada grupo
            for group in dependency_groups:
                tasks = []
                for step in group:
                    # Verificar cache
                    if self._is_in_cache(step):
                        context["partial_results"][step["agent"]] = self._get_cache_result(step)
                    else:
                        tasks.append(self._process_step(step, context))

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for step_def, result in zip(group, results):
                        if isinstance(result, Exception):
                            await self._on_step_fail(step_def, {"error": str(result)}, context)
                        else:
                            context["partial_results"][step_def["agent"]] = result
                            self._set_cache_result(step_def, result)
            
            # Generar resultado final
            final_result = await self._final_result(context)
            self._update_stats(True, start_time)
            
            return final_result

        except Exception as e:
            logger.error(f"Error in AgentManager execute: {e}")
            self._update_stats(False, start_time)
            return {
                "error": str(e),
                "partial_results": context["partial_results"],
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

    def _group_by_dependencies(
        self,
        workflow: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Agrupa pasos del workflow por dependencias.
        """
        groups = []
        remaining = workflow[:]
        completed = set()

        while remaining:
            # Encontrar pasos que pueden ejecutarse
            current_layer = [
                w for w in remaining
                if all(dep in completed for dep in w.get("requires", []))
            ]
            
            # Si no hay pasos disponibles, tomar el primero (evitar ciclos)
            if not current_layer:
                current_layer = [remaining[0]]

            groups.append(current_layer)
            completed.update(x["agent"] for x in current_layer)
            remaining = [x for x in remaining if x not in current_layer]

        return groups

    async def _process_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa un paso individual del workflow.
        """
        # Crear agente
        agent = self._create_agent(step, context)
        if not agent:
            raise RuntimeError(f"Cannot create agent for {step.get('agent_type')}")

        # Ejecutar agente
        return await self._execute_agent(agent, step, context)

    def _create_agent(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[BaseAgent]:
        """
        Crea una instancia de agente.
        """
        agent_type = step.get("agent_type") or step.get("agent", "research")
        agent_class = self.agent_registry.get(agent_type.lower())
        
        if not agent_class:
            logger.warning(f"Unknown agent type {agent_type}, fallback to ResearchAgent")
            agent_class = ResearchAgent

        # Obtener datos compartidos
        shared_data = {}
        for r in step.get("requires", []):
            if r in context["partial_results"]:
                shared_data[r] = context["partial_results"][r]

        # Crear instancia
        return agent_class(
            task=step.get("task", context["task"]),
            openai_api_key=self.api_key,
            metadata={
                **context["metadata"],
                "engine_mode": self.engine_mode
            },
            shared_data=shared_data
        )

    async def _execute_agent(
        self,
        agent: BaseAgent,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ejecuta un agente específico.
        """
        agent_id = step.get("agent_type", step.get("agent"))
        await self.comm_system.send_message(
            agent_id,
            f"Starting task: {step.get('task')}"
        )
        
        context["active_agents"].add(agent_id)
        start_time = datetime.now()

        try:
            result = await agent.execute()
            execution_time = (datetime.now() - start_time).total_seconds()

            # Registrar interacción
            context["agent_interactions"].append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_id,
                "task": step.get("task"),
                "success": "error" not in result,
                "execution_time": execution_time,
                "collaborators": step.get("requires", [])
            })

            # Actualizar estadísticas del agente
            self._update_agent_stats(agent_id, "error" not in result, execution_time)

            # Mensaje de estado
            status_msg = (
                "Task completed"
                if "error" not in result
                else f"Error: {result.get('error')}"
            )
            await self.comm_system.send_message(agent_id, status_msg)
            
            context["active_agents"].discard(agent_id)
            return result

        except Exception as e:
            logger.error(f"Error executing agent {agent_id}: {e}")
            context["active_agents"].discard(agent_id)
            raise

    def _update_agent_stats(
        self,
        agent_id: str,
        success: bool,
        execution_time: float
    ) -> None:
        """
        Actualiza estadísticas de rendimiento del agente.
        """
        if agent_id not in self.execution_stats["agent_performance"]:
            self.execution_stats["agent_performance"][agent_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0
            }

        stats = self.execution_stats["agent_performance"][agent_id]
        stats["total_executions"] += 1
        if success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1

        # Actualizar tiempo promedio
        prev_avg = stats["average_execution_time"]
        prev_total = stats["total_executions"] - 1
        new_avg = (prev_avg * prev_total + execution_time) / stats["total_executions"]
        stats["average_execution_time"] = new_avg

    async def _on_step_fail(
        self,
        step: Dict[str, Any],
        error_res: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """
        Maneja fallos en pasos del workflow.
        """
        await self.comm_system.send_message(
            "System",
            f"Step failed: {step.get('agent_type')} => {error_res.get('error')}",
            priority=0
        )
        
        context["errors"].append({
            "step": step,
            "error": error_res.get("error"),
            "timestamp": datetime.now().isoformat()
        })

    async def _final_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera el resultado final de la ejecución.
        """
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

    def _is_in_cache(self, step: Dict[str, Any]) -> bool:
        """Verifica si un resultado está en caché."""
        cache_key = f"{step['agent']}_{hash(str(step))}"
        return cache_key in self.cache

    def _get_cache_result(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene un resultado de caché."""
        cache_key = f"{step['agent']}_{hash(str(step))}"
        return self.cache[cache_key]

    def _set_cache_result(
        self,
        step: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Guarda un resultado en caché."""
        cache_key = f"{step['agent']}_{hash(str(step))}"
        self.cache[cache_key] = result

    def _update_stats(self, success: bool, start_time: datetime) -> None:
        """Actualiza estadísticas globales de ejecución."""
        duration = (datetime.now() - start_time).total_seconds()
        self.execution_stats["total_executions"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
            
        # Actualizar tiempo promedio
        prev_avg = self.execution_stats["average_duration"]
        prev_total = self.execution_stats["total_executions"] - 1
        new_avg = (prev_avg * prev_total + duration) / self.execution_stats["total_executions"]
        self.execution_stats["average_duration"] = new_avg

    async def cleanup(self) -> None:
        """Limpia recursos y finaliza procesos."""
        try:
            await self.comm_system.cleanup()
            self.cache.clear()
            logger.info("AgentManager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")