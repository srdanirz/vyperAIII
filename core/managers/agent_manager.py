import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..errors import ErrorBoundary, handle_errors, ProcessingError
from ..cache import CacheManager
from ..interfaces import RequestContext, ProcessingResult
from agents.agent_communication import AgentCommunicationSystem
from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.validation_agent import ValidationAgent
from monitoring.monitoring_manager import MonitoringManager

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Gestor central de agentes con manejo robusto de recursos y errores.
    """
    
    def __init__(self, api_key: str, engine_mode: str = "openai"):
        self.api_key = api_key
        self.engine_mode = engine_mode
        self.cache = CacheManager()
        self.comm_system = AgentCommunicationSystem(api_key, engine_mode)
        self.monitoring = MonitoringManager()
        
        # Agentes disponibles
        self.agent_registry = {
            "research": ResearchAgent,
            "analysis": AnalysisAgent,
            "content_generation": ContentGenerationAgent,
            "validation": ValidationAgent
        }
        
        # Estado y métricas
        self.active_agents: Dict[str, BaseAgent] = {}
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "agent_performance": {}
        }

    async def execute(
        self,
        task: str,
        metadata: Dict[str, Any],
        workflow: List[Dict[str, Any]]
    ) -> ProcessingResult:
        """
        Ejecuta un workflow completo de agentes.
        
        Args:
            task: Descripción de la tarea
            metadata: Metadatos adicionales
            workflow: Lista de pasos del workflow
            
        Returns:
            ProcessingResult con resultados y metadatos
        """
        context = RequestContext(
            request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            metadata=metadata
        )
        
        start_time = datetime.now()
        
        try:
            # Agrupar por dependencias
            dependency_groups = self._group_by_dependencies(workflow)
            
            results = {
                "partial_results": {},
                "completed_steps": [],
                "agent_interactions": [],
                "errors": []
            }
            
            # Ejecutar cada grupo
            for group in dependency_groups:
                await self._process_group(group, results, context)
            
            # Generar resultado final
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(True, execution_time)
            
            return ProcessingResult(
                status="success",
                data=results,
                metadata={
                    "request_id": context.request_id,
                    "execution_time": execution_time,
                    "engine_mode": self.engine_mode
                },
                errors=results["errors"]
            )
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            self._update_stats(False, (datetime.now() - start_time).total_seconds())
            raise ProcessingError("Workflow execution failed", {"error": str(e)})

    @handle_errors()
    async def _process_group(
        self,
        group: List[Dict[str, Any]],
        results: Dict[str, Any],
        context: RequestContext
    ) -> None:
        """Procesa un grupo de tareas en paralelo."""
        tasks = []
        
        for step in group:
            # Verificar caché
            cache_key = self._get_cache_key(step)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                results["partial_results"][step["agent"]] = cached_result
                continue
            
            # Crear y ejecutar tarea
            task = asyncio.create_task(
                self._process_step(step, context)
            )
            tasks.append((step, task))
        
        # Esperar resultados
        for step, task in tasks:
            try:
                result = await task
                results["partial_results"][step["agent"]] = result
                results["completed_steps"].append(step)
                await self.cache.set(self._get_cache_key(step), result)
                
            except Exception as e:
                logger.error(f"Error processing step {step}: {e}")
                results["errors"].append({
                    "step": step,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

    async def _process_step(
        self,
        step: Dict[str, Any],
        context: RequestContext
    ) -> Dict[str, Any]:
        """Procesa un paso individual del workflow."""
        agent = await self._get_or_create_agent(step, context)
        if not agent:
            raise ProcessingError(f"Could not create agent for step: {step}")
        
        try:
            # Registrar inicio
            start_time = datetime.now()
            self.monitoring.record_event(
                "step_started",
                {"step": step, "agent": agent.__class__.__name__}
            )
            
            # Ejecutar agente
            result = await agent.execute()
            
            # Registrar fin exitoso
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_agent_stats(agent.__class__.__name__, True, execution_time)
            
            return result
            
        except Exception as e:
            # Registrar error
            self._update_agent_stats(agent.__class__.__name__, False, 0)
            raise ProcessingError(f"Step execution failed: {e}")
        
        finally:
            # Cleanup si es necesario
            if agent.id not in self.active_agents:
                await agent.cleanup()

    async def _get_or_create_agent(
        self,
        step: Dict[str, Any],
        context: RequestContext
    ) -> Optional[BaseAgent]:
        """Obtiene o crea un agente para un paso."""
        agent_type = step.get("agent_type") or step.get("agent", "research")
        agent_class = self.agent_registry.get(agent_type.lower())
        
        if not agent_class:
            logger.warning(f"Unknown agent type {agent_type}, using ResearchAgent")
            agent_class = ResearchAgent
        
        # Obtener datos compartidos
        shared_data = {}
        for r in step.get("requires", []):
            if r in context.metadata.get("partial_results", {}):
                shared_data[r] = context.metadata["partial_results"][r]
        
        # Crear agente
        agent = agent_class(
            task=step.get("task", context.metadata.get("task", "")),
            openai_api_key=self.api_key,
            metadata={
                **context.metadata,
                "engine_mode": self.engine_mode
            },
            shared_data=shared_data
        )
        
        self.active_agents[agent.id] = agent
        return agent

    def _group_by_dependencies(
        self,
        workflow: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Agrupa pasos por dependencias."""
        groups = []
        remaining = workflow[:]
        completed = set()
        
        while remaining:
            # Encontrar pasos que pueden ejecutarse
            current_layer = [
                w for w in remaining
                if all(dep in completed for dep in w.get("requires", []))
            ]
            
            if not current_layer:
                current_layer = [remaining[0]]
                
            groups.append(current_layer)
            completed.update(x["agent"] for x in current_layer)
            remaining = [x for x in remaining if x not in current_layer]
        
        return groups

    def _get_cache_key(self, step: Dict[str, Any]) -> str:
        """Genera clave de caché para un paso."""
        return f"{step['agent']}_{hash(str(step))}"

    def _update_stats(self, success: bool, execution_time: float) -> None:
        """Actualiza estadísticas globales."""
        self.execution_stats["total_executions"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
            
        # Actualizar tiempo promedio
        prev_avg = self.execution_stats["average_duration"]
        prev_total = self.execution_stats["total_executions"] - 1
        new_avg = (prev_avg * prev_total + execution_time) / self.execution_stats["total_executions"]
        self.execution_stats["average_duration"] = new_avg

    def _update_agent_stats(
        self,
        agent_type: str,
        success: bool,
        execution_time: float
    ) -> None:
        """Actualiza estadísticas por agente."""
        if agent_type not in self.execution_stats["agent_performance"]:
            self.execution_stats["agent_performance"][agent_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0
            }
            
        stats = self.execution_stats["agent_performance"][agent_type]
        stats["total_executions"] += 1
        
        if success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
            
        if execution_time > 0:
            prev_avg = stats["average_execution_time"]
            prev_total = stats["total_executions"] - 1
            new_avg = (prev_avg * prev_total + execution_time) / stats["total_executions"]
            stats["average_execution_time"] = new_avg

    async def cleanup(self) -> None:
        """Limpia recursos."""
        try:
            await self.comm_system.cleanup()
            
            for agent in self.active_agents.values():
                await agent.cleanup()
                
            self.active_agents.clear()
            await self.cache.cleanup()
            
            logger.info("Agent manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del gestor."""
        return {
            "active_agents": len(self.active_agents),
            "registered_types": list(self.agent_registry.keys()),
            "execution_stats": self.execution_stats,
            "cache_status": await self.cache.get_status()
        }