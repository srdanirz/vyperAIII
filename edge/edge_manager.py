# edge/edge_manager.py

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import aiohttp
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class EdgeNode:
    """Representa un nodo de procesamiento edge."""
    
    def __init__(
        self,
        node_id: str,
        capabilities: List[str],
        resources: Dict[str, Any]
    ):
        self.node_id = node_id
        self.capabilities = set(capabilities)
        self.resources = resources
        self.status = "ready"
        self.load = 0.0
        self.last_heartbeat = datetime.now()
        self.metrics: Dict[str, Any] = {
            "processed_tasks": 0,
            "errors": 0,
            "average_latency": 0.0
        }

class EdgeManager:
    """
    Gestiona el procesamiento distribuido en nodos edge.
    
    Características:
    - Descubrimiento automático de nodos
    - Balanceo de carga
    - Tolerancia a fallos
    - Optimización de latencia
    """
    
    def __init__(self):
        self.nodes: Dict[str, EdgeNode] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.node_groups: Dict[str, Set[str]] = {}
        
        # Métricas y estado
        self.metrics = {
            "total_nodes": 0,
            "active_tasks": 0,
            "total_processed": 0,
            "average_latency": 0.0
        }
        
        # Configuración
        self._load_config()
        
        # Iniciar monitoreo
        asyncio.create_task(self._monitor_nodes())

    def _load_config(self) -> None:
        """Carga configuración del sistema edge."""
        try:
            config_path = Path(__file__).parent / "edge_config.yaml"
            if not config_path.exists():
                return
            
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Error loading edge config: {e}")
            self.config = {}

    async def register_node(
        self,
        node_id: str,
        capabilities: List[str],
        resources: Dict[str, Any]
    ) -> bool:
        """
        Registra un nuevo nodo edge.
        
        Args:
            node_id: Identificador único del nodo
            capabilities: Lista de capacidades del nodo
            resources: Recursos disponibles
        """
        try:
            # Validar nodo
            if not await self._validate_node(node_id, capabilities, resources):
                return False
            
            # Crear nodo
            node = EdgeNode(node_id, capabilities, resources)
            self.nodes[node_id] = node
            
            # Actualizar grupos
            for capability in capabilities:
                if capability not in self.node_groups:
                    self.node_groups[capability] = set()
                self.node_groups[capability].add(node_id)
            
            # Actualizar métricas
            self.metrics["total_nodes"] += 1
            
            logger.info(f"Registered new edge node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering node {node_id}: {e}")
            return False

    async def submit_task(
        self,
        task: Dict[str, Any],
        required_capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        Envía una tarea para procesamiento edge.
        
        Args:
            task: Tarea a procesar
            required_capabilities: Capacidades requeridas
        """
        try:
            # Seleccionar nodo óptimo
            node = await self._select_optimal_node(required_capabilities)
            if not node:
                raise ValueError("No suitable node found")
            
            # Preparar tarea
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            prepared_task = {
                "id": task_id,
                "content": task,
                "required_capabilities": required_capabilities,
                "submitted_at": datetime.now().isoformat()
            }
            
            # Registrar tarea
            self.active_tasks[task_id] = {
                "task": prepared_task,
                "node_id": node.node_id,
                "status": "submitted"
            }
            
            # Enviar tarea al nodo
            result = await self._send_task_to_node(node, prepared_task)
            
            # Actualizar métricas
            self._update_metrics(node, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            raise

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de una tarea.
        
        Args:
            task_id: ID de la tarea
        """
        try:
            if task_id not in self.active_tasks:
                return {"error": "Task not found"}
            
            task_info = self.active_tasks[task_id]
            node = self.nodes.get(task_info["node_id"])
            
            if not node:
                return {"error": "Processing node not found"}
            
            # Obtener estado actualizado del nodo
            status = await self._get_node_task_status(node, task_id)
            
            return {
                "task_id": task_id,
                "status": status["status"],
                "node_id": node.node_id,
                "submitted_at": task_info["task"]["submitted_at"],
                "result": status.get("result")
            }
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"error": str(e)}

    async def _validate_node(
        self,
        node_id: str,
        capabilities: List[str],
        resources: Dict[str, Any]
    ) -> bool:
        """Valida un nodo antes de registrarlo."""
        try:
            # Verificar requisitos mínimos
            required_capabilities = self.config.get("required_capabilities", [])
            if not all(cap in capabilities for cap in required_capabilities):
                return False
            
            # Verificar recursos mínimos
            min_resources = self.config.get("minimum_resources", {})
            for resource, min_value in min_resources.items():
                if resource not in resources or resources[resource] < min_value:
                    return False
            
            # Verificar conectividad
            return await self._check_node_connectivity(node_id)
            
        except Exception as e:
            logger.error(f"Error validating node: {e}")
            return False

    async def _select_optimal_node(
        self,
        required_capabilities: List[str]
    ) -> Optional[EdgeNode]:
        """Selecciona el nodo óptimo para una tarea."""
        try:
            eligible_nodes = []
            
            # Filtrar nodos por capacidades
            for node in self.nodes.values():
                if node.status == "ready" and all(
                    cap in node.capabilities
                    for cap in required_capabilities
                ):
                    eligible_nodes.append(node)
            
            if not eligible_nodes:
                return None
            
            # Calcular score para cada nodo
            node_scores = []
            for node in eligible_nodes:
                score = self._calculate_node_score(node)
                node_scores.append((score, node))
            
            # Seleccionar mejor nodo
            return max(node_scores, key=lambda x: x[0])[1]
            
        except Exception as e:
            logger.error(f"Error selecting node: {e}")
            return None

    def _calculate_node_score(self, node: EdgeNode) -> float:
        """Calcula un score de optimización para un nodo."""
        try:
            # Factores de score
            load_factor = 1 - node.load
            error_factor = 1 / (node.metrics["errors"] + 1)
            latency_factor = 1 / (node.metrics["average_latency"] + 1)
            
            # Pesos
            weights = {
                "load": 0.4,
                "errors": 0.3,
                "latency": 0.3
            }
            
            # Calcular score final
            score = (
                weights["load"] * load_factor +
                weights["errors"] * error_factor +
                weights["latency"] * latency_factor
            )
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error calculating node score: {e}")
            return 0.0

    async def _send_task_to_node(
        self,
        node: EdgeNode,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Envía una tarea a un nodo específico."""
        try:
            # Simular envío (reemplazar con implementación real)
            await asyncio.sleep(0.1)
            
            node.load += 0.1
            node.metrics["processed_tasks"] += 1
            
            return {
                "status": "success",
                "task_id": task["id"],
                "node_id": node.node_id
            }
            
        except Exception as e:
            logger.error(f"Error sending task to node: {e}")
            raise

    async def _monitor_nodes(self) -> None:
        """Monitorea el estado de los nodos."""
        while True:
            try:
                for node_id, node in list(self.nodes.items()):
                    # Verificar último heartbeat
                    if (datetime.now() - node.last_heartbeat).seconds > 60:
                        await self._handle_node_failure(node_id)
                    
                    # Actualizar métricas
                    await self._update_node_metrics(node)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error monitoring nodes: {e}")
                await asyncio.sleep(10)

    async def _handle_node_failure(self, node_id: str) -> None:
        """Maneja el fallo de un nodo."""
        try:
            logger.warning(f"Node failure detected: {node_id}")
            
            # Recuperar tareas activas
            affected_tasks = [
                task_id for task_id, task in self.active_tasks.items()
                if task["node_id"] == node_id
            ]
            
            # Reasignar tareas
            for task_id in affected_tasks:
                await self._reassign_task(task_id)
            
            # Eliminar nodo
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Limpiar grupos
                for capability in node.capabilities:
                    if capability in self.node_groups:
                        self.node_groups[capability].discard(node_id)
                del self.nodes[node_id]
            
            self.metrics["total_nodes"] -= 1
            
        except Exception as e:
            logger.error(f"Error handling node failure: {e}")

    async def _reassign_task(self, task_id: str) -> None:
        """
        Reasigna una tarea a un nuevo nodo cuando el nodo original falla.
        
        Args:
            task_id: ID de la tarea a reasignar
        """
        try:
            if task_id not in self.active_tasks:
                return
                
            task_info = self.active_tasks[task_id]
            original_node_id = task_info["node_id"]
            
            # Obtener capacidades requeridas
            required_capabilities = task_info["task"]["required_capabilities"]
            
            # Seleccionar nuevo nodo
            new_node = await self._select_optimal_node(required_capabilities)
            if not new_node:
                logger.error(f"No alternative node found for task {task_id}")
                task_info["status"] = "failed"
                return
                
            # Actualizar registro de tarea
            task_info["node_id"] = new_node.node_id
            task_info["reassigned_at"] = datetime.now().isoformat()
            task_info["previous_node"] = original_node_id
            
            # Reenviar tarea
            logger.info(f"Reassigning task {task_id} from {original_node_id} to {new_node.node_id}")
            await self._send_task_to_node(new_node, task_info["task"])
            
        except Exception as e:
            logger.error(f"Error reassigning task {task_id}: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"

    async def _update_node_metrics(self, node: EdgeNode) -> None:
        """
        Actualiza métricas de un nodo.
        
        Args:
            node: Nodo a actualizar
        """
        try:
            # Simular obtención de métricas (reemplazar con implementación real)
            node.load = max(0.0, min(1.0, node.load - 0.1))  # Decay simulado
            
            # Actualizar promedio de latencia
            if node.metrics["processed_tasks"] > 0:
                node.metrics["average_latency"] = (
                    node.metrics["average_latency"] * 0.9 +  # Peso histórico
                    random.uniform(0.1, 0.5) * 0.1  # Nueva muestra simulada
                )
            
            # Verificar salud del nodo
            if node.load > 0.9 or node.metrics["errors"] > 100:
                await self._handle_unhealthy_node(node)
                
        except Exception as e:
            logger.error(f"Error updating metrics for node {node.node_id}: {e}")

    async def _handle_unhealthy_node(self, node: EdgeNode) -> None:
        """
        Maneja un nodo en estado poco saludable.
        
        Args:
            node: Nodo a manejar
        """
        try:
            logger.warning(f"Unhealthy node detected: {node.node_id}")
            
            # Cambiar estado
            node.status = "degraded"
            
            # Reducir carga
            current_tasks = [
                task_id for task_id, task in self.active_tasks.items()
                if task["node_id"] == node.node_id
            ]
            
            # Reasignar tareas no críticas
            for task_id in current_tasks:
                task_info = self.active_tasks[task_id]
                if not task_info.get("critical", False):
                    await self._reassign_task(task_id)
            
            # Programar health check
            asyncio.create_task(self._schedule_health_check(node))
            
        except Exception as e:
            logger.error(f"Error handling unhealthy node {node.node_id}: {e}")

    async def _schedule_health_check(self, node: EdgeNode) -> None:
        """
        Programa una verificación de salud para un nodo.
        
        Args:
            node: Nodo a verificar
        """
        try:
            await asyncio.sleep(300)  # Esperar 5 minutos
            
            # Verificar métricas actuales
            if node.load < 0.7 and node.metrics["errors"] < 50:
                node.status = "ready"
                logger.info(f"Node {node.node_id} recovered")
            else:
                # Mantener en estado degradado
                asyncio.create_task(self._schedule_health_check(node))
                
        except Exception as e:
            logger.error(f"Error in health check for node {node.node_id}: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas globales del sistema edge."""
        try:
            return {
                "nodes": {
                    "total": self.metrics["total_nodes"],
                    "ready": len([n for n in self.nodes.values() if n.status == "ready"]),
                    "degraded": len([n for n in self.nodes.values() if n.status == "degraded"])
                },
                "tasks": {
                    "active": len(self.active_tasks),
                    "total_processed": self.metrics["total_processed"],
                    "average_latency": self.metrics["average_latency"]
                },
                "capabilities": {
                    cap: len(nodes) for cap, nodes in self.node_groups.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    async def cleanup(self) -> None:
        """Limpia recursos y finaliza procesamiento."""
        try:
            # Cancelar tareas activas
            for task_id, task_info in self.active_tasks.items():
                node = self.nodes.get(task_info["node_id"])
                if node:
                    await self._cancel_task(node, task_id)
            
            # Limpiar estructuras de datos
            self.nodes.clear()
            self.active_tasks.clear()
            self.node_groups.clear()
            
            logger.info("Edge Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _cancel_task(self, node: EdgeNode, task_id: str) -> None:
        """
        Cancela una tarea en un nodo.
        
        Args:
            node: Nodo que ejecuta la tarea
            task_id: ID de la tarea a cancelar
        """
        try:
            # Implementar cancelación real aquí
            await asyncio.sleep(0.1)  # Simular delay de red
            
            # Actualizar estado
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "cancelled"
            
            # Actualizar métricas del nodo
            node.load = max(0.0, node.load - 0.1)
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id} on node {node.node_id}: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema edge."""
        return {
            "nodes": {
                node_id: {
                    "status": node.status,
                    "load": node.load,
                    "capabilities": list(node.capabilities),
                    "metrics": node.metrics
                }
                for node_id, node in self.nodes.items()
            },
            "active_tasks": len(self.active_tasks),
            "node_groups": {
                cap: list(nodes)
                for cap, nodes in self.node_groups.items()
            },
            "metrics": await self.get_metrics()
        }