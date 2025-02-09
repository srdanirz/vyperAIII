import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import uuid
import aiohttp
from pydantic import BaseModel, Field

from core.interfaces import ResourceUsage, PerformanceMetrics
from core.errors import ProcessingError, handle_errors

logger = logging.getLogger(__name__)

class EdgeNode(BaseModel):
    """Node representation."""
    node_id: str
    capabilities: Set[str]
    resources: ResourceUsage
    status: str = "ready"
    load: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)

class EdgeManager:
    """
    Edge computing node management.
    
    Features:
    - Auto node discovery
    - Load balancing
    - Fault tolerance
    - Latency optimization
    """
    
    def __init__(self):
        # Active nodes
        self.nodes: Dict[str, EdgeNode] = {}
        
        # Node groups by capability
        self.node_groups: Dict[str, Set[str]] = {}
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # System metrics
        self.performance_metrics = PerformanceMetrics()
        
        # Start monitoring
        self._start_monitoring()

    @handle_errors()
    async def register_node(
        self,
        node_id: str,
        capabilities: List[str],
        resources: Dict[str, Any]
    ) -> bool:
        """Register a new edge node."""
        try:
            # Validate node
            if not await self._validate_node(node_id, capabilities, resources):
                return False
            
            # Create node
            node = EdgeNode(
                node_id=node_id,
                capabilities=set(capabilities),
                resources=ResourceUsage(**resources)
            )
            self.nodes[node_id] = node
            
            # Update groups
            for capability in capabilities:
                if capability not in self.node_groups:
                    self.node_groups[capability] = set()
                self.node_groups[capability].add(node_id)
            
            # Update metrics
            self.performance_metrics.total_requests += 1
            
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
        """Submit task for edge processing."""
        try:
            # Select optimal node
            node = await self._select_optimal_node(required_capabilities)
            if not node:
                raise ValueError("No suitable node found")
            
            # Prepare task
            task_id = str(uuid.uuid4())
            prepared_task = {
                "id": task_id,
                "content": task,
                "required_capabilities": required_capabilities,
                "submitted_at": datetime.now().isoformat()
            }
            
            # Register task
            self.active_tasks[task_id] = {
                "task": prepared_task,
                "node_id": node.node_id,
                "status": "submitted"
            }
            
            # Send task to node
            result = await self._send_task_to_node(node, prepared_task)
            
            # Update metrics
            self._update_metrics(node, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            raise

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task."""
        try:
            if task_id not in self.active_tasks:
                return {"error": "Task not found"}
            
            task_info = self.active_tasks[task_id]
            node = self.nodes.get(task_info["node_id"])
            
            if not node:
                return {"error": "Processing node not found"}
            
            # Get updated status from node
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
        """Validate a node before registration."""
        try:
            # Verify minimum resources
            min_resources = {
                "cpu_percent": 10,
                "memory_percent": 10,
                "disk_percent": 10
            }
            
            for resource, min_value in min_resources.items():
                if resource not in resources or resources[resource] < min_value:
                    return False
            
            # Verify connectivity
            return await self._check_node_connectivity(node_id)
            
        except Exception as e:
            logger.error(f"Error validating node: {e}")
            return False

    async def _select_optimal_node(
        self,
        required_capabilities: List[str]
    ) -> Optional[EdgeNode]:
        """Select the optimal node for a task."""
        try:
            eligible_nodes = []
            
            # Filter nodes by capabilities
            for node in self.nodes.values():
                if node.status == "ready" and all(
                    cap in node.capabilities 
                    for cap in required_capabilities
                ):
                    eligible_nodes.append(node)
            
            if not eligible_nodes:
                return None
            
            # Calculate score for each node
            node_scores = []
            for node in eligible_nodes:
                score = self._calculate_node_score(node)
                node_scores.append((score, node))
            
            # Select best node
            return max(node_scores, key=lambda x: x[0])[1]
            
        except Exception as e:
            logger.error(f"Error selecting node: {e}")
            return None

    def _calculate_node_score(self, node: EdgeNode) -> float:
        """Calculate optimization score for a node."""
        try:
            # Score factors
            load_factor = 1 - node.load
            error_factor = 1 / (node.metrics.failed_requests + 1)
            latency_factor = 1 / (node.metrics.average_response_time + 1)
            
            # Weights
            weights = {
                "load": 0.4,
                "errors": 0.3,
                "latency": 0.3
            }
            
            # Calculate final score
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
        """Send a task to a specific node."""
        try:
            # Simulate node communication
            await asyncio.sleep(0.1)
            
            node.load += 0.1
            node.metrics.total_requests += 1
            
            return {
                "status": "success",
                "task_id": task["id"],
                "node_id": node.node_id
            }
            
        except Exception as e:
            logger.error(f"Error sending task to node: {e}")
            raise

    async def _get_node_task_status(
        self,
        node: EdgeNode,
        task_id: str
    ) -> Dict[str, Any]:
        """Get task status from a node."""
        try:
            # Simulate status check
            await asyncio.sleep(0.1)
            
            return {
                "status": "completed",
                "task_id": task_id,
                "node_id": node.node_id
            }
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {"status": "error", "error": str(e)}

    def _update_metrics(
        self,
        node: EdgeNode,
        result: Dict[str, Any]
    ) -> None:
        """Update performance metrics."""
        try:
            self.performance_metrics.total_requests += 1
            
            if result.get("status") == "success":
                self.performance_metrics.successful_requests += 1
            else:
                self.performance_metrics.failed_requests += 1
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _start_monitoring(self) -> None:
        """Start monitoring tasks."""
        asyncio.create_task(self._monitor_nodes())
        asyncio.create_task(self._monitor_tasks())

    async def _monitor_nodes(self) -> None:
        """Monitor node health."""
        while True:
            try:
                now = datetime.now()
                
                # Check node heartbeats
                for node_id, node in list(self.nodes.items()):
                    if (now - node.last_heartbeat).total_seconds() > 60:
                        await self._handle_node_failure(node_id)
                        
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error monitoring nodes: {e}")
                await asyncio.sleep(10)

    async def _monitor_tasks(self) -> None:
        """Monitor task status."""
        while True:
            try:
                # Update task status
                for task_id, task_info in list(self.active_tasks.items()):
                    node = self.nodes.get(task_info["node_id"])
                    if node:
                        status = await self._get_node_task_status(node, task_id)
                        task_info["status"] = status["status"]
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring tasks: {e}")
                await asyncio.sleep(5)

    async def _handle_node_failure(self, node_id: str) -> None:
        """Handle node failure."""
        try:
            logger.warning(f"Node failure detected: {node_id}")
            
            # Get affected tasks
            affected_tasks = [
                task_id for task_id, task in self.active_tasks.items()
                if task["node_id"] == node_id
            ]
            
            # Reassign tasks
            for task_id in affected_tasks:
                await self._reassign_task(task_id)
            
            # Remove node
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Clean up node groups
                for capability in node.capabilities:
                    if capability in self.node_groups:
                        self.node_groups[capability].discard(node_id)
                del self.nodes[node_id]
            
            self.performance_metrics.total_nodes -= 1
            
        except Exception as e:
            logger.error(f"Error handling node failure: {e}")

    async def _reassign_task(self, task_id: str) -> None:
        """Reassign a task to a new node."""
        try:
            if task_id not in self.active_tasks:
                return
                
            task_info = self.active_tasks[task_id]
            task = task_info["task"]
            
            # Select new node
            new_node = await self._select_optimal_node(task["required_capabilities"])
            if not new_node:
                logger.error(f"No alternative node found for task {task_id}")
                task_info["status"] = "failed"
                return
                
            # Update task info
            task_info["node_id"] = new_node.node_id
            task_info["reassigned_at"] = datetime.now().isoformat()
            
            # Resend task
            logger.info(f"Reassigning task {task_id} to node {new_node.node_id}")
            await self._send_task_to_node(new_node, task)
            
        except Exception as e:
            logger.error(f"Error reassigning task {task_id}: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear data structures
            self.nodes.clear()
            self.node_groups.clear()
            self.active_tasks.clear()
            
            # Reset metrics
            self.performance_metrics = PerformanceMetrics()
            
            logger.info("Edge Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "nodes": {
                "total": len(self.nodes),
                "active": len([n for n in self.nodes.values() if n.status == "ready"])
            },
            "tasks": {
                "active": len(self.active_tasks),
                "completed": self.performance_metrics.successful_requests,
                "failed": self.performance_metrics.failed_requests
            },
            "metrics": self.performance_metrics.dict()
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the system."""
        try:
            total_nodes = len(self.nodes)
            active_nodes = len([n for n in self.nodes.values() if n.status == "ready"])
            
            return {
                "status": "healthy" if active_nodes/total_nodes >= 0.7 else "degraded",
                "nodes": {
                    "total": total_nodes,
                    "active": active_nodes,
                    "failed": total_nodes - active_nodes
                },
                "tasks": {
                    "active": len(self.active_tasks),
                    "completed": self.performance_metrics.successful_requests,
                    "failed": self.performance_metrics.failed_requests
                }
            }
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {"status": "error", "message": str(e)}