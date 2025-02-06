# monitoring/monitoring_manager.py

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path
import aiohttp
from prometheus_client import (
    Counter, Gauge, Histogram,
    CollectorRegistry, push_to_gateway
)

logger = logging.getLogger(__name__)

class MonitoringManager:
    """
    Gestor de monitoreo y métricas en tiempo real.
    
    Características:
    - Métricas Prometheus/Grafana
    - Alertas en tiempo real
    - Análisis de tendencias
    - Reportes automáticos
    """
    
    def __init__(self):
        # Registro Prometheus
        self.registry = CollectorRegistry()
        
        # Métricas
        self._setup_metrics()
        
        # Estado y configuración
        self.alert_config = self._load_alert_config()
        self.active_alerts: Set[str] = set()
        
        # Cache de métricas
        self.metrics_cache: Dict[str, List[float]] = {}
        
        # Iniciar monitores
        self._start_monitoring()

    def _setup_metrics(self) -> None:
        """Configura métricas Prometheus."""
        self.metrics = {
            # Sistema
            "system_load": Gauge(
                "vyper_system_load",
                "System load average",
                registry=self.registry
            ),
            "memory_usage": Gauge(
                "vyper_memory_usage_bytes",
                "Memory usage in bytes",
                registry=self.registry
            ),
            
            # API
            "request_latency": Histogram(
                "vyper_request_latency_seconds",
                "Request latency in seconds",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self.registry
            ),
            "request_count": Counter(
                "vyper_request_total",
                "Total requests processed",
                registry=self.registry
            ),
            
            # Modelos
            "model_latency": Histogram(
                "vyper_model_latency_seconds",
                "Model inference latency",
                labelnames=["model_name"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self.registry
            ),
            "model_errors": Counter(
                "vyper_model_errors_total",
                "Total model errors",
                labelnames=["model_name"],
                registry=self.registry
            ),
            
            # Edge
            "edge_node_count": Gauge(
                "vyper_edge_nodes",
                "Number of active edge nodes",
                registry=self.registry
            ),
            "edge_task_latency": Histogram(
                "vyper_edge_task_latency_seconds",
                "Edge task processing latency",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self.registry
            )
        }

    def _load_alert_config(self) -> Dict[str, Any]:
        """Carga configuración de alertas."""
        try:
            config_path = Path(__file__).parent / "alert_config.yaml"
            if not config_path.exists():
                return {}
                
            with open(config_path) as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Error loading alert config: {e}")
            return {}

    def _start_monitoring(self) -> None:
        """Inicia tareas de monitoreo."""
        asyncio.create_task(self._monitor_system())
        asyncio.create_task(self._monitor_models())
        asyncio.create_task(self._monitor_edge())
        asyncio.create_task(self._push_metrics())

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Registra una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor a registrar
            labels: Labels opcionales
        """
        try:
            metric = self.metrics.get(metric_name)
            if not metric:
                logger.warning(f"Unknown metric: {metric_name}")
                return
                
            # Registrar valor
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
                
            # Actualizar caché
            if metric_name not in self.metrics_cache:
                self.metrics_cache[metric_name] = []
            self.metrics_cache[metric_name].append(value)
            
            # Verificar alertas
            await self._check_alerts(metric_name, value, labels)
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")

    async def _check_alerts(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Verifica si se deben generar alertas."""
        try:
            if metric_name not in self.alert_config:
                return
                
            alert_rules = self.alert_config[metric_name]
            for rule in alert_rules:
                # Verificar condición
                if self._evaluate_alert_condition(rule["condition"], value):
                    alert_id = f"{metric_name}_{rule['name']}"
                    
                    # Evitar duplicados
                    if alert_id not in self.active_alerts:
                        self.active_alerts.add(alert_id)
                        await self._send_alert(
                            rule["name"],
                            rule["message"],
                            {
                                "metric": metric_name,
                                "value": value,
                                "labels": labels,
                                "threshold": rule["condition"]["threshold"]
                            }
                        )
                else:
                    # Limpiar alerta si existe
                    alert_id = f"{metric_name}_{rule['name']}"
                    self.active_alerts.discard(alert_id)
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def _evaluate_alert_condition(
        self,
        condition: Dict[str, Any],
        value: float
    ) -> bool:
        """Evalúa una condición de alerta."""
        try:
            operator = condition["operator"]
            threshold = condition["threshold"]
            
            if operator == ">":
                return value > threshold
            elif operator == "<":
                return value < threshold
            elif operator == ">=":
                return value >= threshold
            elif operator == "<=":
                return value <= threshold
            elif operator == "==":
                return value == threshold
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating alert condition: {e}")
            return False

    async def _send_alert(
        self,
        name: str,
        message: str,
        data: Dict[str, Any]
    ) -> None:
        """Envía una alerta a los canales configurados."""
        try:
            alert = {
                "name": name,
                "message": message,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Enviar a Slack si está configurado
            if "slack_webhook" in self.alert_config:
                await self._send_slack_alert(alert)
            
            # Enviar por email si está configurado
            if "email_config" in self.alert_config:
                await self._send_email_alert(alert)
            
            # Registrar alerta
            logger.warning(f"Alert triggered: {name} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    async def _send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """Envía alerta a Slack."""
        try:
            webhook_url = self.alert_config["slack_webhook"]
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    webhook_url,
                    json={
                        "text": f"🚨 *Alert: {alert['name']}*\n"
                               f"Message: {alert['message']}\n"
                               f"Data: ```{json.dumps(alert['data'], indent=2)}```"
                    }
                )
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    async def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Envía alerta por email."""
        try:
            email_config = self.alert_config["email_config"]
            
            # Implementar envío de email
            logger.info(f"Email alert would be sent to {email_config['recipients']}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

    async def _monitor_system(self) -> None:
        """Monitorea recursos del sistema."""
        while True:
            try:
                # Obtener métricas del sistema
                system_metrics = await self._get_system_metrics()
                
                # Actualizar métricas
                self.metrics["system_load"].set(system_metrics["load"])
                self.metrics["memory_usage"].set(system_metrics["memory"])
                
                await asyncio.sleep(60)  # Actualizar cada minuto
                
            except Exception as e:
                logger.error(f"Error monitoring system: {e}")
                await asyncio.sleep(60)

    async def _monitor_models(self) -> None:
        """Monitorea rendimiento de modelos."""
        while True:
            try:
                # Obtener métricas de modelos
                model_metrics = await self._get_model_metrics()
                
                # Actualizar métricas por modelo
                for model_name, metrics in model_metrics.items():
                    self.metrics["model_latency"].labels(
                        model_name=model_name
                    ).observe(metrics["latency"])
                    
                    if metrics.get("errors", 0) > 0:
                        self.metrics["model_errors"].labels(
                            model_name=model_name
                        ).inc(metrics["errors"])
                
                await asyncio.sleep(30)  # Actualizar cada 30 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring models: {e}")
                await asyncio.sleep(30)

    async def _monitor_edge(self) -> None:
        """Monitorea nodos edge."""
        while True:
            try:
                # Obtener métricas edge
                edge_metrics = await self._get_edge_metrics()
                
                # Actualizar métricas
                self.metrics["edge_node_count"].set(edge_metrics["node_count"])
                
                for latency in edge_metrics.get("latencies", []):
                    self.metrics["edge_task_latency"].observe(latency)
                
                await asyncio.sleep(15)  # Actualizar cada 15 segundos
                
            except Exception as e:
                logger.error(f"Error monitoring edge: {e}")
                await asyncio.sleep(15)

    async def _push_metrics(self) -> None:
        """Envía métricas a Prometheus Gateway."""
        while True:
            try:
                push_to_gateway(
                    self.alert_config.get("prometheus_gateway", "localhost:9091"),
                    job="vyper_ai",
                    registry=self.registry
                )
                
                await asyncio.sleep(10)  # Enviar cada 10 segundos
                
            except Exception as e:
                logger.error(f"Error pushing metrics: {e}")
                await asyncio.sleep(10)

    async def _get_system_metrics(self) -> Dict[str, float]:
        """Obtiene métricas del sistema."""
        try:
            import psutil
            
            cpu_load = psutil.cpu_percent() / 100.0
            memory = psutil.virtual_memory()
            
            return {
                "load": cpu_load,
                "memory": memory.used
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"load": 0.0, "memory": 0.0}

    async def _get_model_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene métricas de los modelos."""
        # Implementar obtención real de métricas
        return {
            "gpt4": {
                "latency": 0.5,
                "errors": 0
            },
            "deepseek": {
                "latency": 0.3,
                "errors": 0
            }
        }

    async def _get_edge_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de nodos edge."""
        # Implementar obtención real de métricas
        return {
            "node_count": 5,
            "latencies": [0.1, 0.2, 0.15]
        }

    async def get_metrics_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Genera un reporte de métricas para un período.
        
        Args:
            start_time: Inicio del período
            end_time: Fin del período
        """
        try:
            report = {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "system": await self._get_system_report(start_time, end_time),
                "models": await self._get_models_report(start_time, end_time),
                "edge": await self._get_edge_report(start_time, end_time),
                "alerts": await self._get_alerts_report(start_time, end_time)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating metrics report: {e}")
            return {}

    async def _get_system_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Genera reporte de métricas del sistema."""
        return {
            "average_load": self._calculate_average("system_load"),
            "peak_memory": self._calculate_peak("memory_usage"),
            "uptime": self._calculate_uptime()
        }

    async def _get_models_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Genera reporte de métricas de modelos."""
        return {
            "latency": {
                "average": self._calculate_average("model_latency"),
                "p95": self._calculate_percentile("model_latency", 95),
                "p99": self._calculate_percentile("model_latency", 99)
            },
            "errors": {
                "total": self._calculate_sum("model_errors"),
                "by_model": self._calculate_errors_by_model()
            }
        }

    async def _get_edge_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Genera reporte de métricas edge."""
        return {
            "nodes": {
                "average": self._calculate_average("edge_node_count"),
                "min": self._calculate_min("edge_node_count"),
                "max": self._calculate_max("edge_node_count")
            },
            "task_latency": {
                "average": self._calculate_average("edge_task_latency"),
                "p95": self._calculate_percentile("edge_task_latency", 95)
            }
        }

    async def _get_alerts_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Genera reporte de alertas."""
        return {
            "total": len(self.active_alerts),
            "by_type": self._count_alerts_by_type()
        }

    def _calculate_average(self, metric_name: str) -> float:
        """Calcula promedio de una métrica."""
        try:
            values = self.metrics_cache.get(metric_name, [])
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error calculating average for {metric_name}: {e}")
            return 0.0

    def _calculate_percentile(
        self,
        metric_name: str,
        percentile: int
    ) -> float:
        """Calcula percentil de una métrica."""
        try:
            values = self.metrics_cache.get(metric_name, [])
            if not values:
                return 0.0
            return np.percentile(values, percentile)
        except Exception as e:
            logger.error(f"Error calculating percentile for {metric_name}: {e}")
            return 0.0

    def _calculate_peak(self, metric_name: str) -> float:
        """Calcula valor pico de una métrica."""
        try:
            values = self.metrics_cache.get(metric_name, [])
            if not values:
                return 0.0
            return max(values)
        except Exception as e:
            logger.error(f"Error calculating peak for {metric_name}: {e}")
            return 0.0

    def _calculate_uptime(self) -> float:
        """Calcula tiempo de actividad del sistema."""
        try:
            import psutil
            return psutil.boot_time()
        except Exception as e:
            logger.error(f"Error calculating uptime: {e}")
            return 0.0

    def _calculate_errors_by_model(self) -> Dict[str, int]:
        """Calcula errores por modelo."""
        return {
            "gpt4": self.metrics["model_errors"].labels(model_name="gpt4")._value.get(),
            "deepseek": self.metrics["model_errors"].labels(model_name="deepseek")._value.get()
        }

    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Cuenta alertas por tipo."""
        counts = {}
        for alert_id in self.active_alerts:
            alert_type = alert_id.split("_")[0]
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts

    async def cleanup(self) -> None:
        """Limpia recursos del sistema de monitoreo."""
        try:
            # Limpiar métricas
            self.metrics_cache.clear()
            self.active_alerts.clear()
            
            # Desregistrar métricas
            for metric in self.metrics.values():
                self.registry.unregister(metric)
            
            logger.info("Monitoring Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del sistema de monitoreo."""
        return {
            "active_metrics": list(self.metrics.keys()),
            "active_alerts": list(self.active_alerts),
            "metrics_cache_size": {
                metric: len(values)
                for metric, values in self.metrics_cache.items()
            },
            "prometheus_status": "connected" if self.registry else "disconnected"
        }