import logging
import asyncio
import os
import psutil
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp
from prometheus_client import (
    Counter, Gauge, Histogram,
    CollectorRegistry, push_to_gateway
)

from core.interfaces import ResourceUsage, PerformanceMetrics
from core.errors import ProcessingError, handle_errors, ErrorBoundary

logger = logging.getLogger(__name__)

class MonitoringManager:
    """
    Sistema de monitoreo en tiempo real con métricas.
    
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
        
        # Cache de métricas
        self.metrics_cache: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        
        # Resource monitoring
        self.resource_usage = ResourceUsage()
        
        # Estado y alertas
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Tareas de monitoreo
        self._monitoring_tasks: List[asyncio.Task] = []

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
            
            # LLM
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
        
        # Iniciar monitoreo periódico
        self._start_monitoring()

    @handle_errors()
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
        metric = self.metrics.get(metric_name)
        if not metric:
            logger.warning(f"Unknown metric: {metric_name}")
            return

        try:
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
            thresholds = {
                "error_rate": 0.1,
                "response_time": 5.0,
                "cpu_usage": 80,
                "memory_usage": 80
            }

            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                if value > threshold:
                    alert_id = f"{metric_name}_{datetime.now().isoformat()}"
                    await self._send_alert(
                        severity="warning",
                        title=f"Metric {metric_name} exceeded threshold",
                        description=f"Value: {value}, Threshold: {threshold}",
                        metric_data={
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold,
                            "labels": labels
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    async def _send_alert(
        self,
        severity: str,
        title: str,
        description: str,
        metric_data: Dict[str, Any]
    ) -> None:
        """Envía una alerta."""
        try:
            alert = {
                "id": f"alert_{datetime.now().isoformat()}",
                "severity": severity,
                "title": title,
                "description": description,
                "metric_data": metric_data,
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }

            # Registrar alerta
            self.active_alerts[alert["id"]] = alert
            self.alert_history.append(alert)

            # Enviar a Slack si está configurado
            if slack_webhook := os.getenv("SLACK_WEBHOOK_URL"):
                await self._send_slack_alert(alert, slack_webhook)

            # Enviar por email si está configurado
            if email_config := os.getenv("ALERT_EMAIL"):
                await self._send_email_alert(alert, email_config)

            logger.warning(f"Alert triggered: {title}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    async def _send_slack_alert(
        self,
        alert: Dict[str, Any],
        webhook_url: str
    ) -> None:
        """Envía alerta a Slack."""
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    webhook_url,
                    json={
                        "text": f"🚨 *{alert['title']}*\n"
                               f"{alert['description']}\n"
                               f"Severity: {alert['severity']}\n"
                               f"Metric Data: ```{json.dumps(alert['metric_data'], indent=2)}```"
                    }
                )
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def _start_monitoring(self) -> None:
        """Inicia tareas de monitoreo periódico."""
        self._monitoring_tasks = [
            asyncio.create_task(self._monitor_system_metrics()),
            asyncio.create_task(self._monitor_active_alerts())
        ]

    async def _monitor_system_metrics(self) -> None:
        """Monitorea métricas del sistema periódicamente."""
        while True:
            try:
                # Actualizar métricas del sistema
                self.metrics["system_load"].set(os.getloadavg()[0])
                self.metrics["memory_usage"].set(psutil.Process().memory_info().rss)
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(60)

    async def _monitor_active_alerts(self) -> None:
        """Monitorea y actualiza estado de alertas activas."""
        while True:
            try:
                now = datetime.now()
                
                # Revisar alertas activas
                for alert_id, alert in list(self.active_alerts.items()):
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    
                    # Resolver alertas después de 1 hora
                    if (now - alert_time).total_seconds() > 3600:
                        alert["status"] = "resolved"
                        del self.active_alerts[alert_id]
                
                await asyncio.sleep(300)  # Verificar cada 5 minutos
                
            except Exception as e:
                logger.error(f"Error monitoring alerts: {e}")
                await asyncio.sleep(300)

    async def get_metrics_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Genera un reporte de métricas para un período."""
        try:
            report = {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "system": {
                    "average_load": self._calculate_average("system_load"),
                    "peak_memory": self._calculate_peak("memory_usage"),
                    "resource_usage": self.resource_usage.dict()
                },
                "performance": {
                    "request_latency": {
                        "average": self._calculate_average("request_latency"),
                        "p95": self._calculate_percentile("request_latency", 95),
                        "p99": self._calculate_percentile("request_latency", 99)
                    },
                    "model_performance": self._get_model_performance(),
                    "metrics": self.performance_metrics.dict()
                },
                "alerts": {
                    "active": len(self.active_alerts),
                    "history": len(self.alert_history),
                    "by_severity": self._count_alerts_by_severity()
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating metrics report: {e}")
            return {}

    def _calculate_average(self, metric_name: str) -> float:
        """Calcula promedio de una métrica."""
        try:
            values = self.metrics_cache.get(metric_name, [])
            return sum(values) / len(values) if values else 0.0
        except Exception as e:
            logger.error(f"Error calculating average: {e}")
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
            return float(np.percentile(values, percentile))
        except Exception as e:
            logger.error(f"Error calculating percentile: {e}")
            return 0.0

    def _calculate_peak(self, metric_name: str) -> float:
        """Calcula valor pico de una métrica."""
        try:
            values = self.metrics_cache.get(metric_name, [])
            return max(values) if values else 0.0
        except Exception as e:
            logger.error(f"Error calculating peak: {e}")
            return 0.0

    def _get_model_performance(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento de modelos."""
        models = ["gpt-4", "deepseek", "claude"]
        return {
            model: {
                "latency": self._calculate_average(f"model_latency_{model}"),
                "errors": self.metrics["model_errors"].labels(model_name=model)._value.get()
            }
            for model in models
        }

    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Cuenta alertas por severidad."""
        counts = {}
        for alert in self.alert_history:
            severity = alert["severity"]
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    async def cleanup(self) -> None:
        """Limpia recursos del sistema de monitoreo."""
        try:
            # Cancelar tareas de monitoreo
            for task in self._monitoring_tasks:
                task.cancel()
                
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
            # Limpiar datos
            self.metrics_cache.clear()
            self.active_alerts.clear()
            
            # Reset metrics
            self.performance_metrics = PerformanceMetrics()
            self.resource_usage = ResourceUsage()
            
            logger.info("Monitoring Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema de monitoreo."""
        return {
            "active_metrics": list(self.metrics.keys()),
            "active_alerts": len(self.active_alerts),
            "metrics_cache_size": {
                metric: len(values)
                for metric, values in self.metrics_cache.items()
            },
            "performance": self.performance_metrics.dict(),
            "resources": self.resource_usage.dict()
        }