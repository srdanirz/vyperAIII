import logging
import asyncio
import os
import psutil
import numpy as np
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
import aiohttp
from pathlib import Path
import weakref
from prometheus_client import (
    Counter, 
    Gauge, 
    Histogram,
    CollectorRegistry, 
    push_to_gateway,
    REGISTRY,
    ProcessCollector,
    PlatformCollector,
    GCCollector
)
import threading
from functools import partial

from core.interfaces import ResourceUsage, PerformanceMetrics
from core.errors import ProcessingError, handle_errors, ErrorBoundary
from dataclasses import dataclass, field
import weakref

logger = logging.getLogger(__name__)

__all__ = ['MonitoringManager', 'MetricValidation']

@dataclass
class MetricValidation:
    """Validation rules for metrics."""
    min_value: float = float('-inf')
    max_value: float = float('inf')
    allowed_types: Set[type] = field(default_factory=lambda: {int, float})
    required_fields: Set[str] = field(default_factory=set)
    custom_validator: Optional[callable] = None

class MonitoringManager:
    """Enhanced monitoring system with resource management and validation."""
    
    # Metric validation rules
    METRIC_VALIDATORS = {
        "system_load": MetricValidation(min_value=0, max_value=100),
        "memory_usage": MetricValidation(min_value=0),
        "request_latency": MetricValidation(min_value=0),
        "error_rate": MetricValidation(min_value=0, max_value=1)
    }
    
    def __init__(self):
        # Prometheus registry with automatic cleanup
        self.registry = CollectorRegistry()
        self._registry_ref = weakref.ref(self.registry)
        
        # Initialize metrics with validation
        self._setup_metrics()
        
        # Thread-safe metric cache
        self._metrics_lock = asyncio.Lock()
        self.metrics_cache: Dict[str, List[float]] = {}
        
        # Resource monitoring
        self.resource_usage = ResourceUsage()
        self._resource_monitor_task: Optional[asyncio.Task] = None
        
        # Alert management
        self._alert_lock = asyncio.Lock()
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._should_stop = False
        
        # Start monitoring
        self._start_monitoring()

    def _setup_metrics(self) -> None:
        """Configure Prometheus metrics with validation."""
        try:
            self.metrics = {
                # System metrics
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
                
                # API metrics
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
                
                # Error metrics
                "error_rate": Gauge(
                    "vyper_error_rate",
                    "Current error rate",
                    registry=self.registry
                ),
                
                # Resource metrics
                "cpu_usage": Gauge(
                    "vyper_cpu_usage_percent",
                    "CPU usage percentage",
                    registry=self.registry
                ),
                "memory_available": Gauge(
                    "vyper_memory_available_bytes",
                    "Available memory in bytes",
                    registry=self.registry
                )
            }
            
            # Validate metric configuration
            for metric_name, validator in self.METRIC_VALIDATORS.items():
                if metric_name not in self.metrics:
                    logger.warning(f"Validator defined for non-existent metric: {metric_name}")
            
        except Exception as e:
            logger.error(f"Error setting up metrics: {e}")
            raise

    @handle_errors()
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric with validation and thread safety."""
        try:
            # Validate metric
            if not self._validate_metric(metric_name, value):
                raise ValueError(f"Invalid metric value for {metric_name}: {value}")
            
            metric = self.metrics.get(metric_name)
            if not metric:
                logger.warning(f"Unknown metric: {metric_name}")
                return

            # Thread-safe update of cache
            async with self._metrics_lock:
                if metric_name not in self.metrics_cache:
                    self.metrics_cache[metric_name] = []
                self.metrics_cache[metric_name].append(value)

                # Keep cache size reasonable
                if len(self.metrics_cache[metric_name]) > 1000:
                    self.metrics_cache[metric_name] = self.metrics_cache[metric_name][-1000:]

            # Update Prometheus metric
            # Continuación del método record_metric
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

            # Check alerts
            await self._check_alerts(metric_name, value, labels)
            
            # Push to Prometheus if configured
            if pushgateway_url := os.getenv("PROMETHEUS_PUSHGATEWAY"):
                try:
                    push_to_gateway(
                        pushgateway_url,
                        job='vyper_metrics',
                        registry=self.registry
                    )
                except Exception as e:
                    logger.error(f"Error pushing to Prometheus: {e}")
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
            raise

    def _validate_metric(self, metric_name: str, value: float) -> bool:
        """Validate a metric value against defined rules."""
        try:
            validator = self.METRIC_VALIDATORS.get(metric_name)
            if not validator:
                return True  # No validation rules defined
            
            # Type validation
            if not isinstance(value, tuple(validator.allowed_types)):
                logger.error(f"Invalid type for metric {metric_name}: {type(value)}")
                return False
            
            # Range validation
            if not validator.min_value <= float(value) <= validator.max_value:
                logger.error(
                    f"Value {value} for metric {metric_name} outside valid range "
                    f"[{validator.min_value}, {validator.max_value}]"
                )
                return False
            
            # Custom validation if defined
            if validator.custom_validator and not validator.custom_validator(value):
                logger.error(f"Custom validation failed for metric {metric_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating metric: {e}")
            return False

    async def _check_alerts(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Check if metric value should trigger alerts."""
        try:
            # Define alert thresholds
            thresholds = {
                "error_rate": 0.1,  # 10% error rate
                "system_load": 80,  # 80% load
                "memory_usage": 90,  # 90% memory usage
                "request_latency": 5.0  # 5 seconds latency
            }
            
            if metric_name not in thresholds:
                return
                
            threshold = thresholds[metric_name]
            if value > threshold:
                await self._create_alert(
                    severity="warning",
                    title=f"High {metric_name}",
                    description=f"{metric_name} exceeded threshold: {value} > {threshold}",
                    metric_data={
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "labels": labels
                    }
                )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    async def _create_alert(
        self,
        severity: str,
        title: str,
        description: str,
        metric_data: Dict[str, Any]
    ) -> None:
        """Create and manage an alert."""
        try:
            async with self._alert_lock:
                alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                alert = {
                    "id": alert_id,
                    "severity": severity,
                    "title": title,
                    "description": description,
                    "metric_data": metric_data,
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                }
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Cleanup old alerts
                self._cleanup_old_alerts()
                
                # Send notifications
                await self._send_alert_notifications(alert)
                
        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    async def _send_alert_notifications(self, alert: Dict[str, Any]) -> None:
        """Send alert notifications through configured channels."""
        try:
            # Email notification
            if email_config := os.getenv("ALERT_EMAIL"):
                await self._send_email_alert(alert, email_config)
            
            # Slack notification
            if slack_webhook := os.getenv("SLACK_WEBHOOK_URL"):
                await self._send_slack_alert(alert, slack_webhook)
                
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")

    async def _send_slack_alert(
        self,
        alert: Dict[str, Any],
        webhook_url: str
    ) -> None:
        """Send alert to Slack."""
        try:
            message = {
                "text": f"*{alert['title']}*\n{alert['description']}",
                "attachments": [{
                    "fields": [
                        {"title": "Severity", "value": alert["severity"], "short": True},
                        {"title": "Time", "value": alert["timestamp"], "short": True}
                    ],
                    "color": "danger" if alert["severity"] == "critical" else "warning"
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    if response.status != 200:
                        logger.error(f"Error sending Slack alert: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def _cleanup_old_alerts(self) -> None:
        """Clean up resolved and old alerts."""
        try:
            current_time = datetime.now()
            
            # Remove old alerts from active alerts
            old_alerts = [
                alert_id
                for alert_id, alert in self.active_alerts.items()
                if (current_time - datetime.fromisoformat(alert["timestamp"])).total_seconds() > 3600
            ]
            
            for alert_id in old_alerts:
                alert = self.active_alerts.pop(alert_id)
                alert["status"] = "resolved"
                alert["resolved_at"] = current_time.isoformat()
            
            # Limit alert history size
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")

    async def _monitor_system_metrics(self) -> None:
        """Monitor system metrics periodically."""
        while not self._should_stop:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # Record metrics
                await self.record_metric("cpu_usage", cpu_percent)
                await self.record_metric("memory_usage", memory.used)
                await self.record_metric("memory_available", memory.available)
                
                # Update resource usage
                self.resource_usage.cpu_percent = cpu_percent
                self.resource_usage.memory_percent = memory.percent
                self.resource_usage.disk_usage_percent = psutil.disk_usage('/').percent
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(60)

    def _start_monitoring(self) -> None:
        """Start monitoring background tasks."""
        self._monitoring_tasks = [
            asyncio.create_task(self._monitor_system_metrics()),
            asyncio.create_task(self._monitor_active_alerts())
        ]

    async def cleanup(self) -> None:
        """Clean up resources and stop monitoring."""
        try:
            # Signal stop
            self._should_stop = True
            
            # Cancel monitoring tasks
            for task in self._monitoring_tasks:
                task.cancel()
                
            # Wait for tasks to complete
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
            # Clear metrics and alerts
            self.metrics_cache.clear()
            self.active_alerts.clear()
            
            # Clear Prometheus registry
            if reg := self._registry_ref():
                reg.clear()
            
            logger.info("Monitoring Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        try:
            return {
                "active_metrics": list(self.metrics.keys()),
                "active_alerts": len(self.active_alerts),
                "metrics_cache_size": {
                    metric: len(values)
                    for metric, values in self.metrics_cache.items()
                },
                "resource_usage": self.resource_usage.dict(),
                "monitoring_tasks": {
                    "total": len(self._monitoring_tasks),
                    "running": sum(
                        1 for task in self._monitoring_tasks
                        if not task.done() and not task.cancelled()
                    )
                }
            }
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }