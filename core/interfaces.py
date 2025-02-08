from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class ProcessingMode(str, Enum):
    """Modos de procesamiento disponibles."""
    STANDARD = "standard"
    BATCH = "batch"
    STREAMING = "streaming"
    EDGE = "edge"

class Priority(int, Enum):
    """Niveles de prioridad para tareas."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class RequestContext:
    """Contexto de una solicitud."""
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    mode: ProcessingMode = ProcessingMode.STANDARD
    priority: Priority = Priority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_edge: bool = False
    requires_blockchain: bool = False

@dataclass
class TeamMember:
    """Miembro de un equipo."""
    id: str
    role: str
    capabilities: List[str]
    active_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "ready"

@dataclass
class Team:
    """Equipo de trabajo."""
    id: str
    name: str
    members: Dict[str, TeamMember]
    objectives: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active_tasks: Set[str] = field(default_factory=set)

class RequestConfig(BaseModel):
    """Configuración de una solicitud."""
    mode: ProcessingMode = Field(default=ProcessingMode.STANDARD)
    priority: Priority = Field(default=Priority.MEDIUM)
    timeout: Optional[int] = Field(default=None)
    retry_attempts: int = Field(default=3)
    cache_ttl: Optional[int] = Field(default=None)
    requirements: List[str] = Field(default_factory=list)

class ResponseMetadata(BaseModel):
    """Metadatos de una respuesta."""
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float
    mode: ProcessingMode
    cache_hit: bool = False
    optimizations_applied: List[str] = Field(default_factory=list)

class ProcessingResult(BaseModel):
    """Resultado del procesamiento."""
    status: str
    data: Dict[str, Any]
    metadata: ResponseMetadata
    errors: List[Dict[str, Any]] = Field(default_factory=list)

def validate_config(config: Dict[str, Any]) -> bool:
    """Valida configuración del sistema."""
    try:
        required = {
            "api": ["openai", "deepseek"],
            "agent_roles": ["research", "analysis", "validation"],
            "performance": ["timeouts", "retry"],
            "monitoring": ["enabled", "metrics_interval"],
            "security": ["max_requests", "rate_limit"]
        }
        
        for section, fields in required.items():
            if section not in config:
                return False
            for field in fields:
                if field not in config[section]:
                    return False
        return True
        
    except Exception:
        return False