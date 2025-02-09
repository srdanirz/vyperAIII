from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class ProcessingMode(str, Enum):
    """Available processing modes."""
    STANDARD = "standard"
    BATCH = "batch"
    STREAMING = "streaming"
    EDGE = "edge"

class Priority(int, Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class EngineMode(str, Enum):
    """LLM engine modes."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"

@dataclass
class RequestContext:
    """Request context."""
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    mode: ProcessingMode = ProcessingMode.STANDARD
    priority: Priority = Priority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_edge: bool = False
    requires_blockchain: bool = False
    engine_mode: EngineMode = EngineMode.OPENAI

@dataclass
class TeamMember:
    """Team member representation."""
    id: str
    role: str
    capabilities: List[str]
    active_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "ready"

@dataclass
class Team:
    """Work team representation."""
    id: str
    name: str
    members: Dict[str, TeamMember]
    objectives: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active_tasks: Set[str] = field(default_factory=set)

class RequestConfig(BaseModel):
    """Request configuration."""
    mode: ProcessingMode = Field(default=ProcessingMode.STANDARD)
    priority: Priority = Field(default=Priority.MEDIUM)
    timeout: Optional[int] = Field(default=None)
    retry_attempts: int = Field(default=3)
    cache_ttl: Optional[int] = Field(default=None)
    requirements: List[str] = Field(default_factory=list)
    engine_mode: EngineMode = Field(default=EngineMode.OPENAI)

class ResponseMetadata(BaseModel):
    """Response metadata."""
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float
    mode: ProcessingMode
    engine_mode: EngineMode
    cache_hit: bool = False
    optimizations_applied: List[str] = Field(default_factory=list)

class ProcessingResult(BaseModel):
    """Processing result."""
    status: str
    data: Dict[str, Any]
    metadata: ResponseMetadata
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)

class ModelResponse(BaseModel):
    """Standard model response format."""
    content: str
    role: str = "assistant"
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FunctionCall(BaseModel):
    """Function call representation."""
    name: str
    arguments: Dict[str, Any]
    description: Optional[str] = None
    required_permissions: List[str] = Field(default_factory=list)

class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResourceUsage(BaseModel):
    """Resource usage tracking."""
    cpu_percent: float = Field(default=0.0)
    memory_percent: float = Field(default=0.0)
    disk_usage_percent: float = Field(default=0.0)
    network_bytes_sent: int = Field(default=0)
    network_bytes_received: int = Field(default=0)
    active_threads: int = Field(default=0)

class PerformanceMetrics(BaseModel):
    """Performance tracking metrics."""
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    average_response_time: float = Field(default=0.0)
    cache_hits: int = Field(default=0)
    cache_misses: int = Field(default=0)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate system configuration."""
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

def create_request_context(
    request_id: str,
    mode: ProcessingMode = ProcessingMode.STANDARD,
    priority: Priority = Priority.MEDIUM,
    metadata: Optional[Dict[str, Any]] = None,
    engine_mode: EngineMode = EngineMode.OPENAI
) -> RequestContext:
    """Create a new request context."""
    return RequestContext(
        request_id=request_id,
        mode=mode,
        priority=priority,
        metadata=metadata or {},
        engine_mode=engine_mode
    )

def create_team_member(
    member_id: str,
    role: str,
    capabilities: List[str]
) -> TeamMember:
    """Create a new team member."""
    return TeamMember(
        id=member_id,
        role=role,
        capabilities=capabilities
    )

def create_team(
    team_id: str,
    name: str,
    members: Dict[str, TeamMember],
    objectives: List[str]
) -> Team:
    """Create a new team."""
    return Team(
        id=team_id,
        name=name,
        members=members,
        objectives=objectives
    )

def create_processing_result(
    status: str,
    data: Dict[str, Any],
    metadata: ResponseMetadata,
    errors: Optional[List[Dict[str, Any]]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """Create a new processing result."""
    return ProcessingResult(
        status=status,
        data=data,
        metadata=metadata,
        errors=errors or [],
        performance_metrics=performance_metrics or {}
    )

def create_model_response(
    content: str,
    model: str,
    usage: Optional[Dict[str, int]] = None,
    finish_reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ModelResponse:
    """Create a new model response."""
    return ModelResponse(
        content=content,
        model=model,
        usage=usage or {},
        finish_reason=finish_reason,
        metadata=metadata or {}
    )

def create_validation_result(
    is_valid: bool,
    errors: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Create a new validation result."""
    return ValidationResult(
        is_valid=is_valid,
        errors=errors or [],
        warnings=warnings or [],
        metadata=metadata or {}
    )

def create_performance_metrics() -> PerformanceMetrics:
    """Create a new performance metrics instance."""
    return PerformanceMetrics(
        resource_usage=ResourceUsage()
    )