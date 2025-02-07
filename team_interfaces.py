from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class TeamRequest:
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    priority: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TeamResponse:
    status: str
    result: Dict[str, Any]
    execution_time: float
    team_ids: List[str]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TeamMember:
    id: str
    role: str
    capabilities: List[str]
    active_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Team:
    id: str
    name: str
    members: Dict[str, TeamMember]
    objectives: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

def validate_config(config: Dict[str, Any]) -> bool:
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