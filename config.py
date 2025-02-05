import os
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
TEMPLATES_DIR = BASE_DIR / "templates"
LOGS_DIR = BASE_DIR / "logs"

CACHE_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

API_CONFIG = {
    "openai": {
        "model": "gpt-4-turbo",
        "temperature": 0.7,
        "max_tokens": 4000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    "timeout": {
        "default": 300,
        "long": 900,
        "short": 60
    }
}

CACHE_CONFIG = {
    "expiration_hours": 24,
    "max_size_mb": 1000,
    "cleanup_interval": 3600
}

AGENT_CONFIG = {
    "retry": {
        "max_attempts": 3,
        "base_delay": 2,
        "max_delay": 30
    },
    "timeout": {
        "browser": 180,
        "content_generation": 300,
        "research": 240,
        "analysis": 120,
        "validation": 60
    }
}

SERVICE_ENDPOINTS = {
    "google": {
        "docs": "https://docs.googleapis.com/v1",
        "drive": "https://www.googleapis.com/drive/v3",
        "sheets": "https://sheets.googleapis.com/v4"
    },
    "microsoft": {
        "graph": "https://graph.microsoft.com/v1.0",
        "sharepoint": "https://{tenant}.sharepoint.com"
    }
}

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        "detailed": {"format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "assistant.log"),
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

TEMPLATE_CONFIG = {
    "presentation": {
        "slide_layouts": {
            "title": {"title_size": 44, "subtitle_size": 32},
            "content": {"title_size": 36, "content_size": 24},
            "two_column": {"title_size": 36, "content_size": 24}
        },
        "color_schemes": {
            "professional": {
                "primary": [(26, 54, 113), (44, 86, 151)],
                "accent": [(255, 217, 102)],
                "text": [(0, 0, 0), (89, 89, 89)]
            },
            "modern": {
                "primary": [(0, 178, 255), (0, 204, 255)],
                "accent": [(255, 140, 0)],
                "text": [(51, 51, 51), (102, 102, 102)]
            }
        }
    },
    "document": {
        "styles": {
            "heading1": {"size": 16, "bold": True},
            "heading2": {"size": 14, "bold": True},
            "normal": {"size": 11, "bold": False}
        },
        "margins": {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0}
    }
}

PERFORMANCE_CONFIG = {
    "metrics": {
        "execution_time": True,
        "memory_usage": True,
        "api_calls": True
    },
    "thresholds": {
        "execution_time": 300,
        "memory_usage": 1024,
        "api_calls": 100
    }
}

def get_env_var(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)

def get_config() -> Dict[str, Any]:
    return {
        "api": API_CONFIG,
        "cache": CACHE_CONFIG,
        "agent": AGENT_CONFIG,
        "services": SERVICE_ENDPOINTS,
        "logging": LOG_CONFIG,
        "templates": TEMPLATE_CONFIG,
        "performance": PERFORMANCE_CONFIG
    }

def update_config(section: str, updates: Dict[str, Any]) -> None:
    config = globals().get(f"{section.upper()}_CONFIG")
    if config:
        config.update(updates)
