from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    pass

class Config:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config_path: Optional[str] = None, env: str = "development"):
        if hasattr(self, 'initialized'):
            return
            
        self.env = env
        self.config_data = {}
        
        base_config = self._load_yaml_file(Path(__file__).parent / "base.yaml")
        env_config = self._load_yaml_file(Path(__file__).parent / f"{env}.yaml")
        
        self.config_data = self._merge_configs(base_config, env_config)
        
        if config_path:
            custom_config = self._load_yaml_file(Path(config_path))
            self.config_data = self._merge_configs(self.config_data, custom_config)
            
        if not self._validate_config():
            raise ConfigError("Invalid configuration")
            
        self.initialized = True

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open() as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            return {}

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _validate_config(self) -> bool:
        required = {
            "api": ["openai", "deepseek"],
            "agent_roles": ["research", "analysis", "validation"],
            "performance": ["timeouts", "retry"],
            "monitoring": ["enabled", "metrics_interval"],
            "security": ["max_requests", "rate_limit"]
        }
        
        try:
            for section, fields in required.items():
                if section not in self.config_data:
                    logger.error(f"Missing required section: {section}")
                    return False
                for field in fields:
                    if field not in self.config_data[section]:
                        logger.error(f"Missing required field {field} in section {section}")
                        return False
            return True
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False

    @lru_cache(maxsize=1024)
    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = self.config_data
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> Dict[str, Any]:
        return self.config_data.copy()
        
    def reload(self) -> None:
        with self._lock:
            self.initialized = False
            self.__init__(env=self.env)

_config_instance: Optional[Config] = None

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def initialize_config(config_path: Optional[str] = None, env: str = "development") -> None:
    global _config_instance
    with Config._lock:
        _config_instance = Config(config_path, env)