from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional, Union
from functools import lru_cache
import threading
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)

class EngineMode(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"

class APIConfig(BaseModel):
    openai: Dict[str, Any] = Field(
        default={
            "model": "gpt-4-turbo",
            "temperature": 0.7,
            "max_tokens": 4000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )
    deepseek: Dict[str, Any] = Field(
        default={
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 4000
        }
    )

class PerformanceConfig(BaseModel):
    timeouts: Dict[str, int] = Field(
        default={
            "research": 300,
            "analysis": 180,
            "content_generation": 240,
            "validation": 120
        }
    )
    retry: Dict[str, Union[int, float]] = Field(
        default={
            "max_attempts": 3,
            "delay_base": 2,
            "max_delay": 30
        }
    )

class MonitoringConfig(BaseModel):
    enabled: bool = True
    metrics_interval: int = 60
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

class SecurityConfig(BaseModel):
    max_requests: int = 100
    rate_limit: str = "60/minute"
    allowed_ips: list = Field(default=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])

class Config(BaseModel):
    api: APIConfig = Field(default_factory=APIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    engine_mode: EngineMode = EngineMode.OPENAI
    environment: str = "development"

class ConfigManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config: Optional[Config] = None
        self._load_paths = [
            Path("config/base.yaml"),
            Path("config/development.yaml"),
            Path("config/production.yaml")
        ]
        self._initialized = True

    def initialize(self, env: str = "development", config_path: Optional[str] = None) -> None:
        """Initialize configuration with optional custom path"""
        try:
            base_config = self._load_yaml_file(self._load_paths[0])
            
            # Load environment specific config
            env_path = Path(f"config/{env}.yaml")
            env_config = self._load_yaml_file(env_path)
            
            # Load custom config if provided
            custom_config = {}
            if config_path:
                custom_config = self._load_yaml_file(Path(config_path))
            
            # Merge configurations
            merged_config = {
                **base_config,
                **env_config,
                **custom_config,
                "environment": env
            }
            
            # Validate and create config
            self.config = Config(**merged_config)
            
            # Configure logging
            self._setup_logging()
            
            logger.info(f"Configuration initialized for environment: {env}")
            
        except Exception as e:
            logger.error(f"Error initializing configuration: {e}")
            raise

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Load and parse YAML file with error handling"""
        try:
            if not path.exists():
                logger.warning(f"Config file not found: {path}")
                return {}
                
            with open(path) as f:
                return yaml.safe_load(f) or {}
                
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config"""
        if not self.config:
            return
            
        log_config = self.config.monitoring
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Add file handler if log_file specified
        if log_config.log_file:
            file_handler = logging.FileHandler(log_config.log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            logging.getLogger().addHandler(file_handler)

    @property
    def api(self) -> APIConfig:
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        return self.config.api

    @property
    def performance(self) -> PerformanceConfig:
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        return self.config.performance

    @property
    def monitoring(self) -> MonitoringConfig:
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        return self.config.monitoring

    @property
    def security(self) -> SecurityConfig:
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        return self.config.security

    @lru_cache(maxsize=None)
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation key with caching"""
        if not self.config:
            raise RuntimeError("Configuration not initialized")
            
        try:
            value = self.config
            for k in key.split('.'):
                value = getattr(value, k)
            return value
        except AttributeError:
            return default

    def reload(self) -> None:
        """Reload configuration"""
        if not self.config:
            raise RuntimeError("Configuration not initialized")
        self.initialize(self.config.environment)
        get_config.cache_clear()

# Global config instance
_config_instance: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def initialize_config(env: str = "development", config_path: Optional[str] = None) -> None:
    """Initialize global configuration"""
    config_manager = get_config()
    config_manager.initialize(env, config_path)