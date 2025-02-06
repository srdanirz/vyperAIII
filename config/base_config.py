# config/base_config.py

from typing import Dict, Any
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseConfig:
    """Configuración base del sistema"""
    
    DEFAULT_CONFIG = {
        "version": "2.0.0",
        "api": {
            "openai": {
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "deepseek": {
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        },
        "templates": {
            "powerpoint": {
                "themes": {
                    "professional": {
                        "colors": {
                            "primary": ["#2C3E50", "#E74C3C"],
                            "secondary": ["#ECF0F1", "#95A5A6"],
                            "accent": "#3498DB"
                        },
                        "fonts": {
                            "title": {"name": "Calibri", "size": 44},
                            "heading": {"name": "Calibri", "size": 32},
                            "body": {"name": "Calibri", "size": 24}
                        },
                        "layouts": {
                            "title": {"title_size": 44, "subtitle_size": 32},
                            "content": {"title_size": 36, "content_size": 24},
                            "two_column": {"title_size": 36, "content_size": 24},
                            "comparison": {"title_size": 36, "content_size": 20},
                            "image": {"title_size": 36, "caption_size": 18}
                        }
                    },
                    "modern": {
                        "colors": {
                            "primary": ["#1A237E", "#311B92"],
                            "secondary": ["#FAFAFA", "#F5F5F5"],
                            "accent": "#00BCD4"
                        },
                        "fonts": {
                            "title": {"name": "Helvetica", "size": 40},
                            "heading": {"name": "Helvetica", "size": 30},
                            "body": {"name": "Helvetica", "size": 22}
                        }
                    },
                    "creative": {
                        "colors": {
                            "primary": ["#6200EA", "#651FFF"],
                            "secondary": ["#FFFFFF", "#F3E5F5"],
                            "accent": "#00BFA5"
                        },
                        "fonts": {
                            "title": {"name": "Georgia", "size": 42},
                            "heading": {"name": "Georgia", "size": 34},
                            "body": {"name": "Arial", "size": 24}
                        }
                    }
                },
                "master_layouts": ["title", "content", "two_column", "comparison", "image", "section"],
                "default_theme": "professional"
            },
            "document": {
                "styles": {
                    "professional": {
                        "fonts": {
                            "title": {"name": "Arial", "size": 16, "bold": True},
                            "heading1": {"name": "Arial", "size": 14, "bold": True},
                            "heading2": {"name": "Arial", "size": 12, "bold": True},
                            "body": {"name": "Arial", "size": 11, "bold": False}
                        },
                        "colors": {
                            "title": "#000000",
                            "heading": "#2C3E50",
                            "body": "#333333"
                        },
                        "margins": {
                            "top": 1.0,
                            "bottom": 1.0,
                            "left": 1.0,
                            "right": 1.0
                        }
                    }
                }
            }
        },
        "agent_roles": {
            "researcher": {
                "title": "Research Director",
                "description": "Especialista en investigación y análisis de información",
                "capabilities": ["research", "analysis", "data_collection"],
                "priority": 1
            },
            "analysis": {
                "title": "Analysis Director",
                "description": "Experto en análisis de datos y generación de insights",
                "capabilities": ["data_analysis", "pattern_recognition", "insight_generation"],
                "priority": 2
            },
            "contentgeneration": {
                "title": "Content Director",
                "description": "Especialista en generación de contenido de alta calidad",
                "capabilities": ["content_creation", "formatting", "style_application"],
                "priority": 3
            },
            "validation": {
                "title": "Quality Assurance Director",
                "description": "Experto en validación y control de calidad",
                "capabilities": ["quality_control", "validation", "improvement_suggestions"],
                "priority": 4
            }
        },
        "output_formats": {
            "powerpoint": {
                "extensions": [".pptx"],
                "mime_types": ["application/vnd.openxmlformats-officedocument.presentationml.presentation"]
            },
            "document": {
                "extensions": [".docx", ".pdf"],
                "mime_types": [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/pdf"
                ]
            }
        },
        "validation": {
            "quality_thresholds": {
                "min_acceptable_score": 7.5,
                "excellent_score": 9.0
            },
            "criteria_weights": {
                "clarity": 0.2,
                "accuracy": 0.25,
                "coherence": 0.15,
                "relevance": 0.2,
                "completeness": 0.2
            }
        },
        "performance": {
            "timeouts": {
                "research": 300,
                "analysis": 180,
                "content_generation": 240,
                "validation": 120
            },
            "retry": {
                "max_attempts": 3,
                "delay_base": 2,
                "max_delay": 30
            }
        }
    }

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path:
            self.load_config()

    def load_config(self) -> None:
        """Carga configuración desde archivo"""
        try:
            path = Path(self.config_path)
            if not path.exists():
                logger.warning(f"Config file not found at {path}, using defaults")
                return

            with path.open('r') as f:
                if path.suffix == '.yaml':
                    loaded_config = yaml.safe_load(f)
                elif path.suffix == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path.suffix}")

            self.config = self._merge_configs(self.config, loaded_config)
            logger.info(f"Configuration loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def save_config(self, path: str = None) -> None:
        """Guarda la configuración actual"""
        try:
            save_path = Path(path or self.config_path)
            if not save_path:
                raise ValueError("No config path specified")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_to_save = {
                **self.config,
                "_metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "version": self.config["version"]
                }
            }

            with save_path.open('w') as f:
                if save_path.suffix == '.yaml':
                    yaml.dump(config_to_save, f, default_flow_style=False)
                elif save_path.suffix == '.json':
                    json.dump(config_to_save, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path.suffix}")

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Combina configuraciones de manera recursiva"""
        merged = base.copy()
        
        for key, value in override.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Establece un valor de configuración"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value