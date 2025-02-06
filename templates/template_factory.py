# templates/template_factory.py

import logging
from typing import Dict, Any, Type
from .base_template import BaseTemplate
from .presentation_template import PresentationTemplate
from .document_template import DocumentTemplate
from .visualization_template import VisualizationTemplate

logger = logging.getLogger(__name__)

class TemplateRegistry:
    """Registro central de templates disponibles"""
    
    _templates: Dict[str, Type[BaseTemplate]] = {
        "presentation": PresentationTemplate,
        "document": DocumentTemplate,
        "visualization": VisualizationTemplate
    }

    @classmethod
    def register(cls, name: str, template_class: Type[BaseTemplate]) -> None:
        """Registra un nuevo tipo de template"""
        if not issubclass(template_class, BaseTemplate):
            raise ValueError(f"Template class must inherit from BaseTemplate")
        cls._templates[name] = template_class
        logger.info(f"Registered new template type: {name}")

    @classmethod
    def get_template_class(cls, name: str) -> Type[BaseTemplate]:
        """Obtiene una clase de template por nombre"""
        if name not in cls._templates:
            raise ValueError(f"Unknown template type: {name}")
        return cls._templates[name]

    @classmethod
    def available_templates(cls) -> Dict[str, Type[BaseTemplate]]:
        """Retorna todos los templates disponibles"""
        return cls._templates.copy()

class TemplateFactory:
    """Factory para crear instancias de templates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache: Dict[str, BaseTemplate] = {}

    def create_template(self, template_type: str, template_name: str) -> BaseTemplate:
        """Crea una instancia de template"""
        try:
            # Usar caché si existe
            cache_key = f"{template_type}_{template_name}"
            if cache_key in self.cache:
                logger.debug(f"Using cached template: {cache_key}")
                return self.cache[cache_key]

            # Obtener clase de template
            template_class = TemplateRegistry.get_template_class(template_type)
            
            # Crear instancia
            template = template_class(template_name, self.config)
            
            # Validar template
            if not template.validate():
                raise ValueError(f"Invalid template configuration for {template_type}:{template_name}")
            
            # Guardar en caché
            self.cache[cache_key] = template
            
            logger.info(f"Created new template: {template_type}:{template_name}")
            return template

        except Exception as e:
            logger.error(f"Error creating template {template_type}:{template_name}: {e}")
            raise

    def clear_cache(self) -> None:
        """Limpia el caché de templates"""
        self.cache.clear()
        logger.debug("Template cache cleared")

    def get_template_info(self, template_type: str) -> Dict[str, Any]:
        """Obtiene información sobre un tipo de template"""
        try:
            template_class = TemplateRegistry.get_template_class(template_type)
            return {
                "type": template_type,
                "class": template_class.__name__,
                "required_sections": getattr(template_class, 'REQUIRED_SECTIONS', []),
                "supported_formats": getattr(template_class, 'SUPPORTED_FORMATS', []),
                "description": template_class.__doc__
            }
        except Exception as e:
            logger.error(f"Error getting template info for {template_type}: {e}")
            raise