import logging
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from core.llm.deepseek import DeepSeekChat
from .errors import ConfigurationError

logger = logging.getLogger(__name__)

class LLMProvider:
    """Gestiona la creación y configuración de modelos de lenguaje."""
    
    SUPPORTED_ENGINES = {
        "openai": {
            "default_model": "gpt-4-turbo",
            "class": ChatOpenAI
        },
        "deepseek": {
            "default_model": "deepseek-chat",
            "class": DeepSeekChat.llm
        }
    }

    @classmethod
    def get_llm(
        cls,
        engine_mode: str,
        api_key: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """
        Obtiene una instancia de modelo de lenguaje.
        
        Args:
            engine_mode: Tipo de motor ("openai" o "deepseek")
            api_key: API key del proveedor
            model: Nombre del modelo (opcional)
            temperature: Temperatura para generación
            **kwargs: Argumentos adicionales para el modelo
            
        Returns:
            Instancia del modelo con interfaz .agenerate()
            
        Raises:
            ConfigurationError: Si el engine_mode no es soportado
        """
        try:
            engine_mode = engine_mode.lower()
            if engine_mode not in cls.SUPPORTED_ENGINES:
                raise ConfigurationError(f"Engine mode no soportado: {engine_mode}")

            engine_config = cls.SUPPORTED_ENGINES[engine_mode]
            
            # Determinar modelo
            if not model:
                model = engine_config["default_model"]
            elif engine_mode == "deepseek" and model.startswith("gpt-"):
                logger.warning(f"Modelo {model} no soportado en DeepSeek, usando default")
                model = engine_config["default_model"]

            # Crear instancia
            return engine_config["class"](
                api_key=api_key,
                model=model,
                temperature=temperature,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Error creando LLM: {e}")
            raise ConfigurationError(f"Error configurando LLM: {str(e)}")

# Función helper para compatibilidad
def get_llm(*args, **kwargs) -> Any:
    """Helper function que mantiene la interfaz anterior."""
    return LLMProvider.get_llm(*args, **kwargs)