from .adaptive_system import AdaptiveSystem
from .adaptive_meta import AdaptiveMetaSystem

__version__ = "1.0.0"

__all__ = [
    'AdaptiveSystem',
    'AdaptiveMetaSystem'
]

# System states
ADAPTATION_MODES = {
    "LEARNING": "learning",      # Sistema está aprendiendo patrones
    "OPTIMIZING": "optimizing",  # Sistema está optimizando comportamiento
    "GENERATING": "generating",  # Sistema está generando nuevo código
    "STABLE": "stable"          # Sistema está en estado estable
}

# Tipos de adaptación
ADAPTATION_TYPES = {
    "BEHAVIORAL": "behavioral",  # Adaptación de comportamiento
    "STRUCTURAL": "structural",  # Adaptación de estructura
    "CODE": "code",             # Adaptación de código
    "RUNTIME": "runtime"        # Adaptación en tiempo de ejecución
}

def get_adaptive_system(config: dict = None) -> AdaptiveSystem:
    """Obtiene una instancia del sistema adaptativo."""
    return AdaptiveSystem() if config is None else AdaptiveSystem(config)

def get_meta_system(base_dir: str = "core") -> AdaptiveMetaSystem:
    """Obtiene una instancia del sistema de metaprogramación."""
    return AdaptiveMetaSystem(base_dir=base_dir)