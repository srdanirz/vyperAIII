# plugins/plugin_manager.py

import logging
import asyncio
import importlib
import inspect
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Gestor de plugins que permite la extensión dinámica del sistema.
    
    Características:
    - Carga dinámica de plugins
    - Sistema de hooks
    - Gestión de dependencias
    - Hot-reloading
    - Métricas y monitoreo
    """
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_dependencies: Dict[str, List[str]] = {}
        self.active_hooks: Dict[str, bool] = {}
        
        # Métricas
        self.metrics = {
            "loaded_plugins": 0,
            "active_hooks": 0,
            "failed_loads": 0,
            "execution_times": {}
        }
        
        # Cargar configuración
        self._load_plugin_config()

    async def load_plugin(
        self,
        plugin_name: str,
        plugin_path: Optional[str] = None
    ) -> bool:
        """
        Carga un plugin dinámicamente.
        
        Args:
            plugin_name: Nombre del plugin
            plugin_path: Ruta opcional al plugin
        """
        try:
            # Verificar dependencias
            if not await self._check_dependencies(plugin_name):
                logger.error(f"Dependencies not met for plugin: {plugin_name}")
                return False
            
            # Determinar ruta
            if not plugin_path:
                plugin_path = f"plugins.{plugin_name}"
            
            # Cargar módulo
            module = importlib.import_module(plugin_path)
            
            # Verificar interfaz
            if not hasattr(module, 'setup') or not hasattr(module, 'execute'):
                raise ValueError(f"Invalid plugin interface: {plugin_name}")
            
            # Configurar plugin
            config = self.plugin_configs.get(plugin_name, {})
            await module.setup(config)
            
            # Registrar hooks
            self._register_hooks(module, plugin_name)
            
            # Guardar plugin
            self.plugins[plugin_name] = module
            self.metrics["loaded_plugins"] += 1
            
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            self.metrics["failed_loads"] += 1
            return False

    async def execute_hook(
        self,
        hook_name: str,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Ejecuta los hooks registrados para un evento.
        
        Args:
            hook_name: Nombre del hook a ejecutar
            args, kwargs: Argumentos para el hook
        """
        try:
            if hook_name not in self.hooks:
                return []
            
            results = []
            start_time = datetime.now()
            
            # Ejecutar hooks en paralelo
            tasks = [
                self._execute_single_hook(hook, *args, **kwargs)
                for hook in self.hooks[hook_name]
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Actualizar métricas
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics["execution_times"][hook_name] = execution_time
            
            return [
                r for r in results
                if not isinstance(r, Exception)
            ]
            
        except Exception as e:
            logger.error(f"Error executing hook {hook_name}: {e}")
            return []

    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Recarga un plugin en caliente.
        
        Args:
            plugin_name: Nombre del plugin a recargar
        """
        try:
            # Descargar plugin actual
            if plugin_name in self.plugins:
                await self.unload_plugin(plugin_name)
            
            # Recargar
            return await self.load_plugin(plugin_name)
            
        except Exception as e:
            logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Descarga un plugin.
        
        Args:
            plugin_name: Nombre del plugin a descargar
        """
        try:
            if plugin_name not in self.plugins:
                return False
            
            # Ejecutar limpieza del plugin
            plugin = self.plugins[plugin_name]
            if hasattr(plugin, 'cleanup'):
                await plugin.cleanup()
            
            # Eliminar hooks
            for hooks in self.hooks.values():
                hooks[:] = [h for h in hooks if getattr(h, '__module__', None) != plugin_name]
            
            # Eliminar plugin
            del self.plugins[plugin_name]
            self.metrics["loaded_plugins"] -= 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def _load_plugin_config(self) -> None:
        """Carga configuración de plugins desde YAML."""
        try:
            config_path = Path(__file__).parent / "plugin_config.yaml"
            if not config_path.exists():
                return
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            self.plugin_configs = config.get("plugins", {})
            self.plugin_dependencies = config.get("dependencies", {})
            
        except Exception as e:
            logger.error(f"Error loading plugin config: {e}")

    def _register_hooks(self, module: Any, plugin_name: str) -> None:
        """Registra los hooks de un plugin."""
        try:
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if hasattr(func, '_hook'):
                    hook_name = getattr(func, '_hook')
                    if hook_name not in self.hooks:
                        self.hooks[hook_name] = []
                    self.hooks[hook_name].append(func)
                    self.active_hooks[hook_name] = True
                    self.metrics["active_hooks"] += 1
                    
        except Exception as e:
            logger.error(f"Error registering hooks for {plugin_name}: {e}")

    async def _check_dependencies(self, plugin_name: str) -> bool:
        """Verifica las dependencias de un plugin."""
        try:
            dependencies = self.plugin_dependencies.get(plugin_name, [])
            
            for dep in dependencies:
                if dep not in self.plugins:
                    logger.error(f"Missing dependency {dep} for plugin {plugin_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies for {plugin_name}: {e}")
            return False

    async def _execute_single_hook(
        self,
        hook: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Ejecuta un único hook con manejo de errores."""
        try:
            start_time = datetime.now()
            result = await hook(*args, **kwargs)
            
            # Actualizar métricas
            execution_time = (datetime.now() - start_time).total_seconds()
            hook_name = getattr(hook, '_hook', 'unknown')
            if hook_name not in self.metrics["execution_times"]:
                self.metrics["execution_times"][hook_name] = []
            self.metrics["execution_times"][hook_name].append(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing hook {hook.__name__}: {e}")
            raise

    async def get_plugin_status(self, plugin_name: str) -> Dict[str, Any]:
        """Obtiene el estado de un plugin."""
        try:
            if plugin_name not in self.plugins:
                return {"error": "Plugin not found"}
            
            plugin = self.plugins[plugin_name]
            
            return {
                "name": plugin_name,
                "active": True,
                "hooks": [
                    hook_name
                    for hook_name, hooks in self.hooks.items()
                    if any(h.__module__ == plugin.__name__ for h in hooks)
                ],
                "config": self.plugin_configs.get(plugin_name, {}),
                "metrics": {
                    "execution_times": self.metrics["execution_times"].get(plugin_name, [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting status for plugin {plugin_name}: {e}")
            return {"error": str(e)}

    async def get_all_metrics(self) -> Dict[str, Any]:
        """Obtiene todas las métricas del sistema de plugins."""
        return {
            "total_plugins": self.metrics["loaded_plugins"],
            "active_hooks": self.metrics["active_hooks"],
            "failed_loads": self.metrics["failed_loads"],
            "performance": {
                hook: {
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                    "min_time": min(times)
                }
                for hook, times in self.metrics["execution_times"].items()
                if times
            }
        }
