import logging
import asyncio
import importlib
import inspect
import ast
from typing import Dict, Any, List, Optional, Type, Callable, Set
from pathlib import Path
from datetime import datetime
import yaml
import json

from core.errors import ProcessingError, handle_errors, ErrorBoundary
from .base_plugin import BasePlugin, hook

logger = logging.getLogger(__name__)

class CodeTransformer(ast.NodeTransformer):
    """AST transformer for modifying plugin code."""
    
    def __init__(self, modifications: List[Dict[str, Any]]):
        self.modifications = modifications
        super().__init__()
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Modify function definitions."""
        for mod in self.modifications:
            if mod.get("target") == "function" and mod.get("name") == node.name:
                if "new_name" in mod:
                    node.name = mod["new_name"]
                if "decorators" in mod:
                    node.decorator_list.extend([
                        ast.Name(id=dec, ctx=ast.Load())
                        for dec in mod["decorators"]
                    ])
                if "async" in mod:
                    if mod["async"] and not isinstance(node, ast.AsyncFunctionDef):
                        return ast.AsyncFunctionDef(
                            name=node.name,
                            args=node.args,
                            body=node.body,
                            decorator_list=node.decorator_list,
                            returns=node.returns
                        )
        return self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Modify class definitions."""
        for mod in self.modifications:
            if mod.get("target") == "class" and mod.get("name") == node.name:
                if "new_name" in mod:
                    node.name = mod["new_name"]
                if "bases" in mod:
                    node.bases = [
                        ast.Name(id=base, ctx=ast.Load())
                        for base in mod["bases"]
                    ]
        return self.generic_visit(node)
        
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Modify assignments."""
        for mod in self.modifications:
            if mod.get("target") == "assignment":
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == mod.get("name"):
                        node.value = ast.parse(mod["value"]).body[0].value
        return self.generic_visit(node)

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
        self.plugins: Dict[str, BasePlugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_dependencies: Dict[str, List[str]] = {}
        self.active_hooks: Dict[str, bool] = {}
        
        # Control de estado
        self._should_stop = False
        self._watch_task: Optional[asyncio.Task] = None
        
        # Métricas
        self.metrics = {
            "loaded_plugins": 0,
            "active_hooks": 0,
            "failed_loads": 0,
            "execution_times": {},
            "last_error": None
        }
        
        # Cargar configuración
        self._load_plugin_config()

    async def initialize(self) -> None:
        """Inicializa el sistema de plugins."""
        try:
            # Crear directorios necesarios
            plugin_dir = Path("plugins")
            plugin_dir.mkdir(exist_ok=True)
            
            # Cargar plugins habilitados
            enabled_plugins = self.plugin_configs.get("enabled_plugins", [])
            for plugin_name in enabled_plugins:
                await self.load_plugin(plugin_name)
                
            # Iniciar monitoreo si está habilitado
            if self.plugin_configs.get("auto_reload", False):
                self._watch_task = asyncio.create_task(self._watch_plugins())
                
            logger.info("Plugin system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing plugin system: {e}")
            raise ProcessingError("Plugin system initialization failed", {"error": str(e)})

    @handle_errors()
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
            
            # Buscar clase de plugin
            plugin_class = None
            for item_name, item in inspect.getmembers(module):
                if (inspect.isclass(item) and 
                    issubclass(item, BasePlugin) and 
                    item != BasePlugin):
                    plugin_class = item
                    break
                    
            if not plugin_class:
                raise ValueError(f"No valid plugin class found in {plugin_name}")
            
            # Instanciar plugin
            plugin = plugin_class(plugin_name)
            
            # Configurar plugin
            config = self.plugin_configs.get(plugin_name, {})
            await plugin.setup(config)
            
            # Registrar hooks
            self._register_hooks(plugin)
            
            # Guardar plugin
            self.plugins[plugin_name] = plugin
            self.metrics["loaded_plugins"] += 1
            
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            self.metrics["failed_loads"] += 1
            self.metrics["last_error"] = {
                "timestamp": datetime.now().isoformat(),
                "plugin": plugin_name,
                "error": str(e)
            }
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

    def _register_hooks(self, plugin: BasePlugin) -> None:
        """Registra los hooks de un plugin."""
        try:
            plugin_hooks: Set[str] = set()
            
            for name, method in inspect.getmembers(plugin, inspect.ismethod):
                if hasattr(method, '_hook'):
                    hook_name = getattr(method, '_hook')
                    if hook_name not in self.hooks:
                        self.hooks[hook_name] = []
                    self.hooks[hook_name].append(method)
                    plugin_hooks.add(hook_name)
                    self.active_hooks[hook_name] = True
                    self.metrics["active_hooks"] += 1
            
            # Guardar hooks del plugin para desregistro
            plugin._registered_hooks = plugin_hooks
            
        except Exception as e:
            logger.error(f"Error registering hooks for {plugin.name}: {e}")

    def _unregister_hooks(self, plugin: BasePlugin) -> None:
        """Elimina los hooks de un plugin."""
        try:
            if not hasattr(plugin, '_registered_hooks'):
                return
                
            for hook_name in plugin._registered_hooks:
                if hook_name in self.hooks:
                    # Filtrar hooks del plugin
                    self.hooks[hook_name] = [
                        h for h in self.hooks[hook_name]
                        if not hasattr(h, '__self__') or h.__self__ != plugin
                    ]
                    
                    if not self.hooks[hook_name]:
                        del self.hooks[hook_name]
                        self.active_hooks[hook_name] = False
                        self.metrics["active_hooks"] -= 1
                        
        except Exception as e:
            logger.error(f"Error unregistering hooks for {plugin.name}: {e}")

    async def _check_dependencies(self, plugin_name: str) -> bool:
        """Verifica las dependencias de un plugin."""
        try:
            dependencies = self.plugin_dependencies.get(plugin_name, [])
            
            for dep in dependencies:
                if dep not in self.plugins:
                    logger.error(f"Missing dependency {dep} for plugin {plugin_name}")
                    return False
                    
                # Verificar estado del plugin dependencia
                if not await self.plugins[dep].get_status():
                    logger.error(f"Dependency {dep} is not ready")
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
            
            # Ejecutar hook con timeout
            result = await asyncio.wait_for(
                hook(*args, **kwargs),
                timeout=self.plugin_configs.get("hook_timeout", 30)
            )
            
            # Actualizar métricas
            execution_time = (datetime.now() - start_time).total_seconds()
            hook_name = getattr(hook, '_hook', 'unknown')
            if hook_name not in self.metrics["execution_times"]:
                self.metrics["execution_times"][hook_name] = []
            self.metrics["execution_times"][hook_name].append(execution_time)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Hook {hook.__name__} timed out")
            raise
        except Exception as e:
            logger.error(f"Error executing hook {hook.__name__}: {e}")
            raise

    async def _watch_plugins(self) -> None:
        """Monitorea cambios en plugins para hot-reload."""
        try:
            plugin_dir = Path("plugins")
            last_modified: Dict[str, float] = {}
            
            while not self._should_stop:
                # Verificar cada archivo .py
                for file_path in plugin_dir.glob("*.py"):
                    if file_path.name == "__init__.py":
                        continue
                        
                    mtime = file_path.stat().st_mtime
                    if file_path.stem in last_modified:
                        if mtime > last_modified[file_path.stem]:
                            # Archivo modificado, recargar plugin
                            logger.info(f"Detected changes in {file_path.stem}")
                            await self.reload_plugin(file_path.stem)
                    
                    last_modified[file_path.stem] = mtime
                    
                await asyncio.sleep(2)  # Verificar cada 2 segundos
                
        except Exception as e:
            logger.error(f"Error watching plugins: {e}")

    async def cleanup(self) -> None:
        """Limpia recursos del sistema de plugins."""
        try:
            # Detener monitoreo
            self._should_stop = True
            if self._watch_task:
                self._watch_task.cancel()
                
            # Descargar plugins
            for plugin_name in list(self.plugins.keys()):
                await self.unload_plugin(plugin_name)
            
            # Limpiar estado
            self.hooks.clear()
            self.active_hooks.clear()
            
            logger.info("Plugin system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

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
                    if any(
                        hasattr(h, '__self__') and h.__self__ == plugin
                        for h in hooks
                    )
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
            "last_error": self.metrics["last_error"],
            "performance": {
                hook: {
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                    "min_time": min(times),
                    "total_executions": len(times)
                }
                for hook, times in self.metrics["execution_times"].items()
                if times
            }
        }

    def _check_health(self) -> bool:
        """Verifica el estado de salud del sistema de plugins."""
        try:
            # Verificar error rate
            if self.metrics["failed_loads"] > self.metrics["loaded_plugins"] * 0.2:
                return False
            
            # Verificar hooks activos
            if not self.active_hooks:
                return False
                
            # Verificar timeouts en ejecuciones
            for hook_times in self.metrics["execution_times"].values():
                if any(t > self.plugin_configs.get("hook_timeout", 30) for t in hook_times):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking plugin system health: {e}")
            return False

    @handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema de plugins."""
        try:
            # Obtener estado de plugins activos
            active_plugins = {}
            for name, plugin in self.plugins.items():
                try:
                    active_plugins[name] = await plugin.get_status()
                except Exception as e:
                    logger.error(f"Error getting status for plugin {name}: {e}")
                    active_plugins[name] = {
                        "error": str(e),
                        "status": "error"
                    }

            return {
                "active_plugins": active_plugins,
                "hooks": {
                    name: len(hooks)
                    for name, hooks in self.hooks.items()
                },
                "metrics": await self.get_all_metrics(),
                "config": {
                    "auto_reload": self.plugin_configs.get("auto_reload", False),
                    "hook_timeout": self.plugin_configs.get("hook_timeout", 30),
                    "enabled_plugins": self.plugin_configs.get("enabled_plugins", [])
                },
                "health": {
                    "status": "healthy" if self._check_health() else "degraded",
                    "last_error": self.metrics["last_error"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting plugin system status: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }