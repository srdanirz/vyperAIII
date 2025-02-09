import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from plugins.plugin_manager import PluginManager
from plugins.base_plugin import BasePlugin, hook

# Plugin de prueba
class TestPlugin(BasePlugin):
    def __init__(self):
        super().__init__("test_plugin")
        
    async def setup(self, config):
        await super().setup(config)
        
    @hook("test_hook")
    async def test_hook_method(self, data):
        return f"Processed: {data}"
        
    async def execute(self, *args, **kwargs):
        await super().execute(*args, **kwargs)
        return "Executed"
        
    async def cleanup(self):
        await super().cleanup()

@pytest.fixture
async def plugin_manager():
    manager = PluginManager()
    await manager.initialize()
    yield manager
    await manager.cleanup()

@pytest.mark.asyncio
async def test_plugin_loading(plugin_manager):
    # Simular módulo de plugin
    mock_module = Mock()
    mock_module.TestPlugin = TestPlugin
    
    with patch('importlib.import_module', return_value=mock_module):
        # Cargar plugin
        success = await plugin_manager.load_plugin("test_plugin")
        assert success
        
        # Verificar que el plugin está cargado
        assert "test_plugin" in plugin_manager.plugins
        assert plugin_manager.metrics["loaded_plugins"] == 1

@pytest.mark.asyncio
async def test_hook_execution(plugin_manager):
    # Cargar plugin de prueba
    plugin = TestPlugin()
    await plugin.setup({})
    plugin_manager.plugins["test_plugin"] = plugin
    plugin_manager._register_hooks(plugin)
    
    # Ejecutar hook
    results = await plugin_manager.execute_hook("test_hook", "test_data")
    
    assert len(results) == 1
    assert results[0] == "Processed: test_data"

@pytest.mark.asyncio
async def test_plugin_reload(plugin_manager):
    # Cargar plugin inicial
    mock_module = Mock()
    mock_module.TestPlugin = TestPlugin
    
    with patch('importlib.import_module', return_value=mock_module):
        await plugin_manager.load_plugin("test_plugin")
        
        # Recargar plugin
        success = await plugin_manager.reload_plugin("test_plugin")
        assert success
        
        # Verificar que el plugin sigue cargado
        assert "test_plugin" in plugin_manager.plugins

@pytest.mark.asyncio
async def test_plugin_dependencies(plugin_manager):
    # Configurar dependencias
    plugin_manager.plugin_dependencies = {
        "plugin2": ["plugin1"]
    }
    
    # Cargar plugins en orden incorrecto
    mock_module = Mock()
    mock_module.TestPlugin = TestPlugin
    
    with patch('importlib.import_module', return_value=mock_module):
        # Intentar cargar plugin2 primero (debería fallar)
        success = await plugin_manager.load_plugin("plugin2")
        assert not success
        
        # Cargar plugin1
        success = await plugin_manager.load_plugin("plugin1")
        assert success
        
        # Ahora cargar plugin2
        success = await plugin_manager.load_plugin("plugin2")
        assert success

@pytest.mark.asyncio
async def test_plugin_cleanup(plugin_manager):
    # Cargar plugin
    plugin = TestPlugin()
    await plugin.setup({})
    plugin_manager.plugins["test_plugin"] = plugin
    plugin_manager._register_hooks(plugin)
    
    # Descargar plugin
    success = await plugin_manager.unload_plugin("test_plugin")
    assert success
    
    # Verificar que se limpió correctamente
    assert "test_plugin" not in plugin_manager.plugins
    assert plugin_manager.metrics["loaded_plugins"] == 0
    assert "test_hook" not in plugin_manager.hooks

@pytest.mark.asyncio
async def test_hook_timeout():
    manager = PluginManager()
    manager.plugin_configs["hook_timeout"] = 0.1
    
    # Plugin con hook que tarda demasiado
    class SlowPlugin(BasePlugin):
        async def setup(self, config):
            await super().setup(config)
        
        @hook("slow_hook")
        async def slow_method(self):
            await asyncio.sleep(0.2)
            return "Done"
    
    plugin = SlowPlugin("slow_plugin")
    await plugin.setup({})
    manager.plugins["slow_plugin"] = plugin
    manager._register_hooks(plugin)
    
    # El hook debería causar timeout
    results = await manager.execute_hook("slow_hook")
    assert len(results) == 0
    
    await manager.cleanup()

@pytest.mark.asyncio
async def test_metrics_collection(plugin_manager):
    # Cargar plugin y ejecutar algunas operaciones
    plugin = TestPlugin()
    await plugin.setup({})
    plugin_manager.plugins["test_plugin"] = plugin
    plugin_manager._register_hooks(plugin)
    
    await plugin_manager.execute_hook("test_hook", "data1")
    await plugin_manager.execute_hook("test_hook", "data2")
    
    # Verificar métricas
    metrics = await plugin_manager.get_all_metrics()
    
    assert metrics["total_plugins"] == 1
    assert metrics["active_hooks"] >= 1
    assert "test_hook" in metrics["performance"]
    assert metrics["performance"]["test_hook"]["total_executions"] == 2

@pytest.mark.asyncio
async def test_health_check(plugin_manager):
    # Estado saludable
    plugin = TestPlugin()
    await plugin.setup({})
    plugin_manager.plugins["test_plugin"] = plugin
    plugin_manager._register_hooks(plugin)
    
    assert plugin_manager._check_health()
    
    # Estado degradado (muchos errores)
    plugin_manager.metrics["failed_loads"] = 10
    assert not plugin_manager._check_health()

@pytest.mark.asyncio
async def test_status_report(plugin_manager):
    # Cargar plugin
    plugin = TestPlugin()
    await plugin.setup({})
    plugin_manager.plugins["test_plugin"] = plugin
    plugin_manager._register_hooks(plugin)
    
    # Obtener estado
    status = await plugin_manager.get_status()
    
    assert "active_plugins" in status
    assert "hooks" in status
    assert "metrics" in status
    assert "config" in status
    assert "health" in status
    assert "timestamp" in status