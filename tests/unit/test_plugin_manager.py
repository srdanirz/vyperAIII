import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import ast

from plugins.plugin_manager import PluginManager, CodeTransformer
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
async def test_code_transformer():
    # Crear transformador con modificaciones de prueba
    modifications = [
        {
            "target": "function",
            "name": "test_function",
            "new_name": "modified_function",
            "async": True,
            "decorators": ["test_decorator"]
        },
        {
            "target": "class",
            "name": "TestClass",
            "new_name": "ModifiedClass",
            "bases": ["NewBase"]
        }
    ]
    
    transformer = CodeTransformer(modifications)
    
    # Código de prueba
    test_code = """
def test_function():
    return "test"

class TestClass:
    def method(self):
        pass
    """
    
    # Parsear y transformar
    tree = ast.parse(test_code)
    modified_tree = transformer.visit(tree)
    
    # Verificar transformaciones
    for node in ast.walk(modified_tree):
        if isinstance(node, ast.AsyncFunctionDef):
            assert node.name == "modified_function"
            assert any(d.id == "test_decorator" for d in node.decorator_list)
        elif isinstance(node, ast.ClassDef):
            assert node.name == "ModifiedClass"
            assert len(node.bases) == 1
            assert isinstance(node.bases[0], ast.Name)
            assert node.bases[0].id == "NewBase"

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
        def __init__(self):
            super().__init__("slow_plugin")
            
        async def setup(self, config):
            await super().setup(config)
        
        @hook("slow_hook")
        async def slow_method(self):
            await asyncio.sleep(0.2)
            return "Done"
    
    plugin = SlowPlugin()
    await plugin.setup({})
    manager.plugins["slow_plugin"] = plugin
    manager._register_hooks(plugin)
    
    # El hook debería causar timeout
    results = await manager.execute_hook("slow_hook")
    assert len(results) == 0
    
    await manager.cleanup()

@pytest.mark.asyncio
async def test_transform_existing_plugin(plugin_manager):
    # Código de plugin original
    original_code = """
class MyPlugin(BasePlugin):
    def __init__(self):
        super().__init__("my_plugin")
    
    def sync_method(self):
        return "sync"
    """
    
    # Crear transformaciones
    modifications = [
        {
            "target": "function",
            "name": "sync_method",
            "async": True,
            "decorators": ["hook('test')"]
        }
    ]
    
    # Aplicar transformación
    transformer = CodeTransformer(modifications)
    tree = ast.parse(original_code)
    modified_tree = transformer.visit(tree)
    
    # Verificar cambios
    method_found = False
    for node in ast.walk(modified_tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "sync_method":
            method_found = True
            assert len(node.decorator_list) == 1
            
    assert method_found

@pytest.mark.asyncio
async def test_concurrent_hook_execution(plugin_manager):
    # Preparar múltiples hooks
    hooks = []
    for i in range(5):
        plugin = TestPlugin()
        await plugin.setup({})
        plugin_manager.plugins[f"test_plugin_{i}"] = plugin
        plugin_manager._register_hooks(plugin)
        hooks.append(plugin.test_hook_method)
    
    # Ejecutar hooks concurrentemente
    tasks = [
        plugin_manager.execute_hook("test_hook", f"data_{i}")
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verificar resultados
    assert len(results) == 5
    assert all(len(r) > 0 for r in results)
    assert all("Processed:" in r[0] for r in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=plugins.plugin_manager"])