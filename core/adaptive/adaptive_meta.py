import logging
import inspect
import ast
import asyncio
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import importlib
import black
from jinja2 import Template
from dataclasses import dataclass
import autopep8

logger = logging.getLogger(__name__)

@dataclass
class CodePattern:
    """Patrón de código detectado."""
    pattern_id: str
    frequency: int
    success_rate: float
    code_template: str
    use_cases: List[str]
    generated_agents: List[str]
    last_used: datetime

class AdaptiveMetaSystem:
    """
    Sistema de metaprogramación que evoluciona el código del sistema.
    
    Características:
    - Genera y modifica código basado en patrones de uso
    - Crea nuevos agentes y componentes automáticamente
    - Evoluciona la estructura del sistema
    """
    
    def __init__(self, base_dir: str = "core"):
        self.base_dir = base_dir
        self.code_patterns: Dict[str, CodePattern] = {}
        
        # Plantillas base
        self.base_templates = {
            "agent": self._load_template("agent"),
            "component": self._load_template("component"),
            "plugin": self._load_template("plugin")
        }
        
        # Registro de código generado
        self.generated_code: Dict[str, str] = {}
        
        # Métricas
        self.meta_metrics = {
            "generated_agents": 0,
            "generated_components": 0,
            "code_modifications": 0,
            "successful_compilations": 0
        }
        
        # Iniciar tarea de evolución
        self._evolution_task = asyncio.create_task(self._evolve_continuously())

    def _load_template(self, template_type: str) -> Template:
        """Carga una plantilla base."""
        templates = {
            "agent": """
class {{ class_name }}(BaseAgent):
    \"\"\"{{ docstring }}\"\"\"
    
    def __init__(self, task: str, api_key: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(task, api_key, metadata)
        {% for attr in attributes %}
        self.{{ attr.name }} = {{ attr.value }}
        {% endfor %}
    
    async def _execute(self) -> Dict[str, Any]:
        try:
            {% for step in execution_steps %}
            {{ step }}
            {% endfor %}
            
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in {{ class_name }}: {e}")
            return {"error": str(e)}
""",
            "component": """
class {{ class_name }}:
    \"\"\"{{ docstring }}\"\"\"
    
    def __init__(self):
        {% for init_line in init_lines %}
        {{ init_line }}
        {% endfor %}
    
    {% for method in methods %}
    async def {{ method.name }}({{ method.params }}) -> {{ method.return_type }}:
        \"\"\"{{ method.docstring }}\"\"\"
        {% for line in method.code %}
        {{ line }}
        {% endfor %}
    {% endfor %}
""",
            "plugin": """
from plugins.base_plugin import BasePlugin, hook

class {{ class_name }}(BasePlugin):
    \"\"\"{{ docstring }}\"\"\"
    
    def __init__(self):
        super().__init__("{{ plugin_name }}")
    
    async def setup(self, config: Dict[str, Any]) -> None:
        await super().setup(config)
        {% for setup_line in setup_lines %}
        {{ setup_line }}
        {% endfor %}
    
    {% for hook_def in hooks %}
    @hook("{{ hook_def.name }}")
    async def {{ hook_def.method }}({{ hook_def.params }}) -> {{ hook_def.return_type }}:
        \"\"\"{{ hook_def.docstring }}\"\"\"
        {% for line in hook_def.code %}
        {{ line }}
        {% endfor %}
    {% endfor %}
    
    async def cleanup(self) -> None:
        await super().cleanup()
        {% for cleanup_line in cleanup_lines %}
        {{ cleanup_line }}
        {% endfor %}
"""
        }
        return Template(templates[template_type])

    async def analyze_usage_patterns(
        self,
        usage_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analiza patrones de uso para determinar necesidades de código."""
        try:
            patterns = []
            
            # Agrupar por tipo de operación
            operation_groups = {}
            for data in usage_data:
                op_type = data["operation_type"]
                if op_type not in operation_groups:
                    operation_groups[op_type] = []
                operation_groups[op_type].append(data)
            
            # Analizar cada grupo
            for op_type, group in operation_groups.items():
                if len(group) > 10:  # Umbral mínimo
                    # Analizar características comunes
                    common_features = self._extract_common_features(group)
                    
                    if common_features["similarity_score"] > 0.7:
                        patterns.append({
                            "type": op_type,
                            "features": common_features,
                            "frequency": len(group),
                            "potential_optimization": common_features["optimization_potential"]
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []

    async def generate_new_agent(
        self,
        usage_pattern: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Genera un nuevo agente basado en patrones de uso."""
        try:
            # Determinar características del agente
            agent_spec = await self._design_agent_spec(usage_pattern, context)
            
            # Generar código
            code = self.base_templates["agent"].render(
                class_name=agent_spec["class_name"],
                docstring=agent_spec["docstring"],
                attributes=agent_spec["attributes"],
                execution_steps=agent_spec["execution_steps"]
            )
            
            # Formatear código
            formatted_code = autopep8.fix_code(
                code,
                options={"aggressive": 1}
            )
            
            # Validar código
            if await self._validate_code(formatted_code):
                # Guardar código
                file_path = f"{self.base_dir}/agents/generated/{agent_spec['class_name'].lower()}.py"
                await self._save_code(formatted_code, file_path)
                
                # Registrar patrón
                pattern = CodePattern(
                    pattern_id=f"agent_{len(self.code_patterns)}",
                    frequency=1,
                    success_rate=1.0,
                    code_template=code,
                    use_cases=[str(usage_pattern)],
                    generated_agents=[agent_spec["class_name"]],
                    last_used=datetime.now()
                )
                self.code_patterns[pattern.pattern_id] = pattern
                
                # Actualizar métricas
                self.meta_metrics["generated_agents"] += 1
                
                return file_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating agent: {e}")
            return None

    async def modify_existing_code(
        self,
        file_path: str,
        modifications: List[Dict[str, Any]]
    ) -> bool:
        """Modifica código existente basado en patrones."""
        try:
            # Leer código actual
            with open(file_path, "r") as f:
                current_code = f.read()
            
            # Parsear AST
            tree = ast.parse(current_code)
            
            # Aplicar modificaciones
            transformer = CodeTransformer(modifications)
            modified_tree = transformer.visit(tree)
            
            # Generar código modificado
            modified_code = ast.unparse(modified_tree)
            
            # Formatear
            formatted_code = black.format_str(
                modified_code,
                mode=black.FileMode()
            )
            
            # Validar
            if await self._validate_code(formatted_code):
                # Guardar
                await self._save_code(formatted_code, file_path)
                
                # Actualizar métricas
                self.meta_metrics["code_modifications"] += 1
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error modifying code: {e}")
            return False

    async def generate_plugin(
        self,
        functionality: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Genera un nuevo plugin basado en funcionalidad requerida."""
        try:
            # Diseñar plugin
            plugin_spec = await self._design_plugin_spec(functionality, context)
            
            # Generar código
            code = self.base_templates["plugin"].render(
                class_name=plugin_spec["class_name"],
                docstring=plugin_spec["docstring"],
                plugin_name=plugin_spec["name"],
                setup_lines=plugin_spec["setup"],
                hooks=plugin_spec["hooks"],
                cleanup_lines=plugin_spec["cleanup"]
            )
            
            # Formatear
            formatted_code = black.format_str(
                code,
                mode=black.FileMode()
            )
            
            # Validar
            if await self._validate_code(formatted_code):
                # Guardar
                file_path = f"{self.base_dir}/plugins/generated/{plugin_spec['name']}.py"
                await self._save_code(formatted_code, file_path)
                
                # Actualizar métricas
                self.meta_metrics["generated_components"] += 1
                
                return file_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating plugin: {e}")
            return None

    async def _validate_code(self, code: str) -> bool:
        """Valida código generado."""
        try:
            # Validar sintaxis
            ast.parse(code)
            
            # Ejecutar validaciones adicionales
            validation_errors = []
            
            # Verificar imports
            if "import *" in code:
                validation_errors.append("Wildcard imports not allowed")
            
            # Verificar naming
            if "class " in code and not any(
                line.strip().startswith("class ") and line.strip().split()[1][0].isupper()
                for line in code.split("\n")
            ):
                validation_errors.append("Invalid class naming")
            
            # Verificar métodos
            if "async def" in code and "await" not in code:
                validation_errors.append("Async function without await")
            
            # Verificar error handling
            if "try:" in code and "except " not in code:
                validation_errors.append("Try block without except")
            
            return len(validation_errors) == 0
            
        except SyntaxError:
            return False
        except Exception as e:
            logger.error(f"Error validating code: {e}")
            return False

    async def _evolve_continuously(self) -> None:
        """Proceso continuo de evolución del código."""
        try:
            while True:
                # Analizar código actual
                current_code = await self._analyze_current_code()
                
                # Detectar oportunidades de mejora
                improvements = await self._detect_improvements(current_code)
                
                # Aplicar mejoras
                for imp in improvements:
                    success = await self._apply_improvement(imp)
                    if success:
                        logger.info(f"Applied improvement: {imp['type']}")
                
                await asyncio.sleep(3600)  # Cada hora
                
        except Exception as e:
            logger.error(f"Error in continuous evolution: {e}")
        finally:
            if self._evolution_task and not self._evolution_task.cancelled():
                self._evolution_task.cancel()

    async def cleanup(self) -> None:
        """Limpia recursos del sistema."""
        try:
            # Cancelar evolución
            if self._evolution_task:
                self._evolution_task.cancel()
                try:
                    await self._evolution_task
                except asyncio.CancelledError:
                    pass
            
            # Guardar patrones
            await self._save_patterns()
            
            # Limpiar estado
            self.code_patterns.clear()
            self.generated_code.clear()
            
            logger.info("Meta system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema."""
        return {
            "active_patterns": len(self.code_patterns),
            "generated_code": {
                "agents": self.meta_metrics["generated_agents"],
                "components": self.meta_metrics["generated_components"],
                "modifications": self.meta_metrics["code_modifications"]
            },
            "success_rate": self.meta_metrics["successful_compilations"] / 
                          (self.meta_metrics["generated_agents"] + 
                           self.meta_metrics["code_modifications"]) 
                          if (self.meta_metrics["generated_agents"] + 
                              self.meta_metrics["code_modifications"]) > 0 else 0.0
        }