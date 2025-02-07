from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Tipos de tareas que los equipos pueden manejar."""
    
    # Tareas Creativas
    CONTENT_CREATION = "content_creation"
    VISUAL_DESIGN = "visual_design"
    MUSIC_PRODUCTION = "music_production"
    VIDEO_EDITING = "video_editing"
    
    # Tareas de Análisis
    DATA_ANALYSIS = "data_analysis"
    MARKET_RESEARCH = "market_research"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    TREND_ANALYSIS = "trend_analysis"
    
    # Tareas de Desarrollo
    CODE_DEVELOPMENT = "code_development"
    SYSTEM_ARCHITECTURE = "system_architecture"
    API_DESIGN = "api_design"
    DATABASE_DESIGN = "database_design"
    
    # Tareas de Negocio
    BUSINESS_STRATEGY = "business_strategy"
    SALES_PLANNING = "sales_planning"
    MARKETING_CAMPAIGN = "marketing_campaign"
    FINANCIAL_PLANNING = "financial_planning"
    
    # Tareas de Productividad
    PROJECT_MANAGEMENT = "project_management"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    WORKFLOW_AUTOMATION = "workflow_automation"
    QUALITY_ASSURANCE = "quality_assurance"

@dataclass
class TaskRequirement:
    """Requisitos específicos para una tarea."""
    skill_level: int  # 1-10
    time_estimate: float  # horas
    priority: int  # 1-5
    required_tools: List[str]
    dependencies: List[str]
    success_criteria: Dict[str, Any]

class TaskProcessor:
    """
    Procesa y gestiona tareas específicas para cada tipo de equipo.
    Define cómo cada tipo de equipo maneja diferentes tipos de tareas.
    """
    
    def __init__(self):
        self.task_handlers = {
            TaskType.CONTENT_CREATION: self._handle_content_creation,
            TaskType.VISUAL_DESIGN: self._handle_visual_design,
            TaskType.MUSIC_PRODUCTION: self._handle_music_production,
            TaskType.DATA_ANALYSIS: self._handle_data_analysis,
            TaskType.CODE_DEVELOPMENT: self._handle_code_development,
            TaskType.BUSINESS_STRATEGY: self._handle_business_strategy,
            # ... más handlers
        }
        
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []
        
    async def process_task(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesa una tarea según su tipo y las capacidades del equipo."""
        try:
            # Validar capacidades
            if not self._validate_capabilities(task_type, team_capabilities):
                raise ValueError(f"Team lacks capabilities for task type: {task_type}")
            
            # Obtener handler específico
            handler = self.task_handlers.get(task_type)
            if not handler:
                raise ValueError(f"No handler found for task type: {task_type}")
            
            # Crear registro de tarea
            task_id = self._generate_task_id()
            task_record = {
                "id": task_id,
                "type": task_type.value,
                "status": "started",
                "start_time": datetime.now().isoformat(),
                "data": task_data
            }
            self.active_tasks[task_id] = task_record
            
            # Ejecutar handler
            result = await handler(task_data, team_capabilities)
            
            # Actualizar registro
            task_record.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "result": result
            })
            
            # Mover a historial
            self.task_history.append(task_record)
            del self.active_tasks[task_id]
            
            return {
                "task_id": task_id,
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
            return {"error": str(e)}

    async def _handle_content_creation(
        self,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maneja tareas de creación de contenido."""
        try:
            content_type = task_data.get("content_type", "article")
            
            if content_type == "article":
                return await self._create_article(task_data, team_capabilities)
            elif content_type == "social_media":
                return await self._create_social_media_content(task_data, team_capabilities)
            elif content_type == "blog":
                return await self._create_blog_post(task_data, team_capabilities)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
                
        except Exception as e:
            logger.error(f"Error in content creation: {e}")
            raise

    async def _handle_visual_design(
        self,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maneja tareas de diseño visual."""
        try:
            design_type = task_data.get("design_type", "graphic")
            
            if design_type == "graphic":
                return await self._create_graphic_design(task_data, team_capabilities)
            elif design_type == "ui_ux":
                return await self._create_ui_ux_design(task_data, team_capabilities)
            elif design_type == "illustration":
                return await self._create_illustration(task_data, team_capabilities)
            else:
                raise ValueError(f"Unsupported design type: {design_type}")
                
        except Exception as e:
            logger.error(f"Error in visual design: {e}")
            raise

    async def _handle_music_production(
        self,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maneja tareas de producción musical."""
        try:
            production_type = task_data.get("production_type", "composition")
            
            if production_type == "composition":
                return await self._create_composition(task_data, team_capabilities)
            elif production_type == "mixing":
                return await self._mix_audio(task_data, team_capabilities)
            elif production_type == "mastering":
                return await self._master_audio(task_data, team_capabilities)
            else:
                raise ValueError(f"Unsupported production type: {production_type}")
                
        except Exception as e:
            logger.error(f"Error in music production: {e}")
            raise

    async def _handle_data_analysis(
        self,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maneja tareas de análisis de datos."""
        try:
            analysis_type = task_data.get("analysis_type", "exploratory")
            
            if analysis_type == "exploratory":
                return await self._perform_exploratory_analysis(task_data, team_capabilities)
            elif analysis_type == "predictive":
                return await self._perform_predictive_analysis(task_data, team_capabilities)
            elif analysis_type == "statistical":
                return await self._perform_statistical_analysis(task_data, team_capabilities)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            raise

    async def _handle_code_development(
        self,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maneja tareas de desarrollo de código."""
        try:
            dev_type = task_data.get("development_type", "feature")
            
            if dev_type == "feature":
                return await self._develop_feature(task_data, team_capabilities)
            elif dev_type == "bugfix":
                return await self._fix_bug(task_data, team_capabilities)
            elif dev_type == "refactor":
                return await self._refactor_code(task_data, team_capabilities)
            else:
                raise ValueError(f"Unsupported development type: {dev_type}")
                
        except Exception as e:
            logger.error(f"Error in code development: {e}")
            raise

    async def _handle_business_strategy(
        self,
        task_data: Dict[str, Any],
        team_capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maneja tareas de estrategia de negocio."""
        try:
            strategy_type = task_data.get("strategy_type", "market_entry")
            
            if strategy_type == "market_entry":
                return await self._develop_market_entry_strategy(task_data, team_capabilities)
            elif strategy_type == "growth":
                return await self._develop_growth_strategy(task_data, team_capabilities)
            elif strategy_type == "optimization":
                return await self._develop_optimization_strategy(task_data, team_capabilities)
            else:
                raise ValueError(f"Unsupported strategy type: {strategy_type}")
                
        except Exception as e:
            logger.error(f"Error in business strategy: {e}")
            raise

    def _validate_capabilities(
        self,
        task_type: TaskType,
        team_capabilities: Dict[str, Any]
    ) -> bool:
        """Valida que el equipo tenga las capacidades necesarias."""
        required_capabilities = self._get_required_capabilities(task_type)
        team_caps = set(team_capabilities.keys())
        return all(cap in team_caps for cap in required_capabilities)

    def _get_required_capabilities(self, task_type: TaskType) -> Set[str]:
        """Obtiene capacidades requeridas para un tipo de tarea."""
        capability_requirements = {
            TaskType.CONTENT_CREATION: {"content_writing", "editing", "research"},
            TaskType.VISUAL_DESIGN: {"graphic_design", "color_theory", "typography"},
            TaskType.MUSIC_PRODUCTION: {"audio_engineering", "composition", "mixing"},
            TaskType.DATA_ANALYSIS: {"statistics", "data_visualization", "python"},
            TaskType.CODE_DEVELOPMENT: {"programming", "testing", "version_control"},
            TaskType.BUSINESS_STRATEGY: {"market_analysis", "financial_planning", "strategy"}
        }
        return capability_requirements.get(task_type, set())

    def _generate_task_id(self) -> str:
        """Genera un ID único para una tarea."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"task_{timestamp}"

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Obtiene el estado de una tarea."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
            
        for task in self.task_history:
            if task["id"] == task_id:
                return task
                
        return {"error": "Task not found"}

    async def get_task_history(
        self,
        task_type: Optional[TaskType] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de tareas con filtros opcionales."""
        filtered_history = self.task_history
        
        if task_type:
            filtered_history = [
                task for task in filtered_history
                if task["type"] == task_type.value
            ]
            
        if status:
            filtered_history = [
                task for task in filtered_history
                if task["status"] == status
            ]
            
        return filtered_history

    async def cleanup(self) -> None:
        """Limpia recursos del procesador de tareas."""
        try:
            # Cancelar tareas activas
            for task_id, task in self.active_tasks.items():
                if task["status"] == "started":
                    task["status"] = "cancelled"
                    task["end_time"] = datetime.now().isoformat()
                    self.task_history.append(task)
            
            self.active_tasks.clear()
            logger.info("Task Processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")