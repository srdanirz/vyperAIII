from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging

from capabilities.task_processor import TaskType, TaskProcessor

logger = logging.getLogger(__name__)

class TeamSpecialization(Enum):
    """Especializaciones disponibles para equipos."""
    
    # Equipos Creativos
    MUSIC_PRODUCTION = "music_production_team"
    VISUAL_ARTS = "visual_arts_team"
    CONTENT_CREATION = "content_creation_team"
    VIDEO_PRODUCTION = "video_production_team"
    
    # Equipos de Negocios
    STARTUP = "startup_team"
    ENTERPRISE = "enterprise_team"
    MARKETING = "marketing_team"
    SALES = "sales_team"
    
    # Equipos Técnicos
    SOFTWARE_DEV = "software_development_team"
    DATA_SCIENCE = "data_science_team"
    DEVOPS = "devops_team"
    CYBERSECURITY = "cybersecurity_team"
    
    # Equipos de Investigación
    MARKET_RESEARCH = "market_research_team"
    SCIENTIFIC_RESEARCH = "scientific_research_team"
    TREND_ANALYSIS = "trend_analysis_team"
    
    # Equipos de Proyecto
    PROJECT_MANAGEMENT = "project_management_team"
    AGILE_DEVELOPMENT = "agile_development_team"
    QUALITY_ASSURANCE = "quality_assurance_team"

@dataclass
class TeamCapabilities:
    """Capacidades y habilidades específicas de un equipo."""
    primary_skills: Set[str]
    secondary_skills: Set[str]
    tools: Set[str]
    experience_level: int  # 1-10
    specializations: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class SpecializedTeamFactory:
    """
    Fábrica para crear equipos especializados con capacidades predefinidas.
    """
    
    def __init__(self):
        self.task_processor = TaskProcessor()
        self._load_team_templates()

    def _load_team_templates(self):
        """Carga plantillas predefinidas para diferentes tipos de equipos."""
        self.team_templates = {
            # EQUIPOS CREATIVOS
            
            # Equipo de Producción Musical
            TeamSpecialization.MUSIC_PRODUCTION: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "audio_engineering",
                        "music_composition",
                        "sound_design",
                        "mixing",
                        "mastering",
                        "music_theory",
                        "daw_expertise",
                        "live_recording",
                        "midi_programming",
                        "vocal_production"
                    },
                    secondary_skills={
                        "project_management",
                        "client_communication",
                        "acoustics",
                        "audio_branding",
                        "copyright_management",
                        "music_business",
                        "live_performance",
                        "studio_management"
                    },
                    tools={
                        "pro_tools",
                        "ableton_live",
                        "logic_pro",
                        "fl_studio",
                        "waves_plugins",
                        "native_instruments",
                        "universal_audio",
                        "izotope_suite",
                        "midi_controllers",
                        "analog_synthesizers"
                    },
                    experience_level=8,
                    specializations=[
                        "electronic_music",
                        "orchestral_composition",
                        "sound_effects",
                        "voice_over_production",
                        "film_scoring",
                        "game_audio",
                        "podcast_production",
                        "live_sound"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.MUSIC_PRODUCTION,
                    TaskType.CONTENT_CREATION,
                    TaskType.AUDIO_ENGINEERING
                ]
            },

            # Equipo de Artes Visuales
            TeamSpecialization.VISUAL_ARTS: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "graphic_design",
                        "illustration",
                        "3d_modeling",
                        "animation",
                        "photo_editing",
                        "color_theory",
                        "typography",
                        "composition",
                        "motion_graphics",
                        "art_direction"
                    },
                    secondary_skills={
                        "project_planning",
                        "client_presentation",
                        "brand_development",
                        "user_experience",
                        "printing_knowledge",
                        "video_editing",
                        "web_design",
                        "social_media_design"
                    },
                    tools={
                        "adobe_creative_suite",
                        "sketch",
                        "figma",
                        "cinema_4d",
                        "blender",
                        "procreate",
                        "wacom_tablets",
                        "after_effects",
                        "substance_painter",
                        "zbrush"
                    },
                    experience_level=9,
                    specializations=[
                        "ui_design",
                        "character_design",
                        "environmental_art",
                        "concept_art",
                        "brand_identity",
                        "packaging_design",
                        "editorial_design",
                        "motion_design"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.VISUAL_DESIGN,
                    TaskType.CONTENT_CREATION,
                    TaskType.VIDEO_EDITING
                ]
            },

            # EQUIPOS TÉCNICOS
            
            # Equipo de Desarrollo de Software
            TeamSpecialization.SOFTWARE_DEV: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "software_architecture",
                        "backend_development",
                        "frontend_development",
                        "database_design",
                        "api_development",
                        "system_design",
                        "security_implementation",
                        "performance_optimization",
                        "cloud_architecture",
                        "mobile_development"
                    },
                    secondary_skills={
                        "devops",
                        "testing",
                        "documentation",
                        "agile_methodologies",
                        "ci_cd",
                        "code_review",
                        "technical_writing",
                        "mentoring",
                        "requirements_analysis",
                        "ux_collaboration"
                    },
                    tools={
                        "git",
                        "docker",
                        "kubernetes",
                        "jenkins",
                        "jira",
                        "aws_suite",
                        "azure_services",
                        "terraform",
                        "prometheus",
                        "grafana"
                    },
                    experience_level=9,
                    specializations=[
                        "web_applications",
                        "mobile_development",
                        "cloud_architecture",
                        "microservices",
                        "ai_ml_systems",
                        "iot_platforms",
                        "blockchain_development",
                        "real_time_systems"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.CODE_DEVELOPMENT,
                    TaskType.SYSTEM_ARCHITECTURE,
                    TaskType.API_DESIGN,
                    TaskType.DATABASE_DESIGN
                ]
            },

            # EQUIPOS DE NEGOCIO
            
            # Equipo Startup
            TeamSpecialization.STARTUP: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "business_strategy",
                        "market_analysis",
                        "product_development",
                        "growth_hacking",
                        "pitch_creation",
                        "mvp_development",
                        "customer_development",
                        "lean_methodology",
                        "fundraising",
                        "business_modeling"
                    },
                    secondary_skills={
                        "financial_planning",
                        "team_building",
                        "networking",
                        "digital_marketing",
                        "user_research",
                        "data_analysis",
                        "sales_strategy",
                        "operations_management",
                        "legal_compliance",
                        "investor_relations"
                    },
                    tools={
                        "lean_canvas",
                        "analytics_tools",
                        "crm_systems",
                        "project_management_software",
                        "financial_modeling_tools",
                        "market_research_platforms",
                        "collaboration_tools",
                        "product_analytics",
                        "customer_feedback_systems",
                        "pitch_deck_software"
                    },
                    experience_level=8,
                    specializations=[
                        "saas_startups",
                        "marketplace_platforms",
                        "mobile_apps",
                        "b2b_solutions",
                        "fintech",
                        "healthtech",
                        "ai_startups",
                        "social_impact_ventures"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.BUSINESS_STRATEGY,
                    TaskType.MARKET_RESEARCH,
                    TaskType.PROJECT_MANAGEMENT,
                    TaskType.FINANCIAL_PLANNING
                ]
            },

            # Equipo de Marketing Digital
            TeamSpecialization.MARKETING: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "digital_marketing_strategy",
                        "content_marketing",
                        "seo_optimization",
                        "social_media_marketing",
                        "email_marketing",
                        "ppc_advertising",
                        "marketing_analytics",
                        "conversion_optimization",
                        "brand_development",
                        "marketing_automation"
                    },
                    secondary_skills={
                        "copywriting",
                        "web_analytics",
                        "customer_journey_mapping",
                        "affiliate_marketing",
                        "influencer_marketing",
                        "video_marketing",
                        "market_research",
                        "a_b_testing",
                        "crm_management",
                        "marketing_design"
                    },
                    tools={
                        "google_analytics",
                        "google_ads",
                        "facebook_ads_manager",
                        "hubspot",
                        "mailchimp",
                        "semrush",
                        "ahrefs",
                        "hotjar",
                        "buffer",
                        "adobe_analytics"
                    },
                    experience_level=8,
                    specializations=[
                        "b2b_marketing",
                        "b2c_marketing",
                        "saas_marketing",
                        "ecommerce_marketing",
                        "content_strategy",
                        "performance_marketing",
                        "brand_marketing",
                        "growth_marketing"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.MARKETING_CAMPAIGN,
                    TaskType.CONTENT_CREATION,
                    TaskType.MARKET_RESEARCH,
                    TaskType.DATA_ANALYSIS
                ]
            },
            # Equipos Creativos
            TeamSpecialization.MUSIC_PRODUCTION: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "audio_engineering",
                        "music_composition",
                        "sound_design",
                        "mixing",
                        "mastering"
                    },
                    secondary_skills={
                        "project_management",
                        "client_communication",
                        "music_theory"
                    },
                    tools={
                        "pro_tools",
                        "ableton_live",
                        "logic_pro",
                        "waves_plugins",
                        "midi_controllers"
                    },
                    experience_level=8,
                    specializations=[
                        "electronic_music",
                        "orchestral_composition",
                        "sound_effects",
                        "voice_over_production"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.MUSIC_PRODUCTION,
                    TaskType.CONTENT_CREATION
                ]
            },
            
            TeamSpecialization.SOFTWARE_DEV: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "software_architecture",
                        "backend_development",
                        "frontend_development",
                        "database_design",
                        "api_development"
                    },
                    secondary_skills={
                        "devops",
                        "testing",
                        "documentation",
                        "agile_methodologies"
                    },
                    tools={
                        "git",
                        "docker",
                        "kubernetes",
                        "jenkins",
                        "jira"
                    },
                    experience_level=9,
                    specializations=[
                        "web_applications",
                        "mobile_development",
                        "cloud_architecture",
                        "microservices"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.CODE_DEVELOPMENT,
                    TaskType.SYSTEM_ARCHITECTURE
                ]
            },
            
            TeamSpecialization.STARTUP: {
                "capabilities": TeamCapabilities(
                    primary_skills={
                        "business_strategy",
                        "market_analysis",
                        "product_development",
                        "growth_hacking",
                        "pitch_creation"
                    },
                    secondary_skills={
                        "financial_planning",
                        "team_building",
                        "networking",
                        "digital_marketing"
                    },
                    tools={
                        "lean_canvas",
                        "analytics_tools",
                        "crm_systems",
                        "project_management_software"
                    },
                    experience_level=8,
                    specializations=[
                        "saas_startups",
                        "marketplace_platforms",
                        "mobile_apps",
                        "b2b_solutions"
                    ]
                ),
                "preferred_tasks": [
                    TaskType.BUSINESS_STRATEGY,
                    TaskType.MARKET_RESEARCH
                ]
            }
            # ... más templates de equipos
        }

    async def create_specialized_team(
        self,
        specialization: TeamSpecialization,
        custom_capabilities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crea un equipo especializado con capacidades específicas.
        
        Args:
            specialization: Tipo de especialización del equipo
            custom_capabilities: Capacidades personalizadas opcionales
        """
        try:
            template = self.team_templates.get(specialization)
            if not template:
                raise ValueError(f"No template found for specialization: {specialization}")
            
            # Combinar template con capacidades personalizadas
            capabilities = template["capabilities"]
            if custom_capabilities:
                capabilities = self._merge_capabilities(
                    capabilities,
                    custom_capabilities
                )
            
            # Crear equipo especializado
            team = {
                "id": self._generate_team_id(specialization),
                "specialization": specialization.value,
                "capabilities": capabilities,
                "preferred_tasks": template["preferred_tasks"],
                "created_at": datetime.now().isoformat(),
                "status": "ready",
                "performance_history": []
            }
            
            return team
            
        except Exception as e:
            logger.error(f"Error creating specialized team: {e}")
            raise

    def _merge_capabilities(
        self,
        base_capabilities: TeamCapabilities,
        custom_capabilities: Dict[str, Any]
    ) -> TeamCapabilities:
        """Combina capacidades base con personalizadas."""
        return TeamCapabilities(
            primary_skills=base_capabilities.primary_skills.union(
                custom_capabilities.get("primary_skills", set())
            ),
            secondary_skills=base_capabilities.secondary_skills.union(
                custom_capabilities.get("secondary_skills", set())
            ),
            tools=base_capabilities.tools.union(
                custom_capabilities.get("tools", set())
            ),
            experience_level=custom_capabilities.get(
                "experience_level",
                base_capabilities.experience_level
            ),
            specializations=list(set(base_capabilities.specializations + 
                custom_capabilities.get("specializations", [])))
        )

    def _generate_team_id(self, specialization: TeamSpecialization) -> str:
        """Genera un ID único para el equipo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{specialization.value}_{timestamp}"

    async def get_team_capabilities(
        self,
        specialization: TeamSpecialization
    ) -> Dict[str, Any]:
        """Obtiene las capacidades de un tipo de equipo."""
        template = self.team_templates.get(specialization)
        if not template:
            raise ValueError(f"No template found for specialization: {specialization}")
        return template

    async def validate_team_for_task(
        self,
        team: Dict[str, Any],
        task_type: TaskType
    ) -> bool:
        """Valida si un equipo puede manejar un tipo de tarea."""
        return task_type in team["preferred_tasks"]

class SpecializedTeamManager:
    """
    Gestiona equipos especializados y sus interacciones.
    """
    
    def __init__(self):
        self.factory = SpecializedTeamFactory()
        self.active_teams: Dict[str, Dict[str, Any]] = {}
        self.team_assignments: Dict[str, List[str]] = {}  # team_id -> task_ids

    async def create_team(
        self,
        specialization: TeamSpecialization,
        custom_capabilities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Crea y registra un nuevo equipo especializado."""
        team = await self.factory.create_specialized_team(
            specialization,
            custom_capabilities
        )
        self.active_teams[team["id"]] = team
        self.team_assignments[team["id"]] = []
        return team

    async def assign_task(
        self,
        team_id: str,
        task_type: TaskType,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Asigna una tarea a un equipo especializado."""
        try:
            team = self.active_teams.get(team_id)
            if not team:
                raise ValueError(f"Team not found: {team_id}")
                
            # Validar capacidad
            if not await self.factory.validate_team_for_task(team, task_type):
                raise ValueError(f"Team cannot handle task type: {task_type}")
            
            # Procesar tarea
            result = await self.factory.task_processor.process_task(
                task_type,
                task_data,
                team["capabilities"]
            )
            
            # Registrar asignación
            self.team_assignments[team_id].append(result["task_id"])
            
            # Actualizar métricas
            self._update_team_metrics(team_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            raise

    def _update_team_metrics(
        self,
        team_id: str,
        task_result: Dict[str, Any]
    ) -> None:
        """Actualiza métricas de rendimiento del equipo."""
        try:
            team = self.active_teams[team_id]
            
            # Actualizar historial
            team["performance_history"].append({
                "task_id": task_result["task_id"],
                "status": task_result["status"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Actualizar métricas agregadas
            metrics = team["capabilities"].performance_metrics
            metrics["total_tasks"] = metrics.get("total_tasks", 0) + 1
            
            if task_result["status"] == "success":
                metrics["successful_tasks"] = metrics.get("successful_tasks", 0) + 1
            else:
                metrics["failed_tasks"] = metrics.get("failed_tasks", 0) + 1
                
            metrics["success_rate"] = (
                metrics.get("successful_tasks", 0) / metrics["total_tasks"]
            )
            
        except Exception as e:
            logger.error(f"Error updating team metrics: {e}")

    async def get_team_status(self, team_id: str) -> Dict[str, Any]:
        """Obtiene estado actual de un equipo."""
        team = self.active_teams.get(team_id)
        if not team:
            raise ValueError(f"Team not found: {team_id}")
            
        return {
            "id": team["id"],
            "specialization": team["specialization"],
            "status": team["status"],
            "active_tasks": len(self.team_assignments[team_id]),
            "performance_metrics": team["capabilities"].performance_metrics,
            "last_updated": datetime.now().isoformat()
        }

    async def cleanup(self) -> None:
        """Limpia recursos del manager."""
        try:
            self.active_teams.clear()
            self.team_assignments.clear()
            logger.info("Specialized Team Manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")