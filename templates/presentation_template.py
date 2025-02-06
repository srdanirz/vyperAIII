# templates/presentation_template.py

from typing import Dict, Any, List
from .base_template import BaseTemplate
import logging

logger = logging.getLogger(__name__)

class PresentationTemplate(BaseTemplate):
    """Template especializado para presentaciones"""

    REQUIRED_SECTIONS = [
        "title_slide",
        "content_slides",
        "closing_slide"
    ]

    SLIDE_LAYOUTS = {
        "title": {
            "elements": ["title", "subtitle", "date", "author"],
            "proportions": {"title": 0.5, "subtitle": 0.3, "other": 0.2}
        },
        "content": {
            "elements": ["title", "content", "footer"],
            "proportions": {"title": 0.2, "content": 0.7, "footer": 0.1}
        },
        "two_column": {
            "elements": ["title", "left_content", "right_content", "footer"],
            "proportions": {"title": 0.2, "columns": 0.7, "footer": 0.1}
        },
        "image_with_text": {
            "elements": ["title", "image", "description", "footer"],
            "proportions": {"title": 0.2, "image": 0.5, "description": 0.2, "footer": 0.1}
        }
    }

    def __init__(self, template_name: str, config: Dict[str, Any]):
        super().__init__(template_name, config)
        self.theme = config["templates"]["powerpoint"]["themes"].get(
            template_name, 
            config["templates"]["powerpoint"]["themes"]["professional"]
        )
        self.layouts = self.SLIDE_LAYOUTS

    async def apply(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica el template a la presentación"""
        try:
            # Validar contenido
            if not self.validate_content(content):
                raise ValueError("Invalid content structure")

            # Aplicar tema y estilos
            styled_content = self.apply_theme(content)

            # Procesar cada tipo de slide
            processed_content = {
                "metadata": {
                    **self.metadata,
                    "theme_used": self.template_name
                },
                "title_slide": self.process_title_slide(styled_content["title_slide"]),
                "content_slides": [
                    self.process_content_slide(slide)
                    for slide in styled_content["content_slides"]
                ],
                "closing_slide": self.process_closing_slide(styled_content["closing_slide"])
            }

            return processed_content

        except Exception as e:
            logger.error(f"Error applying presentation template: {e}")
            raise

    def validate(self) -> bool:
        """Valida que el template tenga todos los componentes necesarios"""
        try:
            # Validar tema
            required_theme_elements = ["colors", "fonts", "layouts"]
            if not all(elem in self.theme for elem in required_theme_elements):
                return False

            # Validar layouts
            for layout_name, layout_config in self.layouts.items():
                if not all(key in layout_config for key in ["elements", "proportions"]):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating template: {e}")
            return False

    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Valida la estructura del contenido"""
        try:
            # Verificar secciones requeridas
            if not all(section in content for section in self.REQUIRED_SECTIONS):
                return False

            # Validar estructura de cada slide
            if not self._validate_title_slide(content["title_slide"]):
                return False

            for slide in content["content_slides"]:
                if not self._validate_content_slide(slide):
                    return False

            if not self._validate_closing_slide(content["closing_slide"]):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating content: {e}")
            return False

    def apply_theme(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica el tema al contenido"""
        try:
            themed_content = content.copy()
            
            # Aplicar colores y fuentes a cada sección
            themed_content["title_slide"] = self._apply_theme_to_slide(
                themed_content["title_slide"],
                "title"
            )
            
            themed_content["content_slides"] = [
                self._apply_theme_to_slide(slide, slide.get("layout", "content"))
                for slide in themed_content["content_slides"]
            ]
            
            themed_content["closing_slide"] = self._apply_theme_to_slide(
                themed_content["closing_slide"],
                "content"
            )

            return themed_content

        except Exception as e:
            logger.error(f"Error applying theme: {e}")
            raise

    def _apply_theme_to_slide(self, slide: Dict[str, Any], layout_type: str) -> Dict[str, Any]:
        """Aplica el tema a un slide individual"""
        themed_slide = slide.copy()
        layout_config = self.layouts[layout_type]

        # Aplicar colores
        themed_slide["background_color"] = self.theme["colors"]["primary"][0]
        themed_slide["text_color"] = self.theme["colors"]["secondary"][0]

        # Aplicar fuentes
        if "title" in themed_slide:
            themed_slide["title_font"] = self.theme["fonts"]["title"]
        if "content" in themed_slide:
            themed_slide["content_font"] = self.theme["fonts"]["body"]

        # Aplicar proporciones del layout
        themed_slide["layout"] = {
            "type": layout_type,
            "proportions": layout_config["proportions"]
        }

        return themed_slide

    def _validate_title_slide(self, slide: Dict[str, Any]) -> bool:
        """Valida la estructura del slide de título"""
        required_elements = ["title"]
        return all(elem in slide for elem in required_elements)

    def _validate_content_slide(self, slide: Dict[str, Any]) -> bool:
        """Valida la estructura de un slide de contenido"""
        required_elements = ["title", "content"]
        return all(elem in slide for elem in required_elements)

    def _validate_closing_slide(self, slide: Dict[str, Any]) -> bool:
        """Valida la estructura del slide de cierre"""
        required_elements = ["title", "content"]
        return all(elem in slide for elem in required_elements)

    def process_title_slide(self, slide: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el slide de título"""
        return {
            **slide,
            "layout": "title",
            "style": {
                "title_size": self.theme["fonts"]["title"]["size"],
                "subtitle_size": self.theme["fonts"]["heading"]["size"]
            }
        }

    def process_content_slide(self, slide: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa un slide de contenido"""
        layout_type = slide.get("layout", "content")
        return {
            **slide,
            "layout": layout_type,
            "style": {
                "title_size": self.theme["fonts"]["heading"]["size"],
                "content_size": self.theme["fonts"]["body"]["size"]
            }
        }

    def process_closing_slide(self, slide: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el slide de cierre"""
        return {
            **slide,
            "layout": "content",
            "style": {
                "title_size": self.theme["fonts"]["heading"]["size"],
                "content_size": self.theme["fonts"]["body"]["size"]
            }
        }