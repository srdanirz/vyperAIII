# templates/document_template.py

from typing import Dict, Any, List, Optional
from .base_template import BaseTemplate
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentTemplate(BaseTemplate):
    """Template especializado para documentos (Word/PDF)"""

    REQUIRED_SECTIONS = [
        "header",
        "title",
        "executive_summary",
        "content_sections",
        "footer"
    ]

    SECTION_TYPES = {
        "header": {
            "elements": ["title", "date", "author", "logo"],
            "style": "header"
        },
        "title": {
            "elements": ["main_title", "subtitle"],
            "style": "title"
        },
        "executive_summary": {
            "elements": ["title", "content"],
            "style": "summary"
        },
        "content": {
            "elements": ["title", "paragraphs", "lists", "tables", "images"],
            "style": "content"
        },
        "footer": {
            "elements": ["page_number", "copyright", "contact"],
            "style": "footer"
        }
    }

    def __init__(self, template_name: str, config: Dict[str, Any]):
        super().__init__(template_name, config)
        self.style_config = config["templates"]["document"]["styles"].get(
            template_name,
            config["templates"]["document"]["styles"]["professional"]
        )
        self.section_types = self.SECTION_TYPES

    async def apply(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica el template al documento"""
        try:
            # Validar contenido
            if not self.validate_content(content):
                raise ValueError("Invalid document content structure")

            # Aplicar estilos
            styled_content = self.apply_styles(content)

            # Procesar cada sección
            processed_content = {
                "metadata": {
                    **self.metadata,
                    "template_used": self.template_name,
                    "processing_date": datetime.now().isoformat()
                },
                "document_properties": self._build_document_properties(styled_content),
                "sections": {
                    "header": self.process_header(styled_content["header"]),
                    "title": self.process_title_section(styled_content["title"]),
                    "executive_summary": self.process_summary(styled_content["executive_summary"]),
                    "content_sections": [
                        self.process_content_section(section)
                        for section in styled_content["content_sections"]
                    ],
                    "footer": self.process_footer(styled_content["footer"])
                }
            }

            return processed_content

        except Exception as e:
            logger.error(f"Error applying document template: {e}")
            raise

    def validate(self) -> bool:
        """Valida que el template tenga todos los componentes necesarios"""
        try:
            # Validar estilos requeridos
            required_styles = ["title", "heading1", "heading2", "body"]
            if not all(style in self.style_config["fonts"] for style in required_styles):
                return False

            # Validar configuración de márgenes
            required_margins = ["top", "bottom", "left", "right"]
            if not all(margin in self.style_config["margins"] for margin in required_margins):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating document template: {e}")
            return False

    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Valida la estructura del contenido del documento"""
        try:
            # Verificar secciones requeridas
            if not all(section in content for section in self.REQUIRED_SECTIONS):
                return False

            # Validar estructura de cada sección
            validations = {
                "header": self._validate_header,
                "title": self._validate_title_section,
                "executive_summary": self._validate_summary,
                "content_sections": self._validate_content_sections,
                "footer": self._validate_footer
            }

            for section_name, validator in validations.items():
                if not validator(content[section_name]):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating document content: {e}")
            return False

    def apply_styles(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica los estilos al contenido del documento"""
        try:
            styled_content = content.copy()
            
            # Aplicar estilos a cada sección
            styled_content["header"] = self._apply_style_to_section(
                styled_content["header"],
                "header"
            )
            
            styled_content["title"] = self._apply_style_to_section(
                styled_content["title"],
                "title"
            )
            
            styled_content["executive_summary"] = self._apply_style_to_section(
                styled_content["executive_summary"],
                "summary"
            )
            
            styled_content["content_sections"] = [
                self._apply_style_to_section(section, "content")
                for section in styled_content["content_sections"]
            ]
            
            styled_content["footer"] = self._apply_style_to_section(
                styled_content["footer"],
                "footer"
            )

            return styled_content

        except Exception as e:
            logger.error(f"Error applying document styles: {e}")
            raise

    def _apply_style_to_section(self, section: Dict[str, Any], section_type: str) -> Dict[str, Any]:
        """Aplica estilos a una sección específica"""
        styled_section = section.copy()
        section_config = self.section_types[section_type]

        # Aplicar fuentes
        if "title" in styled_section:
            styled_section["title_style"] = self.style_config["fonts"]["title"]
        if "content" in styled_section:
            styled_section["content_style"] = self.style_config["fonts"]["body"]

        # Aplicar colores
        styled_section["text_color"] = self.style_config["colors"].get(
            section_type,
            self.style_config["colors"]["body"]
        )

        # Aplicar estilo de sección
        styled_section["section_style"] = section_config["style"]

        return styled_section

    def _build_document_properties(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Construye las propiedades del documento"""
        return {
            "title": content["title"].get("main_title", "Untitled Document"),
            "author": content["header"].get("author", "Unknown"),
            "created_date": datetime.now().isoformat(),
            "margins": self.style_config["margins"],
            "language": content.get("language", "es"),
            "page_size": content.get("page_size", "A4")
        }

    def process_header(self, header: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa la sección de encabezado"""
        return {
            **header,
            "style": {
                "font": self.style_config["fonts"]["heading1"],
                "alignment": "center",
                "spacing_after": 20
            }
        }

    def process_title_section(self, title: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa la sección de título"""
        return {
            **title,
            "style": {
                "font": self.style_config["fonts"]["title"],
                "alignment": "center",
                "spacing_before": 40,
                "spacing_after": 30
            }
        }

    def process_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el resumen ejecutivo"""
        return {
            **summary,
            "style": {
                "title_font": self.style_config["fonts"]["heading2"],
                "content_font": self.style_config["fonts"]["body"],
                "spacing_before": 20,
                "spacing_after": 20
            }
        }

    def process_content_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa una sección de contenido"""
        return {
            **section,
            "style": {
                "title_font": self.style_config["fonts"]["heading2"],
                "content_font": self.style_config["fonts"]["body"],
                "spacing_before": 15,
                "spacing_after": 15
            }
        }

    def process_footer(self, footer: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa la sección de pie de página"""
        return {
            **footer,
            "style": {
                "font": self.style_config["fonts"]["body"],
                "alignment": "center",
                "spacing_before": 20
            }
        }

    # Métodos de validación
    def _validate_header(self, header: Dict[str, Any]) -> bool:
        required_elements = ["title", "date"]
        return all(elem in header for elem in required_elements)

    def _validate_title_section(self, title: Dict[str, Any]) -> bool:
        return "main_title" in title

    def _validate_summary(self, summary: Dict[str, Any]) -> bool:
        required_elements = ["title", "content"]
        return all(elem in summary for elem in required_elements)

    def _validate_content_sections(self, sections: List[Dict[str, Any]]) -> bool:
        if not sections:
            return False
        required_elements = ["title", "content"]
        return all(
            all(elem in section for elem in required_elements)
            for section in sections
        )

    def _validate_footer(self, footer: Dict[str, Any]) -> bool:
        required_elements = ["page_number"]
        return all(elem in footer for elem in required_elements)