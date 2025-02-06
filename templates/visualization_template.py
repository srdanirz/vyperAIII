# templates/visualization_template.py

from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_template import BaseTemplate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class VisualizationTemplate(BaseTemplate):
    """Template especializado para gráficos y visualizaciones"""

    CHART_TYPES = {
        "bar": {
            "function": "create_bar_chart",
            "required_data": ["categories", "values"],
            "optional_data": ["error_bars", "colors", "patterns"]
        },
        "line": {
            "function": "create_line_chart",
            "required_data": ["x_values", "y_values"],
            "optional_data": ["markers", "line_styles", "area_fill"]
        },
        "pie": {
            "function": "create_pie_chart",
            "required_data": ["values", "labels"],
            "optional_data": ["explode", "colors", "start_angle"]
        },
        "scatter": {
            "function": "create_scatter_plot",
            "required_data": ["x_values", "y_values"],
            "optional_data": ["sizes", "colors", "markers"]
        },
        "heatmap": {
            "function": "create_heatmap",
            "required_data": ["matrix"],
            "optional_data": ["x_labels", "y_labels", "color_map"]
        }
    }

    COLOR_PALETTES = {
        "professional": ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F"],
        "modern": ["#1A237E", "#311B92", "#4527A0", "#512DA8", "#5E35B1"],
        "creative": ["#6200EA", "#651FFF", "#7C4DFF", "#8C9EFF", "#536DFE"]
    }

    def __init__(self, template_name: str, config: Dict[str, Any]):
        super().__init__(template_name, config)
        self.style_config = config.get("visualization", {})
        self.output_dir = Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo global de matplotlib
        self._setup_matplotlib_style()

    def _setup_matplotlib_style(self):
        """Configura el estilo global de matplotlib"""
        plt.style.use('seaborn')
        
        # Configuraciones personalizadas
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'lines.linewidth': 2,
            'text.color': '#2C3E50',
            'axes.edgecolor': '#2C3E50',
            'xtick.color': '#2C3E50',
            'ytick.color': '#2C3E50'
        })

    async def apply(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica el template a la visualización"""
        try:
            # Validar contenido
            if not self.validate_content(content):
                raise ValueError("Invalid visualization content structure")

            # Crear visualización
            chart_type = content.get("type", "bar")
            if chart_type not in self.CHART_TYPES:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            # Obtener función de creación de gráfico
            chart_function = getattr(self, self.CHART_TYPES[chart_type]["function"])
            
            # Crear gráfico
            figure = chart_function(content)
            
            # Guardar gráfico
            output_path = self._save_visualization(figure, content)
            
            return {
                "type": "visualization",
                "chart_type": chart_type,
                "output_path": str(output_path),
                "metadata": {
                    **self.metadata,
                    "content_hash": hash(json.dumps(content))
                }
            }

        except Exception as e:
            logger.error(f"Error applying visualization template: {e}")
            raise

    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Valida la estructura del contenido de visualización"""
        try:
            chart_type = content.get("type")
            if chart_type not in self.CHART_TYPES:
                return False

            # Validar datos requeridos
            required_data = self.CHART_TYPES[chart_type]["required_data"]
            return all(data in content for data in required_data)

        except Exception as e:
            logger.error(f"Error validating visualization content: {e}")
            return False

    def create_bar_chart(self, content: Dict[str, Any]) -> plt.Figure:
        """Crea un gráfico de barras"""
        fig, ax = plt.subplots()
        
        categories = content["categories"]
        values = content["values"]
        colors = content.get("colors", self.COLOR_PALETTES[self.template_name])
        
        bars = ax.bar(categories, values, color=colors)
        
        if "error_bars" in content:
            ax.errorbar(
                categories,
                values,
                yerr=content["error_bars"],
                fmt="none",
                color="#2C3E50",
                capsize=5
            )
        
        # Personalización
        ax.set_title(content.get("title", ""))
        ax.set_xlabel(content.get("x_label", ""))
        ax.set_ylabel(content.get("y_label", ""))
        
        if content.get("show_values", False):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:,.0f}',
                    ha='center',
                    va='bottom'
                )
        
        return fig

    def create_line_chart(self, content: Dict[str, Any]) -> plt.Figure:
        """Crea un gráfico de líneas"""
        fig, ax = plt.subplots()
        
        x_values = content["x_values"]
        y_values = content["y_values"]
        
        # Soporte para múltiples líneas
        if isinstance(y_values[0], (list, tuple)):
            for idx, y_data in enumerate(y_values):
                ax.plot(
                    x_values,
                    y_data,
                    label=content.get("labels", [])[idx] if "labels" in content else f"Series {idx+1}",
                    color=self.COLOR_PALETTES[self.template_name][idx % len(self.COLOR_PALETTES[self.template_name])]
                )
        else:
            ax.plot(x_values, y_values, color=self.COLOR_PALETTES[self.template_name][0])
        
        # Personalización
        ax.set_title(content.get("title", ""))
        ax.set_xlabel(content.get("x_label", ""))
        ax.set_ylabel(content.get("y_label", ""))
        
        if content.get("show_grid", True):
            ax.grid(True, linestyle='--', alpha=0.7)
        
        if content.get("show_legend", True) and isinstance(y_values[0], (list, tuple)):
            ax.legend()
        
        return fig

    def create_pie_chart(self, content: Dict[str, Any]) -> plt.Figure:
        """Crea un gráfico circular"""
        fig, ax = plt.subplots()
        
        values = content["values"]
        labels = content["labels"]
        colors = content.get("colors", self.COLOR_PALETTES[self.template_name])
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=content.get("start_angle", 90),
            explode=content.get("explode", None)
        )
        
        # Personalización
        ax.set_title(content.get("title", ""))
        
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=12)
        
        return fig

    def _save_visualization(self, figure: plt.Figure, content: Dict[str, Any]) -> Path:
        """Guarda la visualización en un archivo"""
        filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / filename
        
        # Guardar con alta calidad
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            transparent=content.get("transparent", False)
        )
        
        plt.close(figure)
        return output_path

    def validate(self) -> bool:
        """Valida que el template tenga todos los componentes necesarios"""
        return True  # Siempre válido para visualizaciones básicas