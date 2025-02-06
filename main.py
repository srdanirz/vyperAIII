# main.py

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
from rich.text import Text
from rich.padding import Padding
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.traceback import install as install_rich_traceback

from team_structure import TeamStructure

# Configurar Rich traceback
install_rich_traceback(show_locals=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")
console = Console()

def clean_markdown(text: str) -> str:
    """Limpia y mejora el formato markdown del texto"""
    if not isinstance(text, str):
        return str(text)
        
    # Normaliza los saltos de línea
    text = text.replace('\n\n\n', '\n\n')
    
    # Asegura que los títulos tengan espacio antes
    text = text.replace('\n#', '\n\n#')
    
    # Mejora el formato de las listas
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Convierte bullets punto con número a formato markdown
        if line.strip().startswith(f"{i+1} "):
            lines[i] = line.replace(f"{i+1} ", f"{i+1}. ")
            
        # Asegura que las viñetas tengan el formato correcto
        if line.strip().startswith('•'):
            lines[i] = line.replace('•', '-')
            
    text = '\n'.join(lines)
    
    return text.strip()

def format_result(result: dict) -> str:
    """Formatea el resultado para mostrar solo la información relevante"""
    try:
        if 'error' in result:
            return f"❌ Error: {result['error'].get('message', str(result['error']))}"
            
        if 'result' not in result:
            return "❌ No se obtuvo un resultado válido"
            
        main_result = result['result'].get('result', {})
        
        # Procesamos diferentes tipos de resultados
        formatted_content = []
        
        # Investigación
        if 'research' in main_result and 'research_findings' in main_result['research']:
            formatted_content.append(clean_markdown(main_result['research']['research_findings']))
            
        # Contenido generado
        if 'contentgeneration' in main_result:
            content_info = main_result['contentgeneration']
            if isinstance(content_info, dict):
                if 'output_file' in content_info:
                    formatted_content.append(f"\n📄 Archivo generado: {content_info['output_file']}")
                if 'content_type' in content_info:
                    formatted_content.append(f"📋 Tipo de contenido: {content_info['content_type']}")
                if 'metadata' in content_info:
                    meta = content_info['metadata']
                    if isinstance(meta, dict) and 'statistics' in meta:
                        stats = meta['statistics']
                        formatted_content.append("\n📊 Estadísticas:")
                        for key, value in stats.items():
                            formatted_content.append(f"  - {key}: {value}")
                
        # Validación
        if 'validation' in main_result and 'validation_summary' in main_result['validation']:
            validation = main_result['validation']['validation_summary']
            if isinstance(validation, dict):
                status = validation.get('overall_status', 'N/A')
                score = validation.get('final_score', 'N/A')
                formatted_content.append(f"\n✅ Validación: {status} (Score: {score})")
                
                # Sugerencias de mejora
                suggestions = validation.get('improvement_suggestions', [])
                if suggestions:
                    formatted_content.append("\n📝 Sugerencias de mejora:")
                    for suggestion in suggestions:
                        if isinstance(suggestion, dict):
                            priority = suggestion.get('priority', 'normal')
                            content = suggestion.get('suggestion', str(suggestion))
                            formatted_content.append(f"  - [{priority}] {content}")
                        else:
                            formatted_content.append(f"  - {suggestion}")
        
        return "\n\n".join(formatted_content) if formatted_content else "No se encontró contenido en la respuesta"
        
    except Exception as e:
        logger.error(f"Error formateando resultado: {e}")
        return f"❌ Error al formatear el resultado: {str(e)}"

async def initialize_output_dirs():
    """Inicializa los directorios necesarios"""
    output_dirs = [
        'output',
        'output/presentations',
        'output/documents',
        'output/visualizations',
        'output/temp',
        'logs'
    ]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

async def run():
    """Ejecuta el procesamiento principal"""
    try:
        # Inicializar directorios
        await initialize_output_dirs()
        
        # Crear barra de progreso
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Crear tarea principal
            main_task = progress.add_task(
                "⏳ Procesando solicitud...",
                total=100
            )
            
            # Procesar solicitud
            result = await team.process_request(prompt, engine_mode=mode)
            
            if "error" in result:
                progress.update(main_task, completed=100, description="❌ Error")
                console.print(Panel(
                    Text(f"Error: {result['error'].get('message', 'Error desconocido')}", style="red"),
                    title="Error",
                    border_style="red"
                ))
                return
                
            # Actualizar progreso
            progress.update(main_task, completed=100, description="✅ Completado")
            
            # Formatear y mostrar resultado
            formatted_result = format_result(result)
            
            # Mostrar resultado en panel
            console.print("\n[bold green]🎯 Resultado:[/bold green]")
            console.print(Panel(
                Markdown(formatted_result),
                border_style="green",
                padding=(1, 2)
            ))
            
            # Si se generó un archivo, mostrar información
            if 'result' in result and 'result' in result['result']:
                content_gen = result['result']['result'].get('contentgeneration', {})
                if isinstance(content_gen, dict) and 'output_file' in content_gen:
                    file_path = Path(content_gen['output_file'])
                    if file_path.exists():
                        size = file_path.stat().st_size / 1024  # KB
                        console.print(Panel(
                            f"📄 Archivo generado: {file_path}\n"
                            f"📊 Tamaño: {size:.2f} KB\n"
                            f"⏰ Generado: {datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
                            title="Archivo de salida",
                            border_style="blue"
                        ))
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]Proceso interrumpido por el usuario[/yellow]")
    except Exception as e:
        console.print(Panel(
            f"Error inesperado: {str(e)}",
            title="Error",
            border_style="red"
        ))
        logger.exception("Error inesperado durante la ejecución")

def main():
    """Función principal"""
    load_dotenv()

    if len(sys.argv) < 3:
        console.print(Panel(
            "[yellow]Uso: python main.py (openai|deepseek) \"Tu petición\"[/yellow]\n\n"
            "Ejemplos:\n"
            "  python main.py openai \"Crea un documento sobre IA\"\n"
            "  python main.py deepseek \"Analiza datos de CSV\"\n",
            title="Uso del Asistente",
            border_style="blue"
        ))
        sys.exit(1)

    global mode, prompt, team
    
    mode = sys.argv[1].lower()
    prompt = " ".join(sys.argv[2:])

    # Header
    console.print(Panel(
        Text("🤖 AI Assistant", justify="center", style="bold blue"),
        subtitle=f"Modo: {mode.upper()}",
        border_style="blue"
    ))
    
    # Prompt
    console.print(Panel(
        Text(prompt, justify="left"),
        title="📝 Petición",
        border_style="green"
    ))

    # Verificar claves API
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY")
    }

    if not api_keys[mode]:
        console.print(f"[red]Error: no se encontró {mode.upper()}_API_KEY en .env[/red]")
        return

    # Crear TeamStructure
    team = TeamStructure(api_keys[mode], engine_mode=mode)

    # Ejecutar
    asyncio.run(run())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Programa terminado por el usuario[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error fatal: {str(e)}[/red]")
        logger.exception("Error fatal en la aplicación")