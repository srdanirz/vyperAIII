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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.traceback import install as install_rich_traceback

from dynamic_team_manager import DynamicTeamManager
from config.config import initialize_config, get_config

install_rich_traceback(show_locals=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")
console = Console()

def clean_markdown(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.replace('\n\n\n', '\n\n').replace('\n#', '\n\n#')
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{i+1} "):
            lines[i] = line.replace(f"{i+1} ", f"{i+1}. ")
        if line.strip().startswith('•'):
            lines[i] = line.replace('•', '-')
    return '\n'.join(lines).strip()

def format_result(result: dict) -> str:
    try:
        if 'error' in result:
            return f"❌ Error: {result['error'].get('message', str(result['error']))}"
        if 'result' not in result:
            return "❌ No se obtuvo un resultado válido"

        formatted_content = []
        if 'team_analysis' in result:
            formatted_content.append(clean_markdown(result['team_analysis']))
        if 'execution_results' in result:
            formatted_content.append("\n📊 Resultados de ejecución:")
            for team_id, team_result in result['execution_results'].items():
                formatted_content.append(f"\nEquipo {team_id}:")
                formatted_content.append(clean_markdown(str(team_result)))
        
        return "\n\n".join(formatted_content) if formatted_content else "No se encontró contenido en la respuesta"
    except Exception as e:
        logger.error(f"Error formateando resultado: {e}")
        return f"❌ Error al formatear el resultado: {str(e)}"

async def initialize_output_dirs():
    output_dirs = ['output', 'output/teams', 'output/results', 'logs']
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

async def run():
    try:
        await initialize_output_dirs()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            main_task = progress.add_task("⏳ Procesando solicitud...", total=100)
            
            team_manager = DynamicTeamManager(api_key, mode)
            result = await team_manager.process_request({"prompt": prompt})
            
            if "error" in result:
                progress.update(main_task, completed=100, description="❌ Error")
                console.print(Panel(
                    Text(f"Error: {result['error']}", style="red"),
                    title="Error",
                    border_style="red"
                ))
                return
                
            progress.update(main_task, completed=100, description="✅ Completado")
            formatted_result = format_result(result)
            
            console.print("\n[bold green]🎯 Resultado:[/bold green]")
            console.print(Panel(
                Markdown(formatted_result),
                border_style="green",
                padding=(1, 2)
            ))
            
            await team_manager.cleanup()
                    
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
    load_dotenv()
    initialize_config(env="development")

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

    global mode, prompt, api_key
    mode = sys.argv[1].lower()
    prompt = " ".join(sys.argv[2:])
    api_key = os.getenv(f"{mode.upper()}_API_KEY")

    if not api_key:
        console.print(f"[red]Error: no se encontró {mode.upper()}_API_KEY en .env[/red]")
        return

    console.print(Panel(
        Text("🤖 AI Assistant", justify="center", style="bold blue"),
        subtitle=f"Modo: {mode.upper()}",
        border_style="blue"
    ))
    
    console.print(Panel(
        Text(prompt, justify="left"),
        title="📝 Petición",
        border_style="green"
    ))

    asyncio.run(run())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Programa terminado por el usuario[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error fatal: {str(e)}[/red]")
        logger.exception("Error fatal en la aplicación")