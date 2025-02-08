import os
import sys
import asyncio
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.traceback import install as install_rich_traceback

from core.managers.team_manager import DynamicTeamManager
from config.config import initialize_config, get_config

# Configuración de Rich para mejor visualización de errores
install_rich_traceback(show_locals=True, word_wrap=True)

# Configuración de logging
def setup_logging(log_dir: Path) -> None:
    """Configura el sistema de logging con rotación de archivos y formato enriquecido."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vyper_{datetime.now().strftime('%Y%m%d')}.log"

    # Handler para archivos con rotación
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))

    # Handler para consola con Rich
    console_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True
    )

    # Configuración root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[file_handler, console_handler]
    )

# Inicialización de Rich console
console = Console()

class VyperCLI:
    """Gestiona la interfaz de línea de comandos de Vyper."""
    
    def __init__(self):
        self.logger = logging.getLogger("vyper.cli")
        self.console = Console()
        self.team_manager: Optional[DynamicTeamManager] = None

    def clean_markdown(self, text: str) -> str:
        """Limpia y formatea texto markdown."""
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

    def format_result(self, result: dict) -> str:
        """Formatea el resultado para visualización."""
        try:
            if 'error' in result:
                return f"❌ Error: {result['error'].get('message', str(result['error']))}"
            
            if 'result' not in result:
                return "❌ No se obtuvo un resultado válido"

            formatted_content = []
            if 'team_analysis' in result:
                formatted_content.append(self.clean_markdown(result['team_analysis']))
                
            if 'execution_results' in result:
                formatted_content.append("\n📊 Resultados de ejecución:")
                for team_id, team_result in result['execution_results'].items():
                    formatted_content.append(f"\nEquipo {team_id}:")
                    formatted_content.append(self.clean_markdown(str(team_result)))
            
            return "\n\n".join(formatted_content) if formatted_content else "No se encontró contenido en la respuesta"
            
        except Exception as e:
            self.logger.error(f"Error formateando resultado: {e}", exc_info=True)
            return f"❌ Error al formatear el resultado: {str(e)}"

    @contextmanager
    def error_boundary(self, error_message: str):
        """Context manager para manejo consistente de errores."""
        try:
            yield
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Proceso interrumpido por el usuario[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.logger.exception(error_message)
            self.console.print(Panel(
                f"{error_message}: {str(e)}",
                title="Error",
                border_style="red"
            ))
            sys.exit(1)

    async def initialize_output_dirs(self) -> None:
        """Inicializa directorios necesarios."""
        output_dirs = ['output', 'output/teams', 'output/results', 'logs', 'temp']
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    async def process_request(self, mode: str, prompt: str, api_key: str) -> None:
        """Procesa una solicitud del usuario."""
        try:
            await self.initialize_output_dirs()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                main_task = progress.add_task("⏳ Procesando solicitud...", total=100)
                
                # Inicializar y ejecutar team manager
                self.team_manager = DynamicTeamManager(api_key, mode)
                result = await self.team_manager.process_request({"prompt": prompt})
                
                if "error" in result:
                    progress.update(main_task, completed=100, description="❌ Error")
                    self.console.print(Panel(
                        Text(f"Error: {result['error']}", style="red"),
                        title="Error",
                        border_style="red"
                    ))
                    return
                
                progress.update(main_task, completed=100, description="✅ Completado")
                formatted_result = self.format_result(result)
                
                self.console.print("\n[bold green]🎯 Resultado:[/bold green]")
                self.console.print(Panel(
                    Markdown(formatted_result),
                    border_style="green",
                    padding=(1, 2)
                ))
                
        finally:
            # Cleanup
            if self.team_manager:
                await self.team_manager.cleanup()

    def run(self) -> None:
        """Punto de entrada principal del CLI."""
        with self.error_boundary("Error inicializando la aplicación"):
            # Cargar configuración
            load_dotenv()
            initialize_config(env="development")
            
            # Verificar argumentos
            if len(sys.argv) < 3:
                self.console.print(Panel(
                    "[yellow]Uso: python main.py (openai|deepseek) \"Tu petición\"[/yellow]\n\n"
                    "Ejemplos:\n"
                    "  python main.py openai \"Crea un documento sobre IA\"\n"
                    "  python main.py deepseek \"Analiza datos de CSV\"\n",
                    title="Uso del Asistente",
                    border_style="blue"
                ))
                sys.exit(1)

            # Obtener parámetros
            mode = sys.argv[1].lower()
            prompt = " ".join(sys.argv[2:])
            api_key = os.getenv(f"{mode.upper()}_API_KEY")

            if not api_key:
                self.console.print(f"[red]Error: no se encontró {mode.upper()}_API_KEY en .env[/red]")
                sys.exit(1)

            # Configurar logging
            setup_logging(Path("logs"))

            # Mostrar banner
            self.console.print(Panel(
                Text("🤖 AI Assistant", justify="center", style="bold blue"),
                subtitle=f"Modo: {mode.upper()}",
                border_style="blue"
            ))
            
            self.console.print(Panel(
                Text(prompt, justify="left"),
                title="📝 Petición",
                border_style="green"
            ))

            # Ejecutar procesamiento
            asyncio.run(self.process_request(mode, prompt, api_key))

def main():
    """Función principal."""
    cli = VyperCLI()
    cli.run()

if __name__ == "__main__":
    main()