import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

# Importamos tus clases
from team_structure import TeamStructure
from cache_manager import CacheManager
# Suponiendo que tu AssistantUI sigue existiendo con la lógica openai...
# y que tienes un "DeepSeekUI" o similar para deepseek, o usas la misma UI.

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

console = Console()
logger = logging.getLogger("rich")

class AssistantUI:
    def __init__(self):
        self.console = Console()
        self.cache_manager = CacheManager()

    async def run_assistant_openai(self, prompt: str, api_key: str):
        """Ejecuta el flujo con OpenAI, usando tu TeamStructure normal."""
        try:
            team = TeamStructure(api_key)
            result = await team.process_request(prompt)
            # Aquí mostrarías resultados, etc.
            console.print(result)
        except Exception as e:
            logger.exception("Error en la ejecución con OpenAI")
            self.console.print(f"[red]Error: {str(e)}[/red]")

    async def run_assistant_deepseek(self, prompt: str, api_key: str):
        """
        Si deseas un flujo diferente para Deepseek, podrías tener 
        otra lógica aquí o reusar la misma, pero forzando 
        un workflow que incluya el agente DeepSeekAgent.
        """
        try:
            # Por ejemplo, generas una "TeamStructure" distinta 
            # o la misma, pero con un workflow forzado
            team = TeamStructure(api_key)  
            
            # De algún modo forzar a que se use deepseek:
            # p.ej. un task artificial, o una variable en tu 
            # TeamStructure que detecte "modo deepseek"
            # Aquí, simplificamos y asumimos que lo maneja la TeamStructure:
            result = await team.process_request(f"[DEEPSEEK] {prompt}")

            # Mostrar resultados
            console.print(result)
        except Exception as e:
            logger.exception("Error en la ejecución con Deepseek")
            self.console.print(f"[red]Error: {str(e)}[/red]")

def main():
    # Cargamos variables de entorno (OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.)
    load_dotenv()

    if len(sys.argv) < 3:
        console.print(Panel(
            "[yellow]Uso: python main.py (openai|deepseek) \"Tu petición\"[/yellow]\n\n"
            "Ejemplos:\n"
            "  - python main.py openai \"Crea una presentación sobre IA\"\n"
            "  - python main.py deepseek \"Realiza una búsqueda avanzada\"\n",
            title="Uso del Asistente",
            border_style="blue"
        ))
        sys.exit(1)

    # El primer argumento define el "modo"
    mode = sys.argv[1].lower()
    # El resto se interpretará como la petición
    prompt = " ".join(sys.argv[2:])

    console.print(Panel(f"[bold blue]🤖 AI Assistant - Modo: {mode.upper()}[/bold blue]", border_style="blue"))
    console.print(Panel(prompt, title="📝 Petición", border_style="green"))

    # Obtenemos la API Key, (asumiendo la de OpenAI, pero 
    # si tu logic es distinta para deepseek, ajústalo):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("[red]Error: OPENAI_API_KEY no encontrado en variables de entorno[/red]")
        return

    ui = AssistantUI()

    if mode == "openai":
        asyncio.run(ui.run_assistant_openai(prompt, openai_api_key))
    elif mode == "deepseek":
        # Si necesitases la Deepseek API, la cargarías aquí, p. ej.:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            console.print("[red]Error: DEEPSEEK_API_KEY no encontrado en variables de entorno[/red]")
            return
        # Llamamos a la función que maneja Deepseek
        asyncio.run(ui.run_assistant_deepseek(prompt, deepseek_api_key))
    else:
        console.print(f"[red]Modo desconocido: {mode}[/red]")

if __name__ == "__main__":
    main()
