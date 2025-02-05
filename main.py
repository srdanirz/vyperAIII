import os
import asyncio
import logging
import sys
import base64
import json
from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from dotenv import load_dotenv
from team_structure import TeamStructure
from cache_manager import CacheManager

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

console = Console()
logger = logging.getLogger("rich")

class AssistantUI:
    """
    Clase principal de interfaz, orquesta la ejecución y despliegue de resultados.
    """
    def __init__(self):
        self.console = Console()
        self.cache_manager = CacheManager()

    async def run_assistant(self, prompt: str, api_key: str):
        """Ejecuta el flujo del asistente con un prompt y una API Key."""
        try:
            team = TeamStructure(api_key)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task_id = progress.add_task("Procesando solicitud...", total=None)
                result = await team.process_request(prompt)
                progress.remove_task(task_id)

            if result.get("status") == "success":
                messages = result.get("messages", [])
                if messages:
                    await self._display_messages(messages)
                content_result = result.get("result", {}).get("result", {})
                await self._handle_generated_content(content_result)
                self._display_execution_stats(result.get("metadata", {}))
            else:
                error_message = result.get("error", "Error desconocido.")
                self.console.print(f"[red]Error: {error_message}[/red]")
        except Exception as e:
            logger.exception("Error en la ejecución del asistente")
            self.console.print(f"[red]Error: {str(e)}[/red]")

    async def _display_messages(self, messages: List[Dict[str, Any]]):
        """Despliega mensajes con formateo y separaciones."""
        prefixes = {
            "System": "🔄",
            "Team": "👥",
            "Tech Director": "💻",
            "Research Director": "🔍",
            "Creative Director": "💡",
            "Knowledge Director": "📚",
            "Innovation Director": "🚀",
            "Analysis": "🔎",
            "Browser": "🌐",
            "ContentGeneration": "✍️",
            "Research": "📖",
            "DataProcessing": "🔢",
            "Validation": "✅",
            "Coordination": "🔄"
        }
        last_speaker = None
        for msg in messages:
            prefix = prefixes.get(msg["from_agent"], "💬")
            content = msg["content"].strip()
            
            if last_speaker and last_speaker != msg["from_agent"]:
                self.console.print("─" * 60, style="dim")
            
            if msg["from_agent"] == "System":
                self.console.print(Panel(
                    f"{content}",
                    title=f"{prefix} System Message",
                    border_style="blue"
                ))
            else:
                self.console.print(
                    f"{prefix} [bold]{msg['from_agent']}[/bold]: {content}",
                    highlight=True
                )
            last_speaker = msg["from_agent"]
            await asyncio.sleep(0.05)

    async def _handle_generated_content(self, content_result: Dict[str, Any]) -> None:
        """Procesa y guarda contenido según formato (pptx, docx, etc.)."""
        if not isinstance(content_result, dict):
            return
        for agent_key, agent_val in content_result.items():
            if isinstance(agent_val, dict) and "format" in agent_val and "content" in agent_val:
                await self._save_content(agent_val)

    async def _save_content(self, content: Dict[str, Any]) -> None:
        """Guarda el contenido en un archivo según su formato."""
        FORMAT_HANDLERS = {
            "pptx": {"extension": "pptx", "description": "Presentación", "mode": "wb"},
            "docx": {"extension": "docx", "description": "Documento Word", "mode": "wb"},
            "pdf": {"extension": "pdf", "description": "PDF", "mode": "wb"},
            "png": {"extension": "png", "description": "Imagen", "mode": "wb"},
            "txt": {"extension": "txt", "description": "Texto", "mode": "w"},
            "json": {"extension": "json", "description": "JSON", "mode": "w"},
            "html": {"extension": "html", "description": "HTML", "mode": "w"},
            "code": {"extension": "py", "description": "Código", "mode": "w"}
        }
        fmt = content["format"]
        handler = FORMAT_HANDLERS.get(fmt)
        if not handler:
            self.console.print(f"[yellow]Formato no soportado: {fmt}[/yellow]")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title = content.get("title", "generated").strip().replace(" ", "_")[:50]
        file_name = f"{title}_{timestamp}.{handler['extension']}"
        mode = handler["mode"]
        try:
            if mode == "wb":
                raw_bytes = base64.b64decode(content["content"])
                with open(file_name, mode) as f:
                    f.write(raw_bytes)
            else:
                with open(file_name, mode, encoding='utf-8') as f:
                    if isinstance(content["content"], (dict, list)):
                        json.dump(content["content"], f, indent=2)
                    else:
                        f.write(content["content"])
            self.console.print(f"[green]{handler['description']} guardado como {file_name}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error guardando {handler['description']}: {str(e)}[/red]")

    def _display_execution_stats(self, metadata: Dict[str, Any]) -> None:
        """Muestra estadísticas de ejecución si las hubiera."""
        exec_stats = metadata.get("execution_stats")
        if exec_stats:
            self.console.print("\n[bold]Estadísticas de Ejecución:[/bold]")
            success = exec_stats.get("successful_executions", 0)
            total = exec_stats.get("total_executions", 0)
            avg_time = exec_stats.get("average_duration", 0.0)
            self.console.print(f"  - Éxitos: {success}/{total}")
            self.console.print(f"  - Tiempo promedio: {avg_time:.2f} segundos")

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY no encontrado en variables de entorno[/red]")
        return

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not prompt:
        console.print(Panel(
            "[yellow]Uso: python main.py \"Tu petición\"[/yellow]\n\n"
            "Ejemplos:\n"
            "  - python main.py \"Crea una presentación sobre IA\"\n"
            "  - python main.py \"Genera un reporte de aprendizaje automático\"",
            title="Uso del Asistente",
            border_style="blue"
        ))
        return

    ui = AssistantUI()

    console.print(Panel("[bold blue]🤖 AI Assistant[/bold blue]", border_style="blue"))
    console.print(Panel(prompt, title="📝 Petición", border_style="green"))

    asyncio.run(ui.run_assistant(prompt, api_key))

if __name__ == "__main__":
    main()
