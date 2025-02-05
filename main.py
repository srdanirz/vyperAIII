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
from rich.layout import Layout
from dotenv import load_dotenv
from team_structure import TeamStructure
from cache_manager import CacheManager

# Configure logging
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
        self.layout = Layout()
        self.cache_manager = CacheManager()

    def setup_layout(self):
        """Setup the UI layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

    async def display_messages(self, messages: List[Dict[str, Any]]):
        """Display messages with better formatting and visual cues"""
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
            await asyncio.sleep(0.1)

    async def run_assistant(self, prompt: str, api_key: str):
        """Run the assistant with dynamic content handling"""
        try:
            team = TeamStructure(api_key)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Processing request...", total=None)
                result = await team.process_request(prompt)
                progress.remove_task(task)
            
            if result.get("status") == "success":
                await self.display_messages(result.get("messages", []))
                
                # Handle generated content
                if "result" in result:
                    content_result = result["result"].get("result", {})
                    await self._handle_generated_content(content_result)
                
                # Display execution stats
                self._display_execution_stats(result.get("metadata", {}))
            else:
                error_message = result.get("error", "Unknown error occurred")
                self.console.print(f"[red]Error: {error_message}[/red]")

        except Exception as e:
            logger.exception("Error in assistant execution")
            self.console.print(f"[red]Error: {str(e)}[/red]")

    async def _handle_generated_content(self, content_result: Dict[str, Any]) -> None:
        """Handle various types of generated content"""
        if isinstance(content_result, dict):
            for agent_result in content_result.values():
                if isinstance(agent_result, dict) and "format" in agent_result and "content" in agent_result:
                    try:
                        await self._save_content(agent_result)
                    except Exception as e:
                        logger.error(f"Error saving content: {e}")

    async def _save_content(self, content: Dict[str, Any]) -> None:
        """Save content based on its format"""
        content_format = content["format"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        FORMAT_HANDLERS = {
            "pptx": {"extension": "pptx", "description": "PowerPoint presentation", "mode": "wb"},
            "docx": {"extension": "docx", "description": "Word document", "mode": "wb"},
            "pdf": {"extension": "pdf", "description": "PDF document", "mode": "wb"},
            "png": {"extension": "png", "description": "image", "mode": "wb"},
            "txt": {"extension": "txt", "description": "text document", "mode": "w"},
            "json": {"extension": "json", "description": "JSON file", "mode": "w"},
            "html": {"extension": "html", "description": "HTML document", "mode": "w"},
            "code": {"extension": "py", "description": "code file", "mode": "w"}
        }

        if content_format in FORMAT_HANDLERS:
            handler = FORMAT_HANDLERS[content_format]
            
            extension = content.get("language", handler["extension"]) if content_format == "code" else handler["extension"]
            title = content.get("title", "").lower()
            title = "".join(c for c in title if c.isalnum() or c in "- _").strip()
            title = title[:50] if title else "generated"
            filename = f"{title}_{timestamp}.{extension}"
            
            mode = handler["mode"]
            try:
                if mode == "wb":
                    content_bytes = base64.b64decode(content["content"])
                    with open(filename, mode) as f:
                        f.write(content_bytes)
                else:
                    with open(filename, mode, encoding='utf-8') as f:
                        if isinstance(content["content"], (dict, list)):
                            json.dump(content["content"], f, indent=2)
                        else:
                            f.write(content["content"])
                
                self.console.print(f"\n[green]{handler['description'].capitalize()} saved as: {filename}[/green]")
                
            except Exception as e:
                self.console.print(f"[red]Error saving {handler['description']}: {str(e)}[/red]")

    def _display_execution_stats(self, metadata: Dict[str, Any]) -> None:
        """Display execution statistics"""
        if "execution_stats" in metadata:
            stats = metadata["execution_stats"]
            self.console.print("\n[bold]Execution Statistics:[/bold]")
            self.console.print(f"Total Time: {stats.get('average_duration', 0):.2f}s")
            self.console.print(f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%")

def main():
    # Initialize
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
        return

    # Get prompt
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not prompt:
        console.print(Panel(
            "[yellow]Usage: python main.py \"Your request here\"[/yellow]\n\n"
            "Examples:\n"
            "  - python main.py \"Create a presentation about AI\"\n"
            "  - python main.py \"Write a document about machine learning\"\n"
            "  - python main.py \"Generate a report on deep learning\"\n"
            "  - python main.py \"Create a visualization of AI trends\"",
            title="Assistant Usage",
            border_style="blue"
        ))
        return

    # Initialize UI
    ui = AssistantUI()
    ui.setup_layout()

    # Display header
    console.print(Panel(
        "[bold blue]🤖 AI Assistant[/bold blue]",
        border_style="blue"
    ))
    
    # Display request
    console.print(Panel(
        f"{prompt}",
        title="📝 Request",
        border_style="green"
    ))
    
    # Run assistant
    asyncio.run(ui.run_assistant(prompt, api_key))

if __name__ == "__main__":
    main()