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

from core.orchestrator import CoreOrchestrator
from config.config import initialize_config, get_config

# Configure Rich for better error visualization
install_rich_traceback(show_locals=True, word_wrap=True)

def setup_logging(log_dir: Path) -> None:
    """Configure logging system with file rotation and rich formatting."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vyper_{datetime.now().strftime('%Y%m%d')}.log"

    # Handler for rotating files
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))

    # Console handler with Rich
    console_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True
    )

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[file_handler, console_handler]
    )

# Initialize Rich console
console = Console()

class VyperCLI:
    """Manages Vyper's command line interface."""
    
    def __init__(self):
        self.logger = logging.getLogger("vyper.cli")
        self.console = Console()
        self.orchestrator: Optional[CoreOrchestrator] = None

    def clean_markdown(self, text: str) -> str:
        """Clean and format markdown text."""
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
        """Format result for visualization."""
        try:
            if 'error' in result:
                return f"❌ Error: {result['error'].get('message', str(result['error']))}"
            
            if 'result' not in result:
                return "❌ No valid result obtained"

            formatted_content = []
            if 'team_analysis' in result:
                formatted_content.append(self.clean_markdown(result['team_analysis']))
                
            if 'execution_results' in result:
                formatted_content.append("\n📊 Execution Results:")
                for team_id, team_result in result['execution_results'].items():
                    formatted_content.append(f"\nTeam {team_id}:")
                    formatted_content.append(self.clean_markdown(str(team_result)))
            
            return "\n\n".join(formatted_content) if formatted_content else "No content found in response"
            
        except Exception as e:
            self.logger.error(f"Error formatting result: {e}", exc_info=True)
            return f"❌ Error formatting result: {str(e)}"

    @contextmanager
    def error_boundary(self, error_message: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Process interrupted by user[/yellow]")
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
        """Initialize necessary directories."""
        output_dirs = ['output', 'output/teams', 'output/results', 'logs', 'temp']
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    async def process_request(self, mode: str, prompt: str, api_key: str) -> None:
        """Process a user request."""
        try:
            await self.initialize_output_dirs()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                main_task = progress.add_task("⏳ Processing request...", total=100)
                
                # Initialize and execute orchestrator
                self.orchestrator = CoreOrchestrator(api_key, mode)
                result = await self.orchestrator.process_request({
                    "type": "general",
                    "content": prompt,
                    "metadata": {}
                })
                
                if "error" in result:
                    progress.update(main_task, completed=100, description="❌ Error")
                    self.console.print(Panel(
                        Text(f"Error: {result['error']}", style="red"),
                        title="Error",
                        border_style="red"
                    ))
                    return
                
                progress.update(main_task, completed=100, description="✅ Completed")
                formatted_result = self.format_result(result)
                
                self.console.print("\n[bold green]🎯 Result:[/bold green]")
                self.console.print(Panel(
                    Markdown(formatted_result),
                    border_style="green",
                    padding=(1, 2)
                ))
                
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.cleanup()

    def run(self) -> None:
        """Main CLI entry point."""
        with self.error_boundary("Error initializing application"):
            # Load configuration
            load_dotenv()
            initialize_config(env="development")
            
            # Check arguments
            if len(sys.argv) < 3:
                self.console.print(Panel(
                    "[yellow]Usage: python main.py (openai|deepseek) \"Your request\"[/yellow]\n\n"
                    "Examples:\n"
                    "  python main.py openai \"Create a document about AI\"\n"
                    "  python main.py deepseek \"Analyze CSV data\"\n",
                    title="Assistant Usage",
                    border_style="blue"
                ))
                sys.exit(1)

            # Get parameters
            mode = sys.argv[1].lower()
            prompt = " ".join(sys.argv[2:])
            api_key = os.getenv(f"{mode.upper()}_API_KEY")

            if not api_key:
                self.console.print(f"[red]Error: {mode.upper()}_API_KEY not found in .env[/red]")
                sys.exit(1)

            # Configure logging
            setup_logging(Path("logs"))

            # Show banner
            self.console.print(Panel(
                Text("🤖 AI Assistant", justify="center", style="bold blue"),
                subtitle=f"Mode: {mode.upper()}",
                border_style="blue"
            ))
            
            self.console.print(Panel(
                Text(prompt, justify="left"),
                title="📝 Request",
                border_style="green"
            ))

            # Execute processing
            asyncio.run(self.process_request(mode, prompt, api_key))

def main():
    """Main function."""
    cli = VyperCLI()
    cli.run()

if __name__ == "__main__":
    main()