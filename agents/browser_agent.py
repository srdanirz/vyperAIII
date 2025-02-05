import json
import logging
import asyncio
from typing import Optional, Dict, Any, ClassVar, List, Union
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class BrowserAgent(BaseAgent):
    """
    Agente que interactúa con un navegador para realizar tareas web
    complejas. Soporta login, scraping, navegación, etc.
    """
    _shared_browser: ClassVar[Optional[Browser]] = None
    _auth_state: ClassVar[Dict[str, Dict[str, Any]]] = {}
    _session_data: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.max_retries = 3
        self.retry_delay = 2
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0
        )
        self.screenshots: List[bytes] = []
        self.visited_urls: List[str] = []

    async def _execute(self) -> Dict[str, Any]:
        """Ejecuta la lógica principal de navegación y automatización web."""
        try:
            task_analysis = await self._analyze_task()
            if not BrowserAgent._shared_browser:
                config = self._create_browser_config()
                BrowserAgent._shared_browser = Browser(config=config)

            async with await BrowserAgent._shared_browser.new_context() as context:
                await self._setup_context(context)
                result = await self._execute_task(context, task_analysis)
                processed_result = self._process_result(result)
                return processed_result

        except Exception as e:
            logger.error(f"Error in BrowserAgent: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "screenshots": [f"<{len(img)} bytes>" for img in self.screenshots],
                "visited_urls": self.visited_urls
            }

    def _create_browser_config(self) -> BrowserConfig:
        """Crea la configuración del navegador."""
        return BrowserConfig(
            headless=True,
            disable_security=True,
            new_context_config=BrowserContextConfig(
                wait_for_network_idle_page_load_time=3.0,
                browser_window_size={'width': 1280, 'height': 1100},
                locale='en-US',
                highlight_elements=True,
                viewport_expansion=500
            )
        )

    async def _analyze_task(self) -> Dict[str, Any]:
        """Analiza la tarea específica para planificar acciones web."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Analiza la tarea web y crea un plan de ejecución detallado. "
                    "Identifica acciones de navegación, monitoreo, envío de formularios, etc."
                )
            },
            {
                "role": "user",
                "content": f"Tarea: {self.task}\nContexto: {json.dumps(self.metadata)}"
            }
        ]
        
        response = await self.llm.agenerate([messages])
        content = response.generations[0][0].message.content
        return self._extract_json_from_response(content)

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extrae JSON o crea un plan básico si falla el parseo."""
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            # Como fallback, se busca entre llaves
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > 0:
                return json.loads(text[start:end+1])
        except Exception:
            logger.warning("Fallo parseando JSON, creando plan básico.")
        return {
            "task_type": "navigation",
            "required_actions": [{"type": "navigate", "url": "https://example.com"}],
            "error_handling": {"fallback": "retry"}
        }

    async def _setup_context(self, context: BrowserContext) -> None:
        """Configura contexto (cookies, auth, request interception)."""
        if self.metadata.get("intercept_requests"):
            await context.route("**/*", self._handle_request)
        auth_key = self.metadata.get("auth_key")
        if auth_key and auth_key in self._auth_state:
            await context.add_cookies(self._auth_state[auth_key])

    async def _handle_request(self, route: Any, request: Any) -> None:
        """Intercepta y modifica requests si es necesario."""
        if "headers" in self.metadata:
            await route.continue_(headers=self.metadata["headers"])
        else:
            await route.continue_()

    async def _execute_task(self, context: BrowserContext, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta cada acción planificada en el análisis."""
        try:
            # Usa un agente de browser real (de la librería browser_use) para ejecutar las acciones.
            agent_config = {
                "task": json.dumps(task_analysis),
                "llm": self.llm,
                "browser_context": context,
            }
            browser_agent = Agent(**agent_config)

            for attempt in range(self.max_retries):
                try:
                    result = await browser_agent.run()
                    self.visited_urls.extend(result.urls() if hasattr(result, 'urls') else [])
                    return {
                        "status": "success",
                        "raw_result": result
                    }
                except Exception as e:
                    logger.error(f"Browser action failed attempt {attempt+1}: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

            return {"error": "No se pudo completar la tarea en el navegador."}

        except Exception as e:
            return {"error": str(e)}

    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el resultado final para devolver algo limpio."""
        if "error" in result:
            return {
                "error": result["error"],
                "screenshots": [f"<{len(img)} bytes>" for img in self.screenshots],
                "visited_urls": self.visited_urls
            }
        return {
            "status": result.get("status", "unknown"),
            "visited_urls": self.visited_urls
        }

    @classmethod
    async def cleanup(cls) -> None:
        """Cierra el navegador y limpia estados."""
        if cls._shared_browser:
            await cls._shared_browser.close()
            cls._shared_browser = None
        cls._auth_state.clear()
        cls._session_data.clear()
