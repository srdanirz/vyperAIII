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
    Enhanced browser agent capable of complex web interactions using AI.
    Handles various web tasks including research, monitoring, transactions, etc.
    """
    _shared_browser: ClassVar[Optional[Browser]] = None
    _auth_state: ClassVar[Dict[str, Dict[str, Any]]] = {}
    _session_data: ClassVar[Dict[str, Any]] = {}

    def __init__(self, task: str, openai_api_key: str, metadata: Optional[Dict[str, Any]] = None, partial_data: Optional[Dict[str, Any]] = None):
        super().__init__(task, openai_api_key, metadata, partial_data)
        self.browser_agent = None
        self.max_retries = 3
        self.retry_delay = 2
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0
        )
        self.screenshots = []
        self.visited_urls = []

    def _create_browser_config(self) -> BrowserConfig:
        """Create browser configuration based on documentation"""
        return BrowserConfig(
            headless=True,  # Run without UI
            disable_security=True,  # Helpful for general web interactions
            new_context_config=BrowserContextConfig(
                wait_for_network_idle_page_load_time=3.0,  # Increased for better reliability
                browser_window_size={'width': 1280, 'height': 1100},
                locale='en-US',
                highlight_elements=True,
                viewport_expansion=500  # Default value for context
            )
        )

    async def _execute(self) -> Dict[str, Any]:
        """Execute browser automation task"""
        try:
            # Analyze task to understand required actions
            task_analysis = await self._analyze_task()
            
            # Initialize browser if needed
            if not BrowserAgent._shared_browser:
                config = self._create_browser_config()
                BrowserAgent._shared_browser = Browser(config=config)

            # Execute the task with browser context
            async with await BrowserAgent._shared_browser.new_context() as context:
                # Setup context (cookies, auth, etc.)
                await self._setup_context(context)
                
                # Execute main task
                result = await self._execute_task(context, task_analysis)
                
                # Process and validate results
                processed_result = self._process_result(result)
                
                # Store session data if available
                if "session_data" in result.get("result", {}):
                    self._store_session_data(result["result"]["session_data"])
                
                return processed_result

        except Exception as e:
            logger.error(f"Error in BrowserAgent: {str(e)}")
            return {
                "error": str(e),
                "screenshots": self.screenshots,
                "visited_urls": self.visited_urls
            }

    async def _analyze_task(self) -> Dict[str, Any]:
        """Analyze any type of web task requirements"""
        messages = [
            {"role": "system", "content": """Analyze the given web task and create a detailed execution plan.
            Consider all possible web interactions including:
            - Information gathering and research
            - Monitoring and tracking
            - Form filling and data submission
            - Navigation and interaction with web elements
            - E-commerce and transactions
            - Data extraction and downloading
            - Authentication and login processes
            
            Return your analysis in this format:
            {
                "task_type": "The general category of the task (research/monitor/transaction/etc)",
                "required_actions": [
                    {
                        "type": "action type (navigate/click/type/etc)",
                        "description": "what needs to be done",
                        "target": "where/what to interact with",
                        "expected_result": "what should happen"
                    }
                ],
                "required_data": {
                    "inputs": ["data needed for the task"],
                    "outputs": ["data to collect/verify"]
                },
                "success_criteria": ["how to verify task completion"],
                "error_handling": {
                    "potential_issues": ["what might go wrong"],
                    "fallback_strategies": ["what to do if something fails"]
                }
            }"""},
            {"role": "user", "content": f"Task: {self.task}\nContext: {json.dumps(self.metadata)}"}
        ]
        
        try:
            response = await self.llm.agenerate([messages])
            content = response.generations[0][0].message.content
            
            # Extract and parse JSON
            json_str = self._extract_json_from_response(content)
            analysis = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["task_type", "required_actions", "success_criteria"]
            if not all(field in analysis for field in required_fields):
                return self._create_fallback_analysis()
                
            return analysis
                
        except Exception as e:
            logger.warning(f"Error in task analysis: {e}")
            return self._create_fallback_analysis()

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response"""
        try:
            # Try to find content between triple backticks
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                # Find content between curly braces
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end] if start >= 0 and end > 0 else content
            return json_str
        except Exception as e:
            logger.warning(f"Error extracting JSON: {e}")
            return content

    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when parsing fails"""
        task_lower = self.task.lower()
        
        # Determine basic task type from keywords
        if any(word in task_lower for word in ["buy", "purchase", "order", "cart"]):
            task_type = "transaction"
        elif any(word in task_lower for word in ["monitor", "track", "watch", "check"]):
            task_type = "monitoring"
        elif any(word in task_lower for word in ["login", "signin", "account"]):
            task_type = "authentication"
        else:
            task_type = "research"
            
        return {
            "task_type": task_type,
            "required_actions": [
                {
                    "type": "navigate",
                    "description": "Go to relevant website",
                    "target": "determined from task",
                    "expected_result": "page loads successfully"
                }
            ],
            "required_data": {
                "inputs": ["task requirements"],
                "outputs": ["task results"]
            },
            "success_criteria": ["task completed successfully"],
            "error_handling": {
                "potential_issues": ["page not loading", "element not found"],
                "fallback_strategies": ["retry", "alternative approach"]
            }
        }

    async def _setup_context(self, context: BrowserContext) -> None:
        """Set up browser context with necessary configurations"""

        
        # Set up request interception if needed
        if self.metadata.get("intercept_requests"):
            await context.route("**/*", self._handle_request)
        
        # Restore authentication state if available
        auth_key = self.metadata.get("auth_key")
        if auth_key and auth_key in self._auth_state:
            await context.add_cookies(self._auth_state[auth_key])

    async def _handle_request(self, route: Any, request: Any) -> None:
        """Handle intercepted requests"""
        # Add custom headers if specified
        if "headers" in self.metadata:
            await route.continue_(headers=self.metadata["headers"])
        else:
            await route.continue_()

    def _create_task_description(self, task_analysis: Dict[str, Any]) -> str:
        """Create clear task description for any web task"""
        task_type = task_analysis["task_type"]
        actions = task_analysis["required_actions"]
        
        description = f"""Task: {self.task}
Type: {task_type.upper()}

Required Actions:
{chr(10).join(f'- {action["description"]} ({action["type"]})' for action in actions)}

Expected Results:
{chr(10).join(f'- {action["expected_result"]}' for action in actions)}

Success Criteria:
{chr(10).join(f'- {criterion}' for criterion in task_analysis["success_criteria"])}

Error Handling:
- Potential Issues: {', '.join(task_analysis["error_handling"]["potential_issues"])}
- Fallback Strategies: {', '.join(task_analysis["error_handling"]["fallback_strategies"])}

Instructions:
1. Execute each action in sequence
2. Verify expected results after each action
3. Collect required outputs and verify success criteria
4. Handle any errors using provided fallback strategies
5. Adapt to dynamic page changes and unexpected situations
"""
        return description

    async def _execute_task(self, context: BrowserContext, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web task with appropriate handling based on type"""
        try:
            task_type = task_analysis["task_type"]
            
            # Configure browser agent based on task type
            agent_config = {
                "task": self._create_task_description(task_analysis),
                "llm": self.llm,
                "browser_context": context,
            }
            
            # Add task-specific configurations
            if task_type == "monitoring":
                agent_config["wait_for_changes"] = True
            elif task_type == "transaction":
                agent_config["validate_inputs"] = True
                agent_config["secure_mode"] = True
                
            self.browser_agent = Agent(**agent_config)
            
            # Execute with retry and monitoring
            for attempt in range(self.max_retries):
                try:
                    if self.metadata.get("screenshot_before_actions"):
                        self.screenshots.append(await context.screenshot())
                        
                    result = await self.browser_agent.run()
                    
                    if self.metadata.get("screenshot_after_actions"):
                        self.screenshots.append(await context.screenshot())
                    
                    if result.is_done():
                        # Store visited URLs
                        self.visited_urls.extend(result.urls())
                        return {
                            "status": "success",
                            "task_type": task_type,
                            "result": result,
                            "actions_completed": result.action_names()
                        }
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    
            return {"error": "Task execution failed after retries"}
                
        except Exception as e:
            return {
                "error": str(e),
                "task_type": task_analysis.get("task_type", "unknown")
            }

    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure the browser automation result"""
        if "error" in result:
            return {
                "error": result["error"],
                "screenshots": self.screenshots,
                "visited_urls": self.visited_urls
            }

        try:
            browser_result = result.get("result")
            return {
                "status": "success",
                "task_type": result.get("task_type", "unknown"),
                "output": browser_result.final_result() if hasattr(browser_result, "final_result") else browser_result,
                "actions": result.get("actions_completed", []),
                "artifacts": {
                    "screenshots": self.screenshots,
                    "downloads": browser_result.downloads() if hasattr(browser_result, "downloads") else [],
                    "visited_urls": self.visited_urls
                },
                "metrics": {
                    "total_actions": len(result.get("actions_completed", [])),
                    "unique_urls": len(set(self.visited_urls))
                }
            }
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            return {
                "error": "Error processing result",
                "original_result": result,
                "screenshots": self.screenshots,
                "visited_urls": self.visited_urls
            }

    def _store_session_data(self, session_data: Dict[str, Any]) -> None:
        """Store session data for future use"""
        session_key = self.metadata.get("session_key", "default")
        self._session_data[session_key] = session_data

    @classmethod
    async def cleanup(cls) -> None:
        """Clean up browser resources"""
        if cls._shared_browser:
            await cls._shared_browser.close()
            cls._shared_browser = None
        cls._auth_state.clear()
        cls._session_data.clear()