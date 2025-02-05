import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent
import aiohttp
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ServiceIntegrationAgent(BaseAgent):
    """
    Agente que maneja la integración con servicios externos (Google Docs, Slack, etc.)
    """

    SERVICE_CAPABILITIES = {
        "document_creation": ["google_docs", "microsoft_word", "notion", "dropbox_paper"],
        "presentation": ["google_slides", "microsoft_powerpoint", "prezi"],
        "spreadsheet": ["google_sheets", "microsoft_excel", "airtable"],
        # ...
    }

    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0
        )
        self.service_sessions = {}
        self.action_history = []

    async def _execute(self) -> Dict[str, Any]:
        """Analiza la tarea para ver qué servicios se requieren, luego los ejecuta."""
        try:
            task_analysis = await self._analyze_task_requirements()
            services_status = await self._prepare_services(task_analysis["required_services"])
            if "error" in services_status:
                return services_status

            result = await self._execute_integration_workflow(task_analysis)
            self._record_execution(task_analysis, result)
            return {
                "status": "success",
                "result": result,
                "services_used": task_analysis["required_services"],
                "execution_flow": self.action_history
            }

        except Exception as e:
            logger.error(f"Error in ServiceIntegrationAgent: {e}", exc_info=True)
            return {
                "error": str(e)
            }

    async def _analyze_task_requirements(self) -> Dict[str, Any]:
        """Analyze task to determine required services and actions"""
        messages = [
            {"role": "system", "content": f"""Analyze the task and determine:
            1. Required services from: {json.dumps(self.SERVICE_CAPABILITIES, indent=2)}
            2. Required actions and their sequence
            3. Expected outputs and formats
            4. Integration requirements
            
            Return a structured analysis."""},
            {"role": "user", "content": f"Task: {self.task}"}
        ]

        response = await self.llm.agenerate([messages])
        analysis = self._parse_analysis(response.generations[0][0].message.content)
        
        return {
            "required_services": self._validate_services(analysis.get("services", [])),
            "action_sequence": analysis.get("actions", []),
            "expected_outputs": analysis.get("outputs", {}),
            "integration_requirements": analysis.get("integration", {})
        }

    async def _prepare_services(self, required_services: List[str]) -> Dict[str, Any]:
        """Prepare and validate all required service connections"""
        service_status = {}
        
        for service in required_services:
            try:
                # Initialize service connection
                connection = await self._initialize_service(service)
                
                # Validate service availability
                if await self._validate_service_connection(connection):
                    self.service_sessions[service] = connection
                    service_status[service] = "connected"
                else:
                    service_status[service] = "failed"
                    
            except Exception as e:
                service_status[service] = f"error: {str(e)}"
                
        # Check if we have all required services
        if not all(status == "connected" for status in service_status.values()):
            return {
                "error": "Not all required services could be connected",
                "status": service_status
            }
            
        return {"status": "all_services_ready", "details": service_status}

    async def _execute_integration_workflow(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the integration workflow based on task analysis"""
        results = {}
        action_sequence = task_analysis["action_sequence"]
        
        for action in action_sequence:
            try:
                # Execute action with appropriate service
                action_result = await self._execute_service_action(
                    action["service"],
                    action["action"],
                    action.get("parameters", {})
                )
                
                # Record action in history
                self.action_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action,
                    "status": "success" if "error" not in action_result else "error",
                    "result": action_result
                })
                
                results[f"{action['service']}_{action['action']}"] = action_result
                
                # Handle action dependencies
                if action.get("requires_previous_result"):
                    await self._handle_action_dependencies(action, results)
                    
            except Exception as e:
                logger.error(f"Error executing action {action}: {e}")
                results[f"{action['service']}_{action['action']}"] = {"error": str(e)}

        return results

    async def _execute_service_action(self, service: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific action on a service"""
        if service not in self.service_sessions:
            return {"error": f"Service {service} not initialized"}
            
        try:
            session = self.service_sessions[service]
            
            # Handle different types of actions
            if action == "create_document":
                return await self._create_document(session, parameters)
            elif action == "update_document":
                return await self._update_document(session, parameters)
            elif action == "share_document":
                return await self._share_document(session, parameters)
            elif action == "search":
                return await self._search_service(session, parameters)
            elif action == "post":
                return await self._post_content(session, parameters)
            elif action == "analyze":
                return await self._analyze_content(session, parameters)
            else:
                return {"error": f"Unsupported action: {action}"}
                
        except Exception as e:
            logger.error(f"Error executing {action} on {service}: {e}")
            return {"error": str(e)}

    async def _initialize_service(self, service: str) -> Any:
        """Initialize connection to a service"""
        # This would be replaced with actual service initialization
        session = aiohttp.ClientSession()
        # Add service-specific initialization here
        return session

    async def _validate_service_connection(self, connection: Any) -> bool:
        """Validate that a service connection is working"""
        try:
            # Add actual validation logic here
            return True
        except Exception as e:
            logger.error(f"Service validation error: {e}")
            return False

    def _validate_services(self, services: List[str]) -> List[str]:
        """Validate that requested services are supported"""
        validated_services = []
        for service in services:
            for capability, supported_services in self.SERVICE_CAPABILITIES.items():
                if service in supported_services:
                    validated_services.append(service)
                    break
        return validated_services

    def _parse_analysis(self, content: str) -> Dict[str, Any]:
        """Parse the LLM analysis into structured data"""
        try:
            # Try to find JSON content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except:
            pass
            
        # Fallback to manual parsing if JSON not found
        analysis = {"services": [], "actions": [], "outputs": {}, "integration": {}}
        
        for line in content.split("\n"):
            line = line.strip()
            if "service:" in line.lower():
                analysis["services"].append(line.split(":")[-1].strip())
            elif "action:" in line.lower():
                analysis["actions"].append({"action": line.split(":")[-1].strip()})
            elif "output:" in line.lower():
                analysis["outputs"]["format"] = line.split(":")[-1].strip()
                
        return analysis

    def _record_execution(self, task_analysis: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Record execution details for future optimization"""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "task_analysis": task_analysis,
            "result": result,
            "action_history": self.action_history
        }
        
        # This could be stored in a database or analytics system
        logger.info(f"Execution record: {json.dumps(execution_record, indent=2)}")

    async def cleanup(self) -> None:
        """Cleanup service connections"""
        for service, session in self.service_sessions.items():
            try:
                await session.close()
            except Exception as e:
                logger.error(f"Error closing {service} session: {e}")
