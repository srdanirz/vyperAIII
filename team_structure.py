import logging
import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from agents.agent_communication import AgentCommunicationSystem
from agent_manager import AgentManager

logger = logging.getLogger(__name__)

class TeamStructure:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key, 
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        self.comm_system = AgentCommunicationSystem(openai_api_key)
        self.agent_manager = AgentManager(openai_api_key)
        self.openai_api_key = openai_api_key
        self.cache = {}

    async def process_request(self, request: str) -> Dict[str, Any]:
        try:
            await self.comm_system.send_message("System", "🤔 Analizando solicitud...")
            
            task_analysis = await self._analyze_task_requirements(request)
            required_agents = task_analysis["required_agents"]
            
            await self.comm_system.send_message("Tech Director", f"Para esta tarea necesitaremos: {', '.join(required_agents)}")

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "task": request,
                "requirements": task_analysis
            }

            workflow = await self._generate_workflow(required_agents, request)
            await self._announce_team_and_plan(required_agents, workflow)
            
            result = await self.agent_manager.execute(
                task=request,
                metadata=metadata,
                workflow=workflow
            )

            if "error" not in result:
                await self._generate_completion_dialogue(result, required_agents)

            return {
                "status": "success" if "error" not in result else "error",
                "messages": await self.comm_system.get_messages(),
                "result": result,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            await self.comm_system.send_message("System", f"❌ Error: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _analyze_task_requirements(self, request: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": """Analiza la tarea y determina los agentes necesarios.
            Agentes disponibles:
            - browser: Navegación web y automatización
            - contentgeneration: Creación de contenido (documentos, presentaciones)
            - research: Investigación y síntesis
            - analysis: Análisis de datos
            - validation: Verificación de calidad
            - dataprocessing: Procesamiento de datos"""},
            {"role": "user", "content": f"Tarea a analizar: {request}"}
        ]

        response = await self.llm.agenerate([messages])
        content = response.generations[0][0].message.content
        
        agents = self._determine_required_agents(request)
        
        return {
            "required_agents": list(agents),
            "task_type": self._determine_task_type(request),
            "initial_analysis": content
        }

    def _determine_required_agents(self, request: str) -> set:
        request_lower = request.lower()
        agents = set()

        task_patterns = {
            "browser": ["navega", "web", "visita", "url", "página"],
            "contentgeneration": ["powerpoint", "documento", "presentacion", "cancion", "escribe", "crea"],
            "research": ["investiga", "busca", "analiza", "información"],
            "analysis": ["analiza", "compara", "evalúa"],
            "dataprocessing": ["procesa", "datos", "csv", "excel"]
        }

        for agent, patterns in task_patterns.items():
            if any(pattern in request_lower for pattern in patterns):
                agents.add(agent)

        # Siempre añadir validation para control de calidad
        agents.add("validation")
        
        # Si se necesita contenido pero no hay research, añadir research
        if "contentgeneration" in agents and "research" not in agents:
            agents.add("research")

        return agents

    def _determine_task_type(self, request: str) -> str:
        request_lower = request.lower()
        task_types = {
            "presentation": ["powerpoint", "presentacion", "slides"],
            "document": ["documento", "word", "texto", "escribe"],
            "web_automation": ["navega", "web", "visita", "url"],
            "creative": ["cancion", "musica", "poema", "historia"],
            "data_analysis": ["analiza", "datos", "csv", "excel"]
        }

        for task_type, patterns in task_types.items():
            if any(pattern in request_lower for pattern in patterns):
                return task_type
        return "general"

    async def _generate_workflow(self, required_agents: List[str], request: str) -> List[Dict[str, Any]]:
        workflow = []
        dependencies = {
            "contentgeneration": ["research"],
            "analysis": ["research", "dataprocessing"],
            "validation": ["contentgeneration", "browser", "analysis"]
        }

        step = 1
        processed_agents = set()

        # Procesar agentes en orden de dependencias
        while len(processed_agents) < len(required_agents):
            for agent in required_agents:
                if agent in processed_agents:
                    continue
                
                agent_deps = dependencies.get(agent, [])
                required_deps = [dep for dep in agent_deps if dep in required_agents]
                
                if all(dep in processed_agents for dep in required_deps):
                    workflow.append({
                        "step": step,
                        "agent": agent,
                        "agent_type": agent,
                        "task": self._generate_agent_task(agent, request),
                        "requires": required_deps
                    })
                    processed_agents.add(agent)
                    step += 1

        return workflow

    def _generate_agent_task(self, agent: str, request: str) -> str:
        task_templates = {
            "research": "Investigar y recopilar información sobre: {}",
            "browser": "Navegar y ejecutar acciones web para: {}",
            "contentgeneration": "Generar contenido para: {}",
            "analysis": "Analizar datos relacionados con: {}",
            "validation": "Validar la calidad y completitud de: {}",
            "dataprocessing": "Procesar datos para: {}"
        }
        return task_templates.get(agent, "Ejecutar tarea para: {}").format(request)

    async def _announce_team_and_plan(self, required_agents: List[str], workflow: List[Dict[str, Any]]) -> None:
        await self.comm_system.send_message("Research Director", "📚 Equipo asignado y listo:")
        
        for agent in required_agents:
            role = agent.replace("_", " ").title()
            await self.comm_system.send_message(role, f"¡Presente! Listo para colaborar")

        await self.comm_system.send_message("Tech Director", "🔄 Plan de trabajo:")
        for step in workflow:
            dependencies = f" (requiere: {', '.join(step['requires'])})" if step['requires'] else ""
            await self.comm_system.send_message(
                step["agent"].title(),
                f"Paso {step['step']}: {step['task']}{dependencies}"
            )

    async def _generate_completion_dialogue(self, result: Dict[str, Any], agents: List[str]) -> None:
        success = "error" not in result
        status = "✅ ¡Tarea completada con éxito!" if success else "❌ Se encontraron algunos problemas..."
        await self.comm_system.send_message("Tech Director", status)
        
        if success:
            for agent in agents:
                agent_result = result.get("result", {}).get(agent, {})
                if agent_result:
                    await self.comm_system.send_message(
                        agent.title(),
                        f"Completé mi parte: {agent_result.get('task', 'la tarea asignada')}"
                    )