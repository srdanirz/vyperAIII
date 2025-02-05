import logging
import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from agents.agent_communication import AgentCommunicationSystem
from agent_manager import AgentManager

logger = logging.getLogger(__name__)

class TeamStructure:
    """
    Orquesta la solicitud del usuario. Determina los agentes requeridos, 
    genera un workflow y delega en AgentManager la ejecución.
    """
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
        """
        Punto de entrada principal: analiza la tarea, determina agentes,
        arma el workflow y delega la ejecución en el AgentManager.
        """
        try:
            await self.comm_system.send_message("System", "🤔 Analizando solicitud...")

            task_analysis = await self._analyze_task_requirements(request)
            required_agents = task_analysis["required_agents"]
            
            await self.comm_system.send_message("Tech Director",
                                                f"Para esta tarea necesitaremos: {', '.join(required_agents)}")

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
            logger.error(f"Error processing request: {e}", exc_info=True)
            await self.comm_system.send_message("System", f"❌ Error: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _analyze_task_requirements(self, request: str) -> Dict[str, Any]:
        """Analiza la solicitud y determina agentes requeridos."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Analiza la tarea y determina los agentes necesarios.\n"
                    "Agentes disponibles:\n"
                    "- browser\n- contentgeneration\n- research\n- analysis\n- validation\n- dataprocessing"
                )
            },
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
        agents.add("validation")
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
        for ttype, patterns in task_types.items():
            if any(p in request_lower for p in patterns):
                return ttype
        return "general"

    async def _generate_workflow(self, required_agents: List[str], request: str) -> List[Dict[str, Any]]:
        """Crea una lista de pasos con dependencias básicas."""
        workflow = []
        dependencies = {
            "contentgeneration": ["research"],
            "analysis": ["research", "dataprocessing"],
            "validation": ["contentgeneration", "browser", "analysis"]
        }
        step = 1
        processed = set()
        while len(processed) < len(required_agents):
            for agent in required_agents:
                if agent in processed:
                    continue
                agent_deps = dependencies.get(agent, [])
                req_deps = [dep for dep in agent_deps if dep in required_agents]
                if all(d in processed for d in req_deps):
                    workflow.append({
                        "step": step,
                        "agent": agent,
                        "agent_type": agent,
                        "task": self._generate_agent_task(agent, request),
                        "requires": req_deps
                    })
                    processed.add(agent)
                    step += 1
        return workflow

    def _generate_agent_task(self, agent: str, request: str) -> str:
        templates = {
            "research": "Investigar sobre: {}",
            "browser": "Navegar y ejecutar acciones para: {}",
            "contentgeneration": "Generar contenido para: {}",
            "analysis": "Analizar datos relacionados con: {}",
            "validation": "Validar resultados de: {}",
            "dataprocessing": "Procesar datos para: {}"
        }
        return templates.get(agent, "Ejecutar tarea para: {}").format(request)

    async def _announce_team_and_plan(self, required_agents: List[str], workflow: List[Dict[str, Any]]) -> None:
        await self.comm_system.send_message("Research Director", "📚 Equipo asignado y listo:")
        for agent in required_agents:
            await self.comm_system.send_message(agent.title(), f"¡Presente! Listo para colaborar")
        await self.comm_system.send_message("Tech Director", "🔄 Plan de trabajo:")
        for step in workflow:
            deps = f"(requiere: {', '.join(step['requires'])})" if step['requires'] else ""
            await self.comm_system.send_message(
                step["agent"].title(),
                f"Paso {step['step']}: {step['task']} {deps}"
            )

    async def _generate_completion_dialogue(self, result: Dict[str, Any], agents: List[str]) -> None:
        success = "error" not in result
        status_msg = "✅ ¡Tarea completada con éxito!" if success else "❌ Se encontraron problemas."
        await self.comm_system.send_message("Tech Director", status_msg)
        if success:
            for agent in agents:
                agent_result = result.get("result", {}).get(agent, {})
                if agent_result:
                    await self.comm_system.send_message(
                        agent.title(),
                        f"Completé mi parte: {agent_result.get('task', 'tarea asignada')}"
                    )
