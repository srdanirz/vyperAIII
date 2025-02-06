# agents/service_integration_agent.py

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
    Agente que maneja integración con servicios externos (Google, Slack, etc.).
    """
    SERVICE_CAPABILITIES = {
        "document_creation": ["google_docs", "microsoft_word", "notion", "dropbox_paper"],
        "presentation": ["google_slides","microsoft_powerpoint","prezi"],
        "spreadsheet": ["google_sheets","microsoft_excel","airtable"]
    }

    def __init__(
        self,
        task: str,
        openai_api_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        shared_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(task, openai_api_key, metadata, shared_data)
        self.llm=ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0
        )
        self.service_sessions={}
        self.action_history=[]

    async def _execute(self)->Dict[str,Any]:
        try:
            analysis=await self._analyze_task_requirements()
            status=await self._prepare_services(analysis["required_services"])
            if "error" in status:
                return status

            result=await self._execute_integration_workflow(analysis)
            self._record_execution(analysis,result)
            return {
                "status":"success",
                "result":result,
                "services_used":analysis["required_services"],
                "execution_flow":self.action_history
            }
        except Exception as e:
            logger.error(f"ServiceIntegrationAgent error: {e}", exc_info=True)
            return {"error": str(e)}

    async def _analyze_task_requirements(self)->Dict[str,Any]:
        msg=[
            {
                "role":"system",
                "content":f"Analyze the task and determine needed services from:\n{json.dumps(self.SERVICE_CAPABILITIES, indent=2)}"
            },
            {
                "role":"user",
                "content":f"Task: {self.task}"
            }
        ]
        resp=await self.llm.agenerate([msg])
        data=self._parse_analysis(resp.generations[0][0].message.content)
        return {
            "required_services":self._validate_services(data.get("services",[])),
            "action_sequence":data.get("actions",[]),
            "expected_outputs":data.get("outputs",{}),
            "integration_requirements":data.get("integration",{})
        }

    async def _prepare_services(self, required_services:List[str])->Dict[str,Any]:
        service_status={}
        for svc in required_services:
            try:
                conn=await self._initialize_service(svc)
                if await self._validate_service_connection(conn):
                    self.service_sessions[svc]=conn
                    service_status[svc]="connected"
                else:
                    service_status[svc]="failed"
            except Exception as e:
                service_status[svc]=f"error: {str(e)}"

        if not all(st=="connected" for st in service_status.values()):
            return{
                "error":"Not all required services connected",
                "status":service_status
            }
        return {"status":"all_services_ready","details":service_status}

    async def _execute_integration_workflow(self,analysis:Dict[str,Any])->Dict[str,Any]:
        results={}
        sequence=analysis["action_sequence"]
        for act in sequence:
            try:
                res=await self._execute_service_action(
                    act["service"],
                    act["action"],
                    act.get("parameters",{})
                )
                self.action_history.append({
                    "timestamp":datetime.now().isoformat(),
                    "action":act,
                    "status":"success" if "error" not in res else "error",
                    "result":res
                })
                results[f"{act['service']}_{act['action']}"]=res
                if act.get("requires_previous_result"):
                    await self._handle_action_dependencies(act, results)
            except Exception as e:
                logger.error(f"Action {act} error: {e}")
                results[f"{act['service']}_{act['action']}"]={"error":str(e)}
        return results

    async def _execute_service_action(self, service:str, action:str, params:Dict[str,Any])->Dict[str,Any]:
        if service not in self.service_sessions:
            return{"error":f"Service {service} not in session"}

        try:
            sess=self.service_sessions[service]
            # Example switch
            if action=="create_document":
                return await self._create_document(sess, params)
            elif action=="update_document":
                return await self._update_document(sess, params)
            else:
                return {"error":f"Unsupported action: {action}"}
        except Exception as e:
            logger.error(f"execute_service_action error: {e}")
            return {"error":str(e)}

    async def _initialize_service(self, svc:str)->Any:
        return aiohttp.ClientSession()

    async def _validate_service_connection(self, connection:Any)->bool:
        return True

    def _validate_services(self, services:List[str])->List[str]:
        validated=[]
        for s in services:
            for cap, supported in self.SERVICE_CAPABILITIES.items():
                if s in supported:
                    validated.append(s)
                    break
        return validated

    def _parse_analysis(self, content:str)->Dict[str,Any]:
        try:
            start=content.find('{')
            end=content.rfind('}')+1
            if start>=0 and end>start:
                return json.loads(content[start:end])
        except:
            pass
        return {"services":[],"actions":[],"outputs":{},"integration":{}}

    def _record_execution(self,analysis:Dict[str,Any], result:Dict[str,Any]):
        ex_record={
            "timestamp":datetime.now().isoformat(),
            "task_analysis":analysis,
            "result":result,
            "action_history":self.action_history
        }
        logger.info(f"Execution record: {json.dumps(ex_record,indent=2)}")

    async def cleanup(self)->None:
        for svc,sess in self.service_sessions.items():
            try:
                await sess.close()
            except:
                pass
