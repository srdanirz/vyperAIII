# agents/agent_communication.py
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
import asyncio

logger = logging.getLogger(__name__)

class AgentCommunicationSystem:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo",
            temperature=0.7
        )
        self.message_queue = asyncio.Queue()
        self.priorities = {
            "System": 0,
            "Team": 1
        }

    async def send_message(self, from_agent: str, content: str, priority: Optional[int] = None) -> None:
        """Send a message to the communication system"""
        if priority is None:
            priority = self.priorities.get(from_agent, 2)
            
        await self.message_queue.put({
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "from_agent": from_agent,
            "content": self._clean_content(content)
        })

    def _clean_content(self, content: str) -> str:
        """Clean content of special markers"""
        return (content
                .replace('[', '')
                .replace(']', '')
                .replace('**', '')
                .replace('*', '')
                .strip())

    async def generate_dialogue(self, task: str, participants: List[str]) -> List[Dict[str, str]]:
        """Generate a dialogue showing agent collaboration"""
        messages = [
            {"role": "system", "content": f"""Task: {task}
Team members: {', '.join(participants)}

Generate a solution where each team member contributes their expertise to create exactly what was requested.
Show the actual development and creation of the solution, not just planning."""},
            {"role": "user", "content": task}
        ]
        
        response = await self.llm.agenerate([messages])
        
        dialogue = []
        current_speaker = None
        current_content = []
        
        for line in response.generations[0][0].message.content.split('\n'):
            line = self._clean_content(line)
            if not line:
                continue
                
            if ':' in line:
                if current_speaker and current_content:
                    dialogue.append({
                        "speaker": current_speaker,
                        "content": ' '.join(current_content)
                    })
                    current_content = []
                
                parts = line.split(':', 1)
                current_speaker = parts[0].strip()
                if len(parts) > 1:
                    current_content = [parts[1].strip()]
            elif current_speaker:
                current_content.append(line)
        
        if current_speaker and current_content:
            dialogue.append({
                "speaker": current_speaker,
                "content": ' '.join(current_content)
            })
        
        return dialogue

    async def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in priority order"""
        messages = []
        while not self.message_queue.empty():
            messages.append(await self.message_queue.get())
        messages.sort(key=lambda x: (x["priority"], x["timestamp"]))
        return messages