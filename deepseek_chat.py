# deepseek_chat.py

import aiohttp
import ssl
import certifi
from typing import Any, Dict, List

class DeepSeekChat:
    """
    Clase para invocar la API de DeepSeek en modo "chat", de forma
    compatible con la interfaz ChatOpenAI (método agenerate).
    
    - Usa un contexto SSL con certifi para evitar errores de certificado.
    - model (p.e. "deepseek-chat" o "deepseek-reasoner").
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        # Ajustamos la URL base según la documentación de DeepSeek
        self.base_url = "https://api.deepseek.com/v1"

        # Creamos un contexto SSL que confíe en las CAs de certifi
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.load_verify_locations(cafile=certifi.where())

        # Validamos que tengamos API key
        if not api_key:
            raise ValueError("DeepSeek API key is required")

    async def agenerate(self, messages_batch: List[List[Dict[str, str]]]) -> Any:
        """
        Emula .agenerate(batch_of_messages), igual que ChatOpenAI.
          - Normalmente se llama con un solo item en messages_batch.
          - Retorna un objeto con .generations[0][0].message.content
        """
        if not messages_batch:
            raise ValueError("No messages to process in agenerate().")

        # Tomamos la primera conversación
        messages = messages_batch[0]

        # Convertimos a formato "OpenAI-like"
        openai_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Creamos un conector que use nuestro ssl_context con certifi
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"DeepSeek API error: {error_text}")
                        raise RuntimeError(f"DeepSeek API error {resp.status}: {error_text}")
                    data = await resp.json()

            # Validamos la respuesta
            if not data.get("choices"):
                raise ValueError("No 'choices' in DeepSeek response")

            # Extraemos el texto principal
            content = data["choices"][0]["message"]["content"]

            # Construimos un objeto con la forma: .generations[0][0].message.content
            class MockMessage:
                def __init__(self, c: str):
                    self.content = c

            class MockGen:
                def __init__(self, c: str):
                    self.message = MockMessage(c)

            class MockResult:
                def __init__(self, c: str):
                    self.generations = [[MockGen(c)]]

            return MockResult(content)
            
        except Exception as e:
            logger.error(f"Error in DeepSeek API call: {str(e)}")
            raise