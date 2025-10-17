from langchain_ollama import ChatOllama
import httpx
import re, json

class OllamaBackend:
    """Wrapper for ChatOllama."""
    def __init__(self):
        self.model = ChatOllama(model="llama3.1", temperature=0.2)

    async def chat(self, messages):
        # Convert message list into a single prompt string
        prompt_parts = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        prompt = "\n".join(prompt_parts)

        response = await self.model.ainvoke(prompt)
        return {"choices": [{"message": {"content": response.content}}]}



# HF backend fallback
class HFBackend:
    def __init__(self, api_url: str):
        self.api_url = api_url

    async def chat_json(self, prompt: str):
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(self.api_url, json={"inputs": prompt})
            r.raise_for_status()
            text = r.text
            m = re.search(r"(\{.*\})", text, re.S)
            if not m:
                raise ValueError("No JSON found in HF model output")
            return json.loads(m.group(1))