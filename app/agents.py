from app.schemas import Trace, ToolCall
from app.tracer import add_step, add_toolcall
from app.retriever import QdrantRetriever
from app.llm_provider import OllamaBackend
from app.utils.timeoutUtils import timeit_async
from app.utils.config import settings
from app.tools import tool_registry
from datetime import datetime
import json
import re

class Agent:
    def __init__(self, trace: Trace):
        self.trace = trace
        self.retriever = QdrantRetriever()
        self.llm = None
        if settings.MODEL_BACKEND == "llama":
            self.llm = OllamaBackend() 
        elif settings.MODEL_BACKEND == "hf" and settings.HF_API_URL:
            from app.llm_provider import HFBackend
            self.llm = HFBackend(settings.HF_API_URL)
        else:
            raise RuntimeError("No model backend configured")

    async def decide(self, user_text: str) -> dict:
        """
        Decide what to do:
        - RESPOND directly
        - SEARCH the knowledge base
        - CALL_TOOL for system utilities (like date_diff)
        """
        add_step(self.trace, "agent", f"Deciding action for query: {user_text}")

        # detect tool intent heuristically first 
        if "date" in user_text.lower() and "between" in user_text.lower():
            return {"action": "CALL_TOOL", "tool_name": "date_diff", "why": "User asked to calculate date difference"}

        if settings.MODEL_BACKEND == "llama":
            tools_context = "\n".join([
                f"- {t['name']}: {t['description']}"
                for t in tool_registry.list()
            ])
            messages = [
                {"role": "system", "content": f"You are an agent that can RESPOND, SEARCH, or CALL_TOOL based on user query.\nAvailable tools:\n{tools_context}"},
                {"role": "user", "content": f"Query: {user_text}\nReturn JSON: {{'action': 'RESPOND'|'SEARCH'|'CALL_TOOL', 'tool_name': optional, 'why': reason}}"}
            ]
            resp = await self.llm.chat(messages)
            content = resp["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                return data
            except Exception:
                # fallback
                if any(k in user_text.lower() for k in ["find", "search", "who", "where"]):
                    return {"action": "SEARCH", "why": "keyword heuristic"}
                return {"action": "RESPOND", "why": "default fallback"}
        else:
            prompt = f"Decide next action for: {user_text}\nReturn JSON object with keys 'action', 'tool_name' (if applicable), and 'why'."
            data = await self.llm.chat_json(prompt)
            return data

    async def call_tool(self, tool_name: str, args: dict):
        """Run a system tool and log trace."""
        tool = tool_registry.get(tool_name)
        started = datetime.now()
        tc = ToolCall(
            name=tool_name,
            type=tool.type,
            args=args,
            started_at=started,
            metadata={}
        )
        try:
            result = await tool.run(args)
            tc.finished_at = datetime.now()
            tc.duration_ms = int((tc.finished_at - started).total_seconds() * 1000)
            tc.result_preview = json.dumps(result)
            add_toolcall(self.trace, tc)
            add_step(self.trace, "tool", f"Executed {tool_name} with args {args}, result={result}")
            return result
        except Exception as e:
            tc.error = str(e)
            add_toolcall(self.trace, tc)
            add_step(self.trace, "tool", f"Tool {tool_name} error: {e}")
            return {"error": str(e)}

    async def call_retrieval(self, query: str, top_k: int):
        started = datetime.now()
        tc = ToolCall(
            name="qdrant_retrieval",
            type="retrieval",
            args={"query": query, "top_k": top_k},
            started_at=started,
            metadata={}
        )
        try:
            (hits, dur) = await timeit_async(self.retriever.search, query, top_k)
            tc.finished_at = datetime.now()
            tc.duration_ms = dur
            tc.result_preview = json.dumps([
                {"id": h["id"], "score": h["score"], "text": h["text"][:200]} for h in hits
            ])[:800]
            tc.metadata = {"top_k": top_k}
            add_toolcall(self.trace, tc)
            add_step(self.trace, "tool", f"Retrieved {len(hits)} docs (preview): {tc.result_preview}")
            return hits
        except Exception as e:
            tc.error = str(e)
            add_toolcall(self.trace, tc)
            add_step(self.trace, "tool", f"Retrieval error: {e}")
            return []

    async def synthesize(self, user_text: str, contexts):
        add_step(self.trace, "agent", f"Synthesizing answer with {len(contexts)} contexts.")
        if settings.MODEL_BACKEND == "llama":
            ctx_text = "\n\n".join([f"[doc {i+1} score={c['score']}] {c['text']}" for i, c in enumerate(contexts)])
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the context below to answer the question."},
                {"role": "user", "content": f"Context:\n{ctx_text}\n\nQuestion:\n{user_text}\nProvide a grounded answer with provenance (mention docs used)."}
            ]
            out = await self.llm.chat(messages)
            ans = out["choices"][0]["message"]["content"]
            add_step(self.trace, "agent", "Generated final answer.")
            self.trace.final_answer = ans
            self.trace.provenance = "Based on docs: " + ", ".join([str(c['id']) for c in contexts[:3]])
            return ans
        else:
            prompt = f"Context:\n{contexts}\n\nQuestion:{user_text}\nReturn JSON: {{'answer':..., 'provenance':...}}"
            data = await self.llm.chat_json(prompt)
            self.trace.final_answer = data.get("answer")
            self.trace.provenance = data.get("provenance")
            return self.trace.final_answer

    async def run(self, user_text: str, top_k: int = None):
        t0 = datetime.now()
        top_k = top_k or 5
        add_step(self.trace, "user", user_text)

        decision = await self.decide(user_text)
        add_step(self.trace, "agent", f"Decision: {decision}")
        contexts = []

    
        if decision.get("action") == "SEARCH":
            contexts = await self.call_retrieval(user_text, top_k)
            ans = await self.synthesize(user_text, contexts)

        elif decision.get("action") == "CALL_TOOL":
            tool_name = decision.get("tool_name")
            # extract dates if the tool is date_diff
            if tool_name == "date_diff":
                date_matches = re.findall(r"\d{4}-\d{2}-\d{2}", user_text)
                if len(date_matches) >= 2:
                    args = {"a": date_matches[0], "b": date_matches[1]}
                else:
                    args = {"a": "2025-01-01", "b": "2025-01-10"} 
                result = await self.call_tool(tool_name, args)
                ans = f"The number of days between {args['a']} and {args['b']} is {result.get('days', 'unknown')}."
            else:
                ans = f"Tool '{tool_name}' executed, but I have no display handler yet."

        else:  
            ans = await self.synthesize(user_text, [])

        self.trace.total_duration_ms = int((datetime.now() - t0).total_seconds() * 1000)
        self.trace.final_answer = ans
        return self.trace
