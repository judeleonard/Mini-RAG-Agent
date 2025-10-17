from .schemas import Trace, Step, ToolCall
from datetime import datetime
from typing import List
import uuid

def new_trace(query: str) -> Trace:
    return Trace(query=query, received_at=datetime.now(), steps=[], tools_used=[])

def add_step(trace: Trace, role: str, content: str):
    trace.steps.append(Step(step_id=str(uuid.uuid4()), role=role, content=content, timestamp=datetime.now()))

def add_toolcall(trace: Trace, toolcall: ToolCall):
    trace.tools_used.append(toolcall)
