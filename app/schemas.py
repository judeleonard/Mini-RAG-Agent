from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from enum import Enum
from datetime import datetime

class ToolType(str, Enum):
    retrieval = "retrieval"
    api = "api"
    system = "system"

class ToolCall(BaseModel):
    name: str
    type: ToolType
    args: Dict[str, Any]
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    result_preview: Optional[str] = None 
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Step(BaseModel):
    step_id: str
    role: str
    content: str
    timestamp: datetime

class Trace(BaseModel):
    query: str
    received_at: datetime
    steps: List[Step] = []
    tools_used: List[ToolCall] = []
    final_answer: Optional[str] = None
    provenance: Optional[str] = None
    total_duration_ms: Optional[int] = None

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    trace: Trace
