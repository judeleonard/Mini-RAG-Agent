from typing import Dict, Any, Protocol, List, Optional
from app.schemas import ToolCall, ToolType
from datetime import datetime
import uuid

class Tool(Protocol):
    name: str
    type: ToolType
    description: Optional[str] = None
    async def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ...

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list(self) -> List[Dict[str, Any]]:
        """Return list of registered tools with metadata."""
        tools_info = []
        for t in self._tools.values():
            tools_info.append({
                "name": t.name,
                "type": t.type.value,
                "description": getattr(t, "description", None),
            })
        return tools_info

tool_registry = ToolRegistry()

# sample system tool: date calculator
# this can also be extended as a separate module for a system requiring more than one tool
class DateDiffTool:
    name = "date_diff"
    type = ToolType.system
    description = "Calculate the number of days between two dates (format: YYYY-MM-DD)."
    
    async def run(self, args):
        from datetime import datetime
        fmt = args.get("format", "%Y-%m-%d")
        d1 = datetime.strptime(args["a"], fmt)
        d2 = datetime.strptime(args["b"], fmt)
        delta = abs((d2 - d1).days)
        return {"days": delta}

# Register data calclator tools 
tool_registry.register(DateDiffTool())
