from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from contextlib import asynccontextmanager
from app.tracer import new_trace
from app.agents import Agent
from app.utils.config import settings
from app.tools import tool_registry
from app.utils.logger import error_logger
import uvicorn


logger = error_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
    )
    yield
    logger.info("application_shutdown")


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

@app.get("/health")
async def health():
    logger.info("health status endpoint called")
    return {"status": "ok", "model_backend": settings.MODEL_BACKEND}

@app.get("/tools")
async def list_tools():
    """List all registered tools."""

    logger.info("tools endpoint called")
    tools = tool_registry.list()
    return {"count": len(tools), "tools": tools}

@app.get("/", tags=["Root"])
async def read_root():
    logger.info("root endpoint called")
    return {"message": "Minimal RAG Agent Service is Live!"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    trace = new_trace(req.text)
    agent = Agent(trace)
    try:
        trace = await agent.run(req.text, top_k=(settings.DEFAULT_TOP_K))
        return {"trace": trace}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
