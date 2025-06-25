"""
Auro-PAI Platform Backend - Main FastAPI Application
====================================================

Central coordinator for the Auro-PAI platform, providing AI assistance
with context-aware, transparent, and controllable capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Dict, Any

from core.config import settings
from api.routes import chat, rag, tools, health
from services.llm_manager import LLMManager
from services.rag_service import RAGService
from services.tool_service import ToolService
from core.context_manager import ContextManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
llm_manager: LLMManager = None
rag_service: RAGService = None
tool_service: ToolService = None
context_manager: ContextManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global llm_manager, rag_service, tool_service, context_manager
    
    # Startup
    logger.info("Starting Auro-PAI Platform Backend...")
    
    try:
        # Initialize services
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        rag_service = RAGService()
        await rag_service.initialize()
        
        tool_service = ToolService()
        await tool_service.initialize()
        
        context_manager = ContextManager()
        await context_manager.initialize()
        
        # Store services in app state for access in routes
        app.state.llm_manager = llm_manager
        app.state.rag_service = rag_service
        app.state.tool_service = tool_service
        app.state.context_manager = context_manager
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Auro-PAI Platform Backend...")
    
    if context_manager:
        await context_manager.cleanup()
    if tool_service:
        await tool_service.cleanup()
    if rag_service:
        await rag_service.cleanup()
    if llm_manager:
        await llm_manager.cleanup()
    
    logger.info("All services cleaned up successfully")


# Create FastAPI application
app = FastAPI(
    title="Auro-PAI Platform Backend",
    description="AI assistance platform with local LLM, RAG, and tool integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
app.include_router(tools.router, prefix="/api/v1", tags=["tools"])

# Add convenience routes without /api prefix for easier access
app.include_router(health.router, prefix="/v1", tags=["health-simple"])
app.include_router(chat.router, prefix="/v1", tags=["chat-simple"])
app.include_router(rag.router, prefix="/v1", tags=["rag-simple"])
app.include_router(tools.router, prefix="/v1", tags=["tools-simple"])


@app.get("/")
async def root():
    """Root endpoint with platform information."""
    return {
        "name": "Auro-PAI Platform Backend",
        "version": "1.0.0",
        "description": "AI assistance platform for Average Joys with local LLM integration",
        "status": "running"
    }


@app.get("/api/v1/status")
async def status():
    """Get overall platform status."""
    try:
        status_info = {
            "platform": "healthy",
            "services": {}
        }
        
        # Check LLM service
        if app.state.llm_manager:
            llm_status = await app.state.llm_manager.health_check()
            status_info["services"]["llm"] = llm_status
        
        # Check RAG service
        if app.state.rag_service:
            rag_status = await app.state.rag_service.health_check()
            status_info["services"]["rag"] = rag_status
        
        # Check Tool service
        if app.state.tool_service:
            tool_status = await app.state.tool_service.health_check()
            status_info["services"]["tools"] = tool_status
        
        return status_info
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
