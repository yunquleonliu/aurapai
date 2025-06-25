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
from api.routes import chat, knowledge, tools # Import the routers
from services.llm_manager import LLMManager
from services.rag_service import RAGService
from services.tool_service import ToolService
from services.image_generation_service import ImageGenerationService
from core.context_manager import ContextManager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
image_generation_service: ImageGenerationService = None
context_manager: ContextManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global llm_manager, rag_service, tool_service, image_generation_service, context_manager
    
    # Startup
    logger.info("Starting Aura-PAI Platform Backend...")
    
    try:
        # Initialize services
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        rag_service = RAGService()
        await rag_service.initialize()
        
        image_generation_service = ImageGenerationService()
        await image_generation_service.initialize()

        tool_service = ToolService(llm_manager=llm_manager)
        await tool_service.initialize()
        
        context_manager = ContextManager(rag_service)
        await context_manager.initialize()
        
        # Store services in app state for access in routes
        app.state.llm_manager = llm_manager
        app.state.rag_service = rag_service
        app.state.tool_service = tool_service
        app.state.image_generation_service = image_generation_service
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
    if image_generation_service:
        await image_generation_service.cleanup()
    if tool_service:
        await tool_service.cleanup()
    if rag_service:
        await rag_service.cleanup()
    if llm_manager:
        await llm_manager.cleanup()
    
    logger.info("All services cleaned up successfully")


# Create FastAPI application
app = FastAPI(
    title="Aura-PAI Platform Backend",
    description="Central coordinator for the Auro-PAI platform.",
    version="0.1.0",
    lifespan=lifespan
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(knowledge.router, tags=["Knowledge"])
app.include_router(tools.router, prefix="/api/v1", tags=["Tools"])


@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse('static/index.html')

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse('static/favicon.ico')

# Main entry point
if __name__ == "__main__":
    logger.info(f"Starting server, listening on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.RELOAD
    )
