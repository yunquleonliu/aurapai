"""
Health check API routes for Auro-PAI Platform
=============================================

Provides health monitoring and status endpoints.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status model."""
    status: str
    timestamp: str
    uptime_seconds: float
    services: Dict[str, Any]


class ServiceHealth(BaseModel):
    """Individual service health model."""
    name: str
    status: str
    details: Dict[str, Any]


# Track startup time for uptime calculation
startup_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check(req: Request):
    """Comprehensive health check for all services."""
    try:
        current_time = time.time()
        uptime = current_time - startup_time
        
        services = {}
        overall_status = "healthy"
        
        # Check LLM Manager
        try:
            llm_health = await req.app.state.llm_manager.health_check()
            services["llm"] = llm_health
            if "unhealthy" in str(llm_health.get("status", "")).lower():
                overall_status = "degraded"
        except Exception as e:
            services["llm"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        # Check RAG Service
        try:
            rag_health = await req.app.state.rag_service.health_check()
            services["rag"] = rag_health
            if rag_health.get("status") != "healthy":
                overall_status = "degraded"
        except Exception as e:
            services["rag"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        # Check Tool Service
        try:
            tool_health = await req.app.state.tool_service.health_check()
            services["tools"] = tool_health
            if tool_health.get("status") != "healthy":
                overall_status = "degraded"
        except Exception as e:
            services["tools"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        # Check Context Manager
        try:
            context_stats = req.app.state.context_manager.get_session_stats()
            services["context"] = {
                "status": "healthy",
                "active_sessions": context_stats["total_sessions"],
                "max_sessions": context_stats["max_sessions"]
            }
        except Exception as e:
            services["context"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        health_status = HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=round(uptime, 2),
            services=services
        )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthStatus(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=round(time.time() - startup_time, 2),
            services={"error": str(e)}
        )


@router.get("/health/simple")
async def simple_health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "auro-pai-backend"
    }


@router.get("/health/llm")
async def llm_health_check(req: Request):
    """Health check specifically for LLM service."""
    try:
        health = await req.app.state.llm_manager.health_check()
        return ServiceHealth(
            name="llm_manager",
            status=health.get("status", "unknown"),
            details=health
        )
    except Exception as e:
        logger.error(f"LLM health check error: {e}")
        return ServiceHealth(
            name="llm_manager",
            status="error",
            details={"error": str(e)}
        )


@router.get("/health/rag")
async def rag_health_check(req: Request):
    """Health check specifically for RAG service."""
    try:
        health = await req.app.state.rag_service.health_check()
        return ServiceHealth(
            name="rag_service",
            status=health.get("status", "unknown"),
            details=health
        )
    except Exception as e:
        logger.error(f"RAG health check error: {e}")
        return ServiceHealth(
            name="rag_service",
            status="error",
            details={"error": str(e)}
        )


@router.get("/health/tools")
async def tools_health_check(req: Request):
    """Health check specifically for tools service."""
    try:
        health = await req.app.state.tool_service.health_check()
        return ServiceHealth(
            name="tool_service",
            status=health.get("status", "unknown"),
            details=health
        )
    except Exception as e:
        logger.error(f"Tools health check error: {e}")
        return ServiceHealth(
            name="tool_service",
            status="error",
            details={"error": str(e)}
        )


@router.get("/health/context")
async def context_health_check(req: Request):
    """Health check specifically for context manager."""
    try:
        stats = req.app.state.context_manager.get_session_stats()
        return ServiceHealth(
            name="context_manager",
            status="healthy",
            details={
                "active_sessions": stats["total_sessions"],
                "max_sessions": stats["max_sessions"],
                "sessions": stats["sessions"]
            }
        )
    except Exception as e:
        logger.error(f"Context health check error: {e}")
        return ServiceHealth(
            name="context_manager",
            status="error",
            details={"error": str(e)}
        )


@router.get("/health/readiness")
async def readiness_check(req: Request):
    """Kubernetes-style readiness probe."""
    try:
        # Check if all essential services are available
        essential_checks = []
        
        # LLM Manager
        try:
            await req.app.state.llm_manager.health_check()
            essential_checks.append(True)
        except:
            essential_checks.append(False)
        
        # RAG Service
        try:
            await req.app.state.rag_service.health_check()
            essential_checks.append(True)
        except:
            essential_checks.append(False)
        
        # Context Manager
        try:
            req.app.state.context_manager.get_session_stats()
            essential_checks.append(True)
        except:
            essential_checks.append(False)
        
        if all(essential_checks):
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"status": "not_ready", "timestamp": datetime.utcnow().isoformat()}
            
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        return {"status": "not_ready", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/liveness")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": round(time.time() - startup_time, 2)
    }
