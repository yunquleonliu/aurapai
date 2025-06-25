"""
Chat API routes for Auro-PAI Platform
=====================================

Handles chat interactions with context-aware AI assistance.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio

from services.llm_manager import LLMProvider
from core.context_manager import ContextManager

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for context continuity")
    provider: Optional[str] = Field(default=None, description="LLM provider to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Response temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096, description="Maximum response tokens")
    include_rag: bool = Field(default=True, description="Include RAG context")
    include_tools: bool = Field(default=True, description="Enable tool usage")
    stream: bool = Field(default=False, description="Stream response")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")
    rag_context: Optional[List[Dict[str, Any]]] = Field(default=None, description="RAG context used")
    tool_usage: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tools used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")


class ConversationHistory(BaseModel):
    """Conversation history model."""
    session_id: str
    messages: List[ChatMessage]
    created_at: str
    last_accessed: str
    message_count: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Main chat endpoint for AI interactions."""
    try:
        # Get services from app state
        llm_manager = req.app.state.llm_manager
        rag_service = req.app.state.rag_service
        tool_service = req.app.state.tool_service
        context_manager = req.app.state.context_manager
        
        # Validate services are available
        if not llm_manager:
            raise HTTPException(status_code=503, detail="LLM service not available")
        if not context_manager:
            raise HTTPException(status_code=503, detail="Context manager not available")
        
        # Create or get session
        if not request.session_id:
            session_id = context_manager.create_session()
        else:
            session_id = request.session_id
            if not context_manager.get_context(session_id):
                context_manager.create_session(session_id)
        
        # Add user message to context
        context_manager.add_user_message(session_id, request.message)
        
        # Build messages for LLM
        messages = await _build_llm_messages(
            session_id=session_id,
            user_message=request.message,
            context_manager=context_manager,
            rag_service=rag_service if request.include_rag else None,
            tool_service=tool_service if request.include_tools else None
        )
        
        # Determine provider
        provider = None
        if request.provider:
            try:
                provider = LLMProvider(request.provider)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
        
        # Generate response
        if request.stream:
            return StreamingResponse(
                _stream_chat_response(
                    messages=messages,
                    session_id=session_id,
                    llm_manager=llm_manager,
                    context_manager=context_manager,
                    provider=provider,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                media_type="text/plain"
            )
        else:
            # Generate response with timeout (use same timeout as LLM config)
            try:
                from core.config import settings
                llm_response = await asyncio.wait_for(
                    llm_manager.generate_response(
                        messages=messages,
                        provider=provider,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    ),
                    timeout=settings.LLAMACPP_TIMEOUT  # Use configured timeout
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail=f"Request timeout - LLM took longer than {settings.LLAMACPP_TIMEOUT} seconds to respond")
            
            # Add assistant response to context
            context_manager.add_assistant_message(session_id, llm_response.content)
            
            # Build response
            response = ChatResponse(
                response=llm_response.content,
                session_id=session_id,
                provider=llm_response.provider,
                model=llm_response.model,
                usage=llm_response.usage
            )
            
            return response
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions")
async def get_sessions(req: Request):
    """Get all active chat sessions."""
    try:
        context_manager = req.app.state.context_manager
        stats = context_manager.get_session_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions/{session_id}/history")
async def get_conversation_history(session_id: str, req: Request, limit: int = 50):
    """Get conversation history for a session."""
    try:
        context_manager = req.app.state.context_manager
        
        context = context_manager.get_context(session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = context.get_recent_messages(limit)
        
        history = ConversationHistory(
            session_id=session_id,
            messages=[ChatMessage(**msg) for msg in messages],
            created_at=context.created_at.isoformat(),
            last_accessed=context.last_accessed.isoformat(),
            message_count=len(context.messages)
        )
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, req: Request):
    """Delete a chat session."""
    try:
        context_manager = req.app.state.context_manager
        
        success = context_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sessions/{session_id}/task")
async def set_current_task(session_id: str, task: str, req: Request):
    """Set the current task for a session."""
    try:
        context_manager = req.app.state.context_manager
        
        context = context_manager.get_context(session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        context_manager.set_current_task(session_id, task)
        
        return {"message": f"Task set for session {session_id}", "task": task}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/test")
async def test_chat(req: Request):
    """Simple test endpoint to verify chat service is working."""
    try:
        llm_manager = req.app.state.llm_manager
        context_manager = req.app.state.context_manager
        
        # Quick health check
        llm_status = await llm_manager.health_check()
        
        return {
            "status": "chat service is ready",
            "llm_status": llm_status,
            "context_sessions": context_manager.get_session_stats() if context_manager else {},
            "endpoints": [
                "POST /chat - Main chat endpoint",
                "GET /chat/sessions - List sessions", 
                "GET /chat/sessions/{id}/history - Get history",
                "DELETE /chat/sessions/{id} - Delete session"
            ]
        }
    except Exception as e:
        logger.error(f"Chat test error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Chat service is not ready"
        }


@router.post("/chat/test")
async def chat_test(req: Request, message: str = "Hello"):
    """Simple test endpoint for debugging chat issues."""
    try:
        llm_manager = req.app.state.llm_manager
        
        if not llm_manager:
            return {"error": "LLM manager not available", "status": "failed"}
        
        # Simple test message
        messages = [{"role": "user", "content": message}]
        
        import time
        start_time = time.time()
        
        # Test with a very short max_tokens to make it faster
        llm_response = await llm_manager.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=50  # Short response for testing
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "status": "success",
            "response": llm_response.content,
            "provider": llm_response.provider,
            "model": llm_response.model,
            "response_time_seconds": round(response_time, 2),
            "usage": llm_response.usage
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# Add convenient aliases at /v1 path for easier access
@router.post("/v1/chat", response_model=ChatResponse)
async def chat_v1(request: ChatRequest, req: Request):
    """Convenient alias for chat endpoint at /v1/chat path."""
    return await chat(request, req)


@router.get("/v1/health")
async def health_v1(req: Request):
    """Health check endpoint at /v1/health."""
    return {"status": "healthy", "message": "Chat service is running"}


async def _build_llm_messages(
    session_id: str,
    user_message: str,
    context_manager: ContextManager,
    rag_service=None,
    tool_service=None
) -> List[Dict[str, str]]:
    """Build messages for LLM including context and tools."""
    
    messages = []
    
    # System message with platform context
    system_prompt = """You are Auro-PAI, an AI assistant designed to help users with code, documentation, and general tasks. You have access to:

1. Context from previous conversations
2. Local codebase and documentation (via RAG)
3. Web search and URL fetching tools (when enabled)

Guidelines:
- Provide helpful, accurate, and contextually relevant responses
- When suggesting code changes, explain the reasoning clearly
- Use available tools when you need external information
- Be transparent about your capabilities and limitations
- Focus on practical, actionable advice"""
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    conversation_context = context_manager.build_context_for_llm(session_id)
    
    # Add recent conversation messages (excluding current user message)
    for msg in conversation_context["messages"][:-1]:  # Exclude the just-added user message
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add RAG context if available
    if rag_service and user_message:
        try:
            rag_context = await rag_service.get_context_for_query(user_message)
            if rag_context["context"]:
                rag_message = f"Relevant context from codebase:\n\n{rag_context['context']}"
                messages.append({"role": "system", "content": rag_message})
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
    
    # Add current task context
    if conversation_context.get("current_task"):
        task_message = f"Current task: {conversation_context['current_task']}"
        messages.append({"role": "system", "content": task_message})
    
    # Add tool availability context
    if tool_service:
        try:
            available_tools = await tool_service.get_available_tools()
            if available_tools["available_tools"]:
                tools_message = f"Available tools: {', '.join(available_tools['available_tools'])}"
                messages.append({"role": "system", "content": tools_message})
        except Exception as e:
            logger.warning(f"Tool availability check failed: {e}")
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    return messages


async def _stream_chat_response(
    messages: List[Dict[str, str]],
    session_id: str,
    llm_manager,
    context_manager: ContextManager,
    provider=None,
    temperature=0.7,
    max_tokens=None
):
    """Stream chat response."""
    try:
        full_response = ""
        
        async for chunk in llm_manager.generate_streaming_response(
            messages=messages,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            full_response += chunk
            yield chunk
        
        # Add complete response to context
        context_manager.add_assistant_message(session_id, full_response)
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_message = f"Error: {str(e)}"
        yield error_message
