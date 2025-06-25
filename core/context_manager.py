"""
Context Manager for Auro-PAI Platform
=====================================

Manages conversation context, session state, and provides context-aware
responses with proper memory management.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from core.config import settings

logger = logging.getLogger(__name__)


class ConversationContext:
    """Represents a single conversation context."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.rag_context: List[Dict[str, Any]] = []
        self.tool_usage_history: List[Dict[str, Any]] = []
        self.current_task: Optional[str] = None
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.last_accessed = datetime.utcnow()
        
    def add_rag_context(self, context: Dict[str, Any]):
        """Add RAG retrieval context."""
        context["timestamp"] = datetime.utcnow().isoformat()
        self.rag_context.append(context)
        
    def add_tool_usage(self, tool_name: str, input_data: Any, output_data: Any):
        """Record tool usage."""
        tool_record = {
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.tool_usage_history.append(tool_record)
        
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages within context limit."""
        return self.messages[-limit:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "rag_contexts": len(self.rag_context),
            "tool_usage_count": len(self.tool_usage_history),
            "current_task": self.current_task,
            "last_accessed": self.last_accessed.isoformat()
        }
    
    def is_expired(self) -> bool:
        """Check if the context has expired."""
        timeout = timedelta(seconds=settings.SESSION_TIMEOUT)
        return datetime.utcnow() - self.last_accessed > timeout


class ContextManager:
    """Manages multiple conversation contexts and provides context-aware operations."""
    
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the context manager."""
        logger.info("Initializing Context Manager...")
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_contexts())
        
        logger.info("Context Manager initialized successfully")
        
    async def cleanup(self):
        """Cleanup resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Context Manager cleaned up")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.contexts:
            logger.warning(f"Session {session_id} already exists, returning existing")
            return session_id
        
        # Check session limit
        if len(self.contexts) >= settings.MAX_SESSIONS:
            # Remove oldest session
            oldest_session = min(
                self.contexts.keys(),
                key=lambda k: self.contexts[k].last_accessed
            )
            del self.contexts[oldest_session]
            logger.info(f"Removed oldest session {oldest_session} due to limit")
        
        self.contexts[session_id] = ConversationContext(session_id)
        logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context by session ID."""
        context = self.contexts.get(session_id)
        if context:
            context.last_accessed = datetime.utcnow()
        return context
    
    def add_user_message(self, session_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a user message to the conversation."""
        context = self.get_context(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, creating new one")
            self.create_session(session_id)
            context = self.get_context(session_id)
        
        context.add_message("user", content, metadata)
        
    def add_assistant_message(self, session_id: str, content: str, metadata: Optional[Dict] = None):
        """Add an assistant message to the conversation."""
        context = self.get_context(session_id)
        if context:
            context.add_message("assistant", content, metadata)
    
    def add_system_message(self, session_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a system message to the conversation."""
        context = self.get_context(session_id)
        if context:
            context.add_message("system", content, metadata)
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        context = self.get_context(session_id)
        if not context:
            return []
        
        return context.get_recent_messages(limit)
    
    def build_context_for_llm(self, session_id: str, include_rag: bool = True, include_tools: bool = True) -> Dict[str, Any]:
        """Build context information for LLM requests."""
        context = self.get_context(session_id)
        if not context:
            return {
                "messages": [],
                "rag_context": [],
                "tool_history": [],
                "current_task": None
            }
        
        result = {
            "messages": context.get_recent_messages(),
            "current_task": context.current_task
        }
        
        if include_rag:
            result["rag_context"] = context.rag_context[-5:]  # Last 5 RAG contexts
        
        if include_tools:
            result["tool_history"] = context.tool_usage_history[-3:]  # Last 3 tool uses
        
        return result
    
    def set_current_task(self, session_id: str, task: str):
        """Set the current task for a session."""
        context = self.get_context(session_id)
        if context:
            context.current_task = task
    
    def clear_current_task(self, session_id: str):
        """Clear the current task for a session."""
        context = self.get_context(session_id)
        if context:
            context.current_task = None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self.contexts:
            del self.contexts[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current sessions."""
        return {
            "total_sessions": len(self.contexts),
            "max_sessions": settings.MAX_SESSIONS,
            "sessions": [
                ctx.get_context_summary() for ctx in self.contexts.values()
            ]
        }
    
    async def _cleanup_expired_contexts(self):
        """Background task to clean up expired contexts."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                expired_sessions = [
                    session_id for session_id, context in self.contexts.items()
                    if context.is_expired()
                ]
                
                for session_id in expired_sessions:
                    del self.contexts[session_id]
                    logger.info(f"Cleaned up expired session: {session_id}")
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in context cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying
