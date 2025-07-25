"""
Context Manager for Auro-PAI Platform
=====================================

Manages conversation context, session state, and provides context-aware
responses with proper memory management.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging

from core.config import settings

if TYPE_CHECKING:
    from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class MultiStepTask:
    """Represents a multi-step task or plan."""
    def __init__(self, original_query: str, plan: List[Dict[str, Any]]):
        self.original_query = original_query
        self.plan = plan
        self.current_step = 0
        self.status = "in_progress"  # in_progress, completed, failed
        self.step_results: Dict[int, Any] = {}

    def get_next_step(self):
        if self.current_step < len(self.plan):
            return self.plan[self.current_step]
        return None

    def complete_step(self, result: Any):
        self.step_results[self.current_step] = result
        self.current_step += 1
        if self.current_step >= len(self.plan):
            self.status = "completed"

    def is_finished(self) -> bool:
        """Check if the task is completed or failed."""
        return self.status in ["completed", "failed"]

    def to_dict(self):
        return {
            "original_query": self.original_query,
            "plan": self.plan,
            "current_step": self.current_step,
            "status": self.status,
            "step_results": self.step_results,
        }


class ConversationContext:
    """Represents a single conversation context."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        self.rag_context: List[Dict[str, Any]] = []
        self.tool_usage_history: List[Dict[str, Any]] = []
        self.current_task: Optional[MultiStepTask] = None # For Plan-and-Execute
        self.react_state: Dict[str, Any] = {} # For ReAct agent state
        self.messages: List[Dict[str, Any]] = []  # Store chat messages
    
    def init(self):
        """Initialize the conversation with a system message."""
        initial_message = "Welcome to Aura-PAI! Your intelligent assistant for a seamless experience. How can I help you today?"
        self.add_message("system", initial_message)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        })
        
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

    def get_react_scratchpad(self) -> str:
        """Get the scratchpad for the ReAct agent."""
        return self.react_state.get("scratchpad", "")

    def update_react_scratchpad(self, thought: str, action: Dict[str, Any], observation: str):
        """Update the ReAct scratchpad with the latest turn."""
        scratchpad_entry = f"Thought: {thought}\nAction: {json.dumps(action)}\nObservation: {observation}\n"
        current_scratchpad = self.react_state.get("scratchpad", "")
        self.react_state["scratchpad"] = current_scratchpad + scratchpad_entry

    def clear_react_state(self):
        """Clear the ReAct state, including the scratchpad."""
        self.react_state = {}


class ContextManager:
    """Manages multiple conversation contexts."""
    
    def __init__(self, rag_service: RAGService):
        self.contexts: Dict[str, ConversationContext] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.rag_service = rag_service
        
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
    
    async def add_user_message(self, user_id: str, session_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a user message to the conversation."""
        context = self.get_context(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, creating new one")
            self.create_session(session_id)
            context = self.get_context(session_id)
        
        context.add_message("user", content, metadata)
        if self.rag_service:
            await self.rag_service.add_chat_message(user_id, session_id, content, "user", metadata)
        
    async def add_assistant_message(self, user_id: str, session_id: str, content: str, metadata: Optional[Dict] = None):
        """Add an assistant message to the conversation."""
        context = self.get_context(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, but adding assistant message anyway.")
            context = self.create_session(session_id)

        context.add_message("assistant", content, metadata)
        if self.rag_service:
            await self.rag_service.add_chat_message(user_id, session_id, content, "assistant", metadata)
    
    async def add_system_message(self, user_id: str, session_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a system message to the conversation."""
        context = self.get_context(session_id)
        if context:
            context.add_message("system", content, metadata)
            if self.rag_service:
                await self.rag_service.add_chat_message(user_id, session_id, content, "system", metadata)
    
    async def get_conversation_history(self, user_id: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if self.rag_service:
            return await self.rag_service.get_chat_history(user_id, session_id, limit)
        return []
    
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
            "current_task": context.current_task.to_dict() if context.current_task else None
        }
        
        if include_rag:
            result["rag_context"] = context.rag_context[-5:]  # Last 5 RAG contexts
        
        if include_tools:
            result["tool_history"] = context.tool_usage_history[-3:]  # Last 3 tool uses
        
        return result
    
    def set_current_task(self, session_id: str, task: MultiStepTask):
        """Set the current task for a session."""
        context = self.get_context(session_id)
        if context:
            context.current_task = task

    def get_current_task(self, session_id: str) -> Optional[MultiStepTask]:
        """Get the current task for a session."""
        context = self.get_context(session_id)
        if context:
            return context.current_task
        return None

    def get_session_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all active sessions."""
        return [
            {
                "session_id": c.session_id,
                "created_at": c.created_at.isoformat(),
                "last_accessed": c.last_accessed.isoformat(),
                "message_count": len(c.messages),
                "current_task": c.current_task.to_dict() if c.current_task else None
            }
            for c in self.contexts.values()
        ]

    async def _cleanup_expired_contexts(self):
        """Periodically clean up expired contexts."""
        while True:
            await asyncio.sleep(settings.SESSION_CLEANUP_INTERVAL)
            
            expired_sessions = [
                sid for sid, context in self.contexts.items() if context.is_expired()
            ]
            
            for sid in expired_sessions:
                del self.contexts[sid]
                logger.info(f"Cleaned up expired session: {sid}")
