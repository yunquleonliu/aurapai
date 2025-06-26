"""
Chat API routes for Auro-PAI Platform
=====================================

Handles chat interactions with context-aware AI assistance.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
import base64
from PIL import Image
import io

from services.llm_manager import LLMProvider
from core.context_manager import ContextManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple token estimation (rough approximation)
def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens in text (approximately 4 characters per token)."""
    return len(text) // 4

def truncate_context_to_limit(messages: List[Dict[str, str]], max_tokens: int = 6000) -> List[Dict[str, str]]:
    """Truncate messages to stay within token limit, keeping system messages and recent context."""
    if not messages:
        return messages
    
    # Always keep system messages
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    # Estimate tokens for system messages
    system_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in system_messages)
    
    # Leave room for system messages and response
    available_tokens = max_tokens - system_tokens - 500  # Reserve 500 tokens for response
    
    if available_tokens <= 0:
        # If system messages are too long, truncate them too
        truncated_system = []
        current_tokens = 0
        for msg in reversed(system_messages):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= max_tokens - 1000:
                truncated_system.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # Truncate this message
                available_chars = (max_tokens - current_tokens - 1000) * 4
                if available_chars > 100:
                    truncated_content = msg.get("content", "")[:available_chars] + "..."
                    truncated_system.insert(0, {"role": msg["role"], "content": truncated_content})
                break
        return truncated_system + non_system_messages[-1:]  # Keep last user message
    
    # Add non-system messages in reverse order until we hit the limit
    selected_messages = []
    current_tokens = 0
    
    for msg in reversed(non_system_messages):
        msg_tokens = estimate_tokens(msg.get("content", ""))
        if current_tokens + msg_tokens <= available_tokens:
            selected_messages.insert(0, msg)
            current_tokens += msg_tokens
        else:
            # Try to fit a truncated version of this message if it's the user message
            if msg.get("role") == "user" and not selected_messages:
                available_chars = (available_tokens - current_tokens) * 4
                if available_chars > 100:
                    truncated_content = msg.get("content", "")[:available_chars] + "..."
                    selected_messages.insert(0, {"role": msg["role"], "content": truncated_content})
            break
    
    return system_messages + selected_messages


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


async def detect_intent_with_llm(llm_manager, prompt: str) -> str:
    """
    Use the LLM to classify the prompt as 'image', 'text', or 'ambiguous'.
    """
    system_prompt = (
        "You are an intent classifier for a chat assistant. "
        "Classify the following user prompt as one of: 'image', 'text', or 'ambiguous'. "
        "Respond with only the label.\n"
        f"Prompt: {prompt}\n"
        "Label:"
    )
    # Use llama.cpp or your default LLM provider for intent detection
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    try:
        response = await llm_manager.generate_response(messages, max_tokens=1)
        label = response.content.strip().lower()
        if label not in {"image", "text", "ambiguous"}:
            label = "ambiguous"
        return label
    except Exception as e:
        # Fallback to ambiguous if LLM fails
        return "ambiguous"


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Main chat endpoint for AI interactions."""
    try:
        # Get services from app state
        llm_manager = req.app.state.llm_manager
        rag_service = req.app.state.rag_service
        tool_service = req.app.state.tool_service
        context_manager = req.app.state.context_manager
        image_gen_service = getattr(req.app.state, 'image_generation_service', None)
        
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
        
        # --- INTENT DETECTION ---
        intent = "text"
        if image_gen_service:
            intent = await detect_intent_with_llm(llm_manager, request.message)
        
        if intent == "image":
            # Route to image generation
            try:
                generated_images = await image_gen_service.generate_image(
                    prompt=request.message,
                    provider=None,  # Use default
                    size="1024x1024",
                    num_images=1
                )
                images_data = [img.to_dict() for img in generated_images]
                response_text = f"Generated {len(generated_images)} image(s) for: '{request.message}'"
                context_manager.add_assistant_message(session_id, response_text)
                return ChatResponse(
                    response=response_text,
                    session_id=session_id,
                    provider=generated_images[0].model if generated_images else "image-generator",
                    model=generated_images[0].model if generated_images else "placeholder",
                    metadata={
                        "type": "image_generation",
                        "images": images_data,
                        "prompt": request.message,
                        "count": len(generated_images)
                    }
                )
            except Exception as e:
                logger.error(f"Image generation error in chat: {e}")
                # Fall through to regular chat if image generation fails
        elif intent == "ambiguous":
            clarification_msg = (
                "Did you want to generate an image or get a text answer? "
                "Please reply with 'image' or 'text'."
            )
            context_manager.add_assistant_message(session_id, clarification_msg)
            return ChatResponse(
                response=clarification_msg,
                session_id=session_id,
                provider="intent-classifier",
                model="intent-classifier",
                usage={},
                metadata={"type": "clarification", "prompt": request.message}
            )
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


@router.get("/chat/stream")
async def stream_chat_v1(
    req: Request,
    message: str,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    include_rag: bool = True,
    include_tools: bool = True
):
    """Stream chat response (v1)."""
    try:
        llm_manager = req.app.state.llm_manager
        context_manager = req.app.state.context_manager
        rag_service = req.app.state.rag_service
        tool_service = req.app.state.tool_service
        
        # Validate services are available
        if not llm_manager:
            raise HTTPException(status_code=503, detail="LLM service not available")
        if not context_manager:
            raise HTTPException(status_code=503, detail="Context manager not available")
        
        # Create or get session
        if not session_id:
            session_id = context_manager.create_session()
        else:
            session_id = session_id
            if not context_manager.get_context(session_id):
                context_manager.create_session(session_id)
        
        # Add user message to context
        context_manager.add_user_message(session_id, message)
        
        # Build messages for LLM
        messages = await _build_llm_messages(
            session_id=session_id,
            user_message=message,
            context_manager=context_manager,
            rag_service=rag_service if include_rag else None,
            tool_service=tool_service if include_tools else None
        )
        
        # Determine provider
        llm_provider = LLMProvider(provider) if provider else None

        async def event_generator():
            full_response = ""
            try:
                # First, send the session_id as a separate event
                session_info = {"session_id": session_id}
                yield f"event: session_start\ndata: {json.dumps(session_info)}\n\n"

                # Then, stream the content
                async for chunk in llm_manager.generate_streaming_response(
                    messages=messages,
                    provider=llm_provider,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if chunk:
                        full_response += chunk
                        # SSE format: event: message, data: {}
                        message_event = {"content": chunk}
                        yield f"event: message\ndata: {json.dumps(message_event)}\n\n"
                
                # Add complete response to context after streaming is finished
                context_manager.add_assistant_message(session_id, full_response)
                
                # Send a final event to signal the end of the stream
                yield "event: end\ndata: {}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_message = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_message)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_v1(req: Request):
    """Health check endpoint."""
    return {"status": "healthy", "message": "Chat service is running"}


@router.post("/chat/image")
async def chat_with_image(
    req: Request,
    message: str = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    max_tokens: Optional[int] = Form(None),
    include_rag: bool = Form(True),
    include_tools: bool = Form(True)
):
    """Chat endpoint with image upload support."""
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
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # Process image
        image_data = await image.read()
        
        # Convert image to base64 for LLM
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            # Resize if too large
            if pil_image.width > 1024 or pil_image.height > 1024:
                pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Create or get session
        if not session_id:
            session_id = context_manager.create_session()
        else:
            if not context_manager.get_context(session_id):
                context_manager.create_session(session_id)
        
        # Add user message to context (with image info)
        user_message_with_image = f"{message} [Image attached: {image.filename}]"
        context_manager.add_user_message(session_id, user_message_with_image)
        
        # Build enhanced prompt for image analysis
        image_prompt = f"""The user has shared an image and asked: "{message}"

Please analyze the uploaded image and respond to their question. Describe what you see in the image and provide relevant insights based on the user's query.

Image data: data:image/jpeg;base64,{image_base64}"""
        
        # Build messages for LLM including image
        messages = await _build_llm_messages_with_image(
            session_id=session_id,
            user_message=message,
            image_base64=image_base64,
            context_manager=context_manager,
            rag_service=rag_service if include_rag else None,
            tool_service=tool_service if include_tools else None
        )
        
        # Determine provider
        llm_provider = None
        if provider:
            try:
                llm_provider = LLMProvider(provider)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
        
        # Generate response (non-streaming for images for now)
        try:
            from core.config import settings
            llm_response = await asyncio.wait_for(
                llm_manager.generate_response(
                    messages=messages,
                    provider=llm_provider,
                    temperature=temperature,
                    max_tokens=max_tokens
                ),
                timeout=settings.LLAMACPP_TIMEOUT
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
            usage=llm_response.usage,
            metadata={"image_processed": True, "image_filename": image.filename}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Image chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/test-search")
async def test_web_search(req: Request, query: str = "latest news"):
    """Test endpoint to verify web search functionality."""
    try:
        tool_service = req.app.state.tool_service
        
        if not tool_service:
            return {"error": "Tool service not available"}
        
        # Test if query should trigger search
        should_search = await _should_search_web(query)
        
        if should_search:
            # Perform actual search
            search_results = await tool_service.web_search(query, max_results=3)
            
            return {
                "query": query,
                "should_search": should_search,
                "search_triggered": True,
                "results_count": len(search_results),
                "results": [result.to_dict() for result in search_results]
            }
        else:
            return {
                "query": query,
                "should_search": should_search,
                "search_triggered": False,
                "message": "Query does not require web search"
            }
            
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@router.get("/test-search")
async def test_search(query: str = Query(...), tool_service: ToolService = Depends(get_tool_service)):
    """Test the web search tool directly."""
    import logging
    logger = logging.getLogger("aurapai.websearch")
    logger.info(f"Test search endpoint called with query: {query}")
    try:
        results = await tool_service.web_search(query, max_results=3)
        logger.info(f"Web search returned {len(results) if results else 0} results for query: {query}")
        return {"results": [r.dict() for r in results]}
    except Exception as e:
        logger.warning(f"Web search failed in /test-search: {e}")
        return {"error": str(e)}


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text string."""
    try:
        # Use tiktoken for more accurate token counting
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4


async def _build_llm_messages(
    session_id: str,
    user_message: str,
    context_manager: ContextManager,
    rag_service=None,
    tool_service=None
) -> List[Dict[str, str]]:
    """Build messages for LLM including context and tools."""
    
    messages = []
    
    # System message with platform context (concise version)
    system_prompt = """You are Aura-PAI, an AI assistant for code, documentation, and general tasks. You have:

1. Conversation context
2. Local codebase access (RAG)
3. Real-time web search (for current info)
4. URL fetching tools

Guidelines:
- Provide helpful, accurate responses
- Explain code changes clearly
- Use web search for current events/news automatically
- Be transparent about capabilities
- Focus on practical advice"""
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    conversation_context = context_manager.build_context_for_llm(session_id)
    
    # Add recent conversation messages (excluding current user message)
    for msg in conversation_context["messages"][:-1]:  # Exclude the just-added user message
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add RAG context if available (optimized)
    if rag_service and user_message:
        try:
            rag_context = await rag_service.get_context_for_query(user_message)
            if rag_context["context"]:
                # Truncate RAG context to prevent overflow
                context_content = rag_context["context"]
                if len(context_content) > 1000:  # Limit RAG context length
                    context_content = context_content[:1000] + "...[truncated for length]"
                rag_message = f"Relevant codebase context:\n{context_content}"
                messages.append({"role": "system", "content": rag_message})
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
    
    # Add web search context for time-sensitive queries (optimized)
    if tool_service and await _should_search_web(user_message):
        logger.info(f"Web search triggered for query: {user_message}")
        try:
            search_results = await tool_service.web_search(user_message, max_results=3)  # Reduced from 5 to 3
            logger.info(f"Web search results: {search_results}")
            if search_results:
                # Create more concise web context
                web_context = "Recent web search results:\n"
                for i, result in enumerate(search_results[:3], 1):
                    # Truncate long snippets
                    snippet = result.snippet[:150] + "..." if len(result.snippet) > 150 else result.snippet
                    web_context += f"{i}. {result.title}\n{snippet}\nSource: {result.url}\n\n"
                messages.append({"role": "system", "content": web_context})
                logger.info(f"Added web search context for query: {user_message}")
            else:
                logger.warning(f"Web search returned no results for query: {user_message}")
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
    
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
    
    # Truncate context to stay within token limits
    messages = truncate_context_to_limit(messages, max_tokens=4096)
    
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


async def _should_search_web(user_message: str) -> bool:
    """Determine if a user message requires web search for current information."""
    # Convert to lowercase for easier matching
    message_lower = user_message.lower()
    
    # Time-sensitive keywords that indicate need for current information
    time_sensitive_keywords = [
        "latest", "recent", "current", "today", "yesterday", "this week", "this month",
        "news", "breaking", "update", "now", "currently", "happening", "just",
        "new", "fresh", "live", "real-time", "trending", "hot", "viral",
        "2024", "2025",  # Current years
        "what's", "whats", "what is happening", "what happened",
        "stock price", "weather", "forecast", "score", "results",
        "election", "covid", "pandemic", "crisis", "emergency"
    ]
    
    # Question patterns that often need current info
    question_patterns = [
        "what's the", "what is the", "how is", "how are",
        "when did", "when will", "when is", "where is",
        "who won", "who is", "who are", "why did",
        "tell me about recent", "give me the latest"
    ]
    
    # Check for time-sensitive keywords
    for keyword in time_sensitive_keywords:
        if keyword in message_lower:
            return True
    
    # Check for question patterns
    for pattern in question_patterns:
        if pattern in message_lower:
            # Additional check for current context
            current_indicators = ["today", "now", "current", "latest", "recent", "new"]
            if any(indicator in message_lower for indicator in current_indicators):
                return True
    
    # Check if message asks about specific events or companies that might need current info
    if any(word in message_lower for word in ["price", "stock", "market", "economy", "gdp"]):
        return True
    
    # Check for news-related queries
    if any(word in message_lower for word in ["news", "report", "announcement", "released"]):
        return True
    
    return False


async def _build_llm_messages_with_image(
    session_id: str,
    user_message: str,
    image_base64: str,
    context_manager: ContextManager,
    rag_service=None,
    tool_service=None
) -> List[Dict[str, str]]:
    """Build messages for LLM including image context."""
    
    messages = []
    
    # System message with image analysis context (concise)
    system_prompt = """You are Aura-PAI with vision capabilities. You can analyze images and have:

1. Conversation context
2. Codebase access (RAG)
3. Web search for current info
4. Image analysis abilities

For images:
- Describe what you see clearly
- Answer specific questions about the image
- Read/transcribe any text accurately
- Identify objects, people, scenes, colors
- Be honest if unclear about something"""
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history (excluding current message)
    conversation_context = context_manager.build_context_for_llm(session_id)
    for msg in conversation_context["messages"][:-1]:
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
    
    # Add web search for image-related queries if needed
    if tool_service and await _should_search_web(user_message):
        try:
            search_results = await tool_service.web_search(user_message, max_results=3)
            if search_results:
                web_context = "Recent web search results:\n\n"
                for result in search_results:
                    web_context += f"**{result.title}**\n{result.snippet}\nSource: {result.url}\n\n"
                messages.append({"role": "system", "content": web_context})
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
    
    # Add current user message with image
    image_message = f"""The user asks: "{user_message}"

They have shared an image. Please analyze this image and respond to their question.

[Image data: data:image/jpeg;base64,{image_base64}]"""
    
    messages.append({"role": "user", "content": image_message})
    
    # Truncate context to stay within token limits
    messages = truncate_context_to_limit(messages, max_tokens=4096)
    
    return messages


@router.post("/chat/generate-images")
async def generate_images_from_menu(
    req: Request,
    message: str = Form(...),
    image: UploadFile = File(...),
    style: str = Form("photorealistic"),
    max_dishes: int = Form(5),
    provider: Optional[str] = Form(None)
):
    """Generate images of dishes from a menu photo."""
    try:
        # Get services from app state
        llm_manager = req.app.state.llm_manager
        context_manager = req.app.state.context_manager
        image_gen_service = getattr(req.app.state, 'image_generation_service', None)
        
        if not image_gen_service:
            raise HTTPException(status_code=503, detail="Image generation service not available")
        
        # Process uploaded menu image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Resize if too large
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Extract dishes from menu using LLM
        extraction_prompt = f"""
Analyze this menu image and extract dish information. For each dish, provide:
1. Dish name
2. Brief description
3. Price (if visible)

Format as JSON list:
[{{"name": "dish name", "description": "brief description", "price": "price"}}]

User request: {message}
Image: data:image/jpeg;base64,{image_base64}
"""
        
        # Get dish list from LLM
        llm_response = await llm_manager.generate_response(
            messages=[{"role": "user", "content": extraction_prompt}],
            max_tokens=1000
        )
        
        # Parse dishes (you'd want more robust JSON parsing here)
        try:
            import json
            import re
            json_match = re.search(r'\[.*\]', llm_response.content, re.DOTALL)
            if json_match:
                dishes = json.loads(json_match.group())
            else:
                # Fallback: simple extraction
                dishes = [{"name": "Sample Dish", "description": "Delicious food item"}]
        except:
            dishes = [{"name": "Sample Dish", "description": "Delicious food item"}]
        
        # Limit number of dishes
        dishes = dishes[:max_dishes]
        
        # Generate images for each dish
        generated_images = await image_gen_service.generate_multiple_dishes(
            dishes=dishes,
            style=style,
            provider=provider
        )
        
        return {
            "success": True,
            "message": f"Generated images for {len(generated_images)} dishes",
            "dishes": dishes,
            "generated_images": generated_images,
            "style": style,
            "provider": provider or "default"
        }
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/generate-single-dish")
async def generate_single_dish_image(
    req: Request,
    dish_name: str = Form(...),
    description: str = Form(""),
    style: str = Form("photorealistic"),
    provider: Optional[str] = Form(None)
):
    """Generate image of a single dish by name and description."""
    try:
        image_gen_service = getattr(req.app.state, 'image_generation_service', None)
        
        if not image_gen_service:
            raise HTTPException(status_code=503, detail="Image generation service not available")
        
        # Generate image
        result = await image_gen_service.generate_dish_image(
            dish_name=dish_name,
            description=description,
            style=style,
            provider=provider
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Single dish generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/generate-image")
async def generate_image(
    req: Request,
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    size: str = Form("1024x1024"),
    style: Optional[str] = Form(None),
    num_images: int = Form(1)
):
    """Generate images from text prompts."""
    try:
        # Get services from app state
        image_service = req.app.state.image_generation_service
        context_manager = req.app.state.context_manager
        
        # Validate services are available
        if not image_service:
            raise HTTPException(status_code=503, detail="Image generation service not available")
        if not context_manager:
            raise HTTPException(status_code=503, detail="Context manager not available")
        
        # Create or get session
        if not session_id:
            session_id = context_manager.create_session()
        else:
            if not context_manager.get_context(session_id):
                context_manager.create_session(session_id)
        
        # Add user request to context
        context_manager.add_user_message(session_id, f"Generate image: {prompt}")
        
        # Generate images
        generated_images = await image_service.generate_image(
            prompt=prompt,
            provider=provider,
            size=size,
            style=style,
            num_images=min(num_images, 4)  # Limit to 4 images max
        )
        
        # Build response
        images_data = [img.to_dict() for img in generated_images]
        
        # Add to context
        response_text = f"Generated {len(generated_images)} image(s) for prompt: '{prompt}'"
        context_manager.add_assistant_message(session_id, response_text)
        
        return {
            "message": response_text,
            "session_id": session_id,
            "images": images_data,
            "metadata": {
                "prompt": prompt,
                "provider": generated_images[0].model if generated_images else "unknown",
                "count": len(generated_images)
            }
        }
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
