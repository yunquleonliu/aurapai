"""
Chat API routes for Auro-PAI Platform
=====================================-

Handles chat interactions with the Plan-and-Execute agent.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.context_manager import ContextManager, MultiStepTask
from services.llm_manager import LLMProvider
from services.tool_service import ToolService

# Define router and logger
router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_usage: List[Dict[str, Any]] = []
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, Any] = {}

class ConversationHistory(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]
    message_count: int

# --- Dependency Injectors ---

def get_llm_manager(request: Request) -> LLMProvider:
    return request.app.state.llm_manager

def get_context_manager(request: Request) -> ContextManager:
    return request.app.state.context_manager

def get_tool_service(request: Request) -> ToolService:
    return request.app.state.tool_service

def get_rag_service(request: Request):
    # Optional dependency
    return getattr(request.app.state, 'rag_service', None)

# --- Helper Functions for Plan-and-Execute ---

def get_planner_prompt(user_query: str, tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Creates the prompt to ask the LLM to generate a plan."""
    tool_list = json.dumps(tools, indent=2)
    return [
        {
            "role": "system",
            "content": f"""You are a master planning agent. Your goal is to create a step-by-step plan to fulfill the user's request.
You have the following tools available:
{tool_list}

- Based on the user's query, create a JSON object with a 'plan' key.
- The 'plan' must be a list of steps. Each step is a dictionary with 'tool' and 'arguments'.
- Each step must have a unique 'step_id' starting from 0.
- If a step requires the output of a previous step, use a placeholder like '<step_0_output>' corresponding to the step_id.
- If the query is simple and doesn't require any tools (e.g., "hello"), respond with an empty plan: {{"plan": []}}.
- If you need to ask the user for clarification, use the 'clarification' tool.
- You must only use the tools provided in the list. Do not invent new tools.
"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

def parse_plan_from_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """Parses the JSON plan from the LLM's raw response."""
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        plan_obj = json.loads(response)
        if isinstance(plan_obj.get('plan'), list):
            for i, step in enumerate(plan_obj['plan']):
                step['step_id'] = i
            return plan_obj['plan']
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Error parsing plan from LLM response: {e}\nResponse: {response}")
    return None

def _substitute_placeholders(arguments: Dict[str, Any], results: Dict[int, Any]) -> Dict[str, Any]:
    """Substitute placeholders like '<step_0_output>' with actual results."""
    substituted_args = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            match = re.search(r'<step_(\d+)_output>', value)
            if match:
                step_index = int(match.group(1))
                if step_index in results:
                    substituted_args[key] = value.replace(match.group(0), str(results[step_index]))
                else:
                    substituted_args[key] = value
            else:
                substituted_args[key] = value
        elif isinstance(value, list):
            substituted_args[key] = [_substitute_placeholders({"item": v}, results)["item"] for v in value]
        elif isinstance(value, dict):
            substituted_args[key] = _substitute_placeholders(value, results)
        else:
            substituted_args[key] = value
    return substituted_args

def get_summarizer_prompt(user_query: str, task: MultiStepTask) -> List[Dict[str, str]]:
    """Creates the prompt for the summarizer LLM."""
    results_summary = []
    MAX_RESULT_LENGTH = 8000  # Max characters for a single tool result to keep context manageable

    for i, step in enumerate(task.plan):
        result = task.step_results.get(i, {"status": "pending"})
        
        # Truncate long results to prevent context overflow
        result_str = str(result)
        if len(result_str) > MAX_RESULT_LENGTH:
            result_str = result_str[:MAX_RESULT_LENGTH] + f"... (Result truncated, original length: {len(result_str)})"

        # Format each step's summary on a new line
        summary_line = f"Step {i+1}:\n- Tool: {step['tool']}\n- Arguments: {step.get('arguments', {})}\n- Result: {result_str}"
        results_summary.append(summary_line)

    # Join the summaries with newlines
    full_summary = "\n".join(results_summary)

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Your task is to synthesize the results of a multi-step operation into a single, coherent, and user-friendly response of about 300 words. The user will provide their original query and the results of each step. Do not just list the results; explain what was done and what the outcome is. If an image was generated, mention it. If a search was performed, summarize the findings."
        },
        {
            "role": "user",
            "content": f"""Original Query: {user_query}

Execution Results:
{full_summary}
"""
        }
    ]

# --- Core Chat Endpoint ---

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_manager: LLMProvider = Depends(get_llm_manager),
    context_manager: ContextManager = Depends(get_context_manager),
    tool_service: ToolService = Depends(get_tool_service),
):
    """
    Main chat endpoint for the Plan-and-Execute agent.
    """
    try:
        user_id = request.user_id
        session_id = request.session_id or context_manager.create_session()
        
        if not context_manager.get_context(session_id):
            context_manager.create_session(session_id)
            
        await context_manager.add_user_message(user_id, session_id, request.message)
        
        current_task = context_manager.get_current_task(session_id)
        
        if not current_task or current_task.is_finished():
            tools = tool_service.get_tool_definitions()
            plan_prompt = get_planner_prompt(request.message, tools)
            
            logger.info("Generating a new plan...")
            plan_response = await llm_manager.generate_response(
                plan_prompt, 
                max_tokens=1024,
                temperature=0.2  # Lower temperature for planning
            )
            plan_steps = parse_plan_from_response(plan_response.content)
            
            if not plan_steps:
                 logger.info("No plan generated. Responding directly.")
                 direct_response = await llm_manager.generate_response(
                     messages=[{"role": "user", "content": request.message}],
                     max_tokens=2000
                 )
                 await context_manager.add_assistant_message(user_id, session_id, direct_response.content)
                 return ChatResponse(
                     response=direct_response.content, 
                     session_id=session_id,
                     provider=direct_response.provider,
                     model=direct_response.model
                 )

            current_task = MultiStepTask(original_query=request.message, plan=plan_steps)
            context_manager.set_current_task(session_id, current_task)
            logger.info(f"New plan created with {len(plan_steps)} steps.")

        results = current_task.step_results
        tool_usage = []

        while not current_task.is_finished():
            step = current_task.get_next_step()
            if not step:
                break

            logger.info(f"Executing step {step['step_id']}: {step['tool']}")
            
            arguments = _substitute_placeholders(step.get("arguments", {}), results)
            
            try:
                result = await tool_service.invoke_tool(step["tool"], arguments)
                current_task.complete_step(result)
                tool_usage.append({"tool": step["tool"], "arguments": arguments, "result": result})
            except Exception as e:
                error_message = f"Error executing tool {step['tool']}: {e}"
                logger.error(error_message)
                current_task.complete_step({"error": error_message})
                tool_usage.append({"tool": step["tool"], "arguments": arguments, "result": {"error": error_message}})
                break 
        
        context_manager.set_current_task(session_id, current_task)

        summary_prompt = get_summarizer_prompt(current_task.original_query, current_task)
        final_response = await llm_manager.generate_response(
            summary_prompt, 
            max_tokens=2000,
            temperature=0.2 # Lower temperature for summarizing
        )
        
        await context_manager.add_assistant_message(user_id, session_id, final_response.content)

        if current_task.is_finished():
            context_manager.set_current_task(session_id, None)

        return ChatResponse(
            response=final_response.content,
            session_id=session_id,
            tool_usage=tool_usage,
            provider=final_response.provider,
            model=final_response.model,
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Session and History Management Endpoints ---

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Chat service is running"}

@router.get("/chat/sessions")
async def get_sessions(context_manager: ContextManager = Depends(get_context_manager)):
    """Get all active chat sessions and their stats."""
    try:
        return context_manager.get_session_stats()
    except Exception as e:
        logger.error(f"Error getting sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions/{session_id}/history", response_model=ConversationHistory)
async def get_conversation_history(
    session_id: str, 
    rag_service: Any = Depends(get_rag_service),
    limit: int = 50
):
    """Get conversation history for a session from persistent storage."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="History service (RAG) not available")
    
    try:
        user_id = "default_user" # TODO: Replace with real user auth
        history_messages = await rag_service.get_chat_history(user_id, session_id, limit=limit)
        return ConversationHistory(
            session_id=session_id,
            history=history_messages,
            message_count=len(history_messages)
        )
    except Exception as e:
        logger.error(f"Error getting conversation history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    context_manager: ContextManager = Depends(get_context_manager)
):
    """Delete a chat session from memory."""
    try:
        if not context_manager.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
