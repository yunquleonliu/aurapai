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

from core.config import settings
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
    use_public_llm: bool = Field(False, description="If true, bypasses the local agent and sends the query directly to the public LLM.")

from typing import Union

class ChatResponse(BaseModel):
    response: Union[str, Dict[str, Any]]
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

def get_planner_prompt(user_query: str, tools: List[Dict[str, Any]]) -> str:
    """Creates the prompt to ask the LLM to generate a plan."""
    tool_list = json.dumps(tools, indent=2)
    
    system_prompt = f"""You are a master planning agent. Your goal is to create a step-by-step plan to fulfill the user's request.
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
    
    return f"{system_prompt}\n\nUser Query: {user_query}"

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

def get_react_prompt(user_query: str, tools: List[Dict[str, Any]], history: List[Dict[str, str]], scratchpad: str) -> str:
    """Creates the prompt for the ReAct agent."""
    tool_list = json.dumps(tools, indent=2)
    
    # Simplified history formatting
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])


    # More robust, explicit prompt for local LLMs
    prompt = f"""
You are a helpful, step-by-step AI assistant that can use tools to solve user queries. You must always output BOTH a Thought and an Action in the following format:

**Thought:** [your reasoning here]
**Action:**
```json
{{
  "tool": "tool_name",
  "arguments": {{ ... }}
}}
```

If you are unsure, always output a valid JSON action for a tool or use the 'clarification' tool. Never invent tools not in the list. If you have the final answer, use the 'finish' tool.

**Tools:**
You have access to the following tools. Only use these tools.
{tool_list}

**Conversation History:**
{formatted_history}

**Instructions:**
To answer the query, you must follow a cycle of Thought, Action, Observation.
1. **Thought:** Reason about the user's query and decide which tool to use (if any) to answer it.
2. **Action:** Output a single JSON object for the tool you want to use. The JSON must have a "tool" key and an "arguments" key, and must be inside a markdown code block (```json ... ```).
3. **FINISH:** When you have the final answer, use the special tool "finish" with the "answer" argument.

**Your Internal Monologue (Previous Steps):**
This is your scratchpad. It shows your previous thoughts, actions, and their results. Use it to decide your next step.
{scratchpad}

**New Task:**
User Query: {user_query}

Now, generate your next thought and action based on the user query and your internal monologue.

**Thought:**"""
    return prompt


def get_summarizer_prompt(user_query: str, task: MultiStepTask) -> str:
    """Creates the prompt for the summarizer LLM as a single string."""
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

    system_prompt = "You are a helpful assistant. Your task is to synthesize the results of a multi-step operation into a single, coherent, and user-friendly response of about 300 words. The user will provide their original query and the results of each step. Do not just list the results; explain what was done and what the outcome is. If an image was generated, mention it. If a search was performed, summarize the findings."
    
    user_prompt = f"""Original Query: {user_query}

Execution Results:
{full_summary}
"""
    # Combine into a single string for the LLM
    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

# --- Core Chat Endpoint ---

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_manager: LLMProvider = Depends(get_llm_manager),
    context_manager: ContextManager = Depends(get_context_manager),
    tool_service: ToolService = Depends(get_tool_service),
):
    """
    Main chat endpoint. 
    Routes to the appropriate agent based on the AGENT_MODE setting and bypass flag.
    """
    user_id = request.user_id
    session_id = request.session_id or context_manager.create_session()
    
    if not context_manager.get_context(session_id):
        context_manager.create_session(session_id)
        
    await context_manager.add_user_message(user_id, session_id, request.message)

    # --- Public LLM Bypass ("Public Send") ---
    if request.use_public_llm:
        logger.info("--- Bypassing to Public LLM (Gemini) ---")
        try:
            history = await context_manager.get_conversation_history(user_id, session_id, limit=10)
            
            response_generator = llm_manager.generate_stream(
                prompt=request.message,
                provider=LLMProvider.GEMINI,
                history=history,
                temperature=0.7,
                max_tokens=2048
            )
            
            # For simplicity in this example, we aggregate the stream. 
            # A real implementation would stream this to the client.
            full_response = ""
            async for chunk in response_generator:
                full_response += chunk

            await context_manager.add_assistant_message(user_id, session_id, full_response)
            return ChatResponse(
                response=full_response,
                session_id=session_id,
                provider=LLMProvider.GEMINI.value,
                model=settings.GEMINI_MODEL
            )
        except Exception as e:
            logger.error(f"Error during public LLM bypass: {e}")
            raise HTTPException(status_code=500, detail="Error communicating with the public LLM.")

    # --- Route to the configured agent ---
    # Only use ReAct agent for local LLMs; Plan-and-Execute is only for public LLMs
    if (settings.AGENT_MODE == "ReAct") or (not request.use_public_llm):
        logger.info("--- Running in ReAct Mode (Local LLM) ---")
        try:
            session_context = context_manager.get_context(session_id)
            if not session_context:
                raise HTTPException(status_code=404, detail="Session context not found.")

            history = await context_manager.get_conversation_history(user_id, session_id, limit=10)
            tools = tool_service.get_tool_definitions()

            # Add a special "finish" tool for the ReAct agent
            tools.append({
                "name": "finish",
                "description": "Use this tool to provide the final answer to the user when you have completed the task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final answer to the user's query."}
                    },
                    "required": ["answer"]
                }
            })

            max_turns = 10  # Prevent infinite loops

            for turn in range(max_turns):
                scratchpad = session_context.get_react_scratchpad()

                # 1. THINK: Generate a thought and action
                react_prompt = get_react_prompt(request.message, tools, history, scratchpad)

                logger.info(f"ReAct Turn {turn + 1}: Generating action...")
                llm_response = await llm_manager.generate_response(
                    prompt=react_prompt,
                    provider=LLMProvider.LLAMACPP,
                    temperature=0.2,
                    max_tokens=1024,
                    stop=["\nObservation:", "\n**Observation:**", "Observation:"]
                )

                # Extract thought and action from the response
                raw_text = llm_response.content
                logger.debug(f"LLM Raw Response for ReAct Action:\n---\n{raw_text}\n---")

                # Regex to handle optional markdown bolding
                thought_match = re.search(r"\*\*?Thought:\*\*?\s*(.*?)\s*\*\*?Action:\*\*?", raw_text, re.DOTALL)
                action_match = re.search(r"\*\*?Action:\*\*?\s*```json\s*(.*?)\s*```", raw_text, re.DOTALL)

                if not thought_match or not action_match:
                    # Fallback for models that might just output the JSON action
                    try:
                        action_json_str = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL).group(1)
                        action = json.loads(action_json_str)
                        thought = "Thought was not explicitly generated, proceeding with action."
                        logger.warning("Could not parse thought, but found a valid action JSON.")
                    except (AttributeError, json.JSONDecodeError):
                        logger.error(f"Could not parse thought/action from response: {raw_text}")
                        # Fallback: ask user for clarification
                        final_response = "Sorry, I couldn't understand the last step. Could you clarify or rephrase your request?"
                        await context_manager.add_assistant_message(user_id, session_id, final_response)
                        # Log the full LLM output for debugging
                        logger.error(f"Raw LLM output (unparsed): {raw_text}")
                        return ChatResponse(response=final_response, session_id=session_id)
                else:
                    thought = thought_match.group(1).strip()
                    action_str = action_match.group(1).strip()
                    try:
                        action = json.loads(action_str)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON for action: {action_str}")
                        final_response = "Sorry, the agent generated an invalid action. Could you rephrase your request?"
                        await context_manager.add_assistant_message(user_id, session_id, final_response)
                        logger.error(f"Raw LLM output (invalid JSON): {raw_text}")
                        return ChatResponse(response=final_response, session_id=session_id)

                logger.info(f"Thought: {thought}")
                logger.info(f"Action: {action}")

                # 2. ACT: Execute the action
                if action.get("tool") == "finish":
                    final_response = action.get("arguments", {}).get("answer", "Task completed.")
                    # If the answer is a dict (e.g., a tool action), extract a string summary
                    if isinstance(final_response, dict):
                        if "message" in final_response:
                            final_response = final_response["message"]
                        else:
                            final_response = str(final_response)
                    session_context.clear_react_state() # Clean up for the next query
                    break

                try:
                    # Special handling for calling the public LLM
                    if action.get("tool") == "call_public_llm":
                        public_prompt = action.get("arguments", {}).get("query", "")
                        observation = await llm_manager.generate_response(
                            prompt=public_prompt,
                            provider=LLMProvider.GEMINI,
                            history=[]
                        )
                        observation = observation.content
                    else:
                        observation = await tool_service.invoke_tool(
                            action.get("tool"),
                            action.get("arguments", {})
                        )
                except Exception as e:
                    observation = f"Error executing tool '{action.get('tool')}': {e}"
                    logger.error(observation)

                # 3. OBSERVE: Check if observation is a present-to-user tool result

                logger.info(f"Observation: {observation}")

                present_types = {"image", "video", "clarification"}
                # --- PATCH: Present web_search results to user ---
                is_web_search_result = (
                    isinstance(observation, dict)
                    and "search_results" in observation
                    and isinstance(observation["search_results"], list)
                )

                if isinstance(observation, dict) and observation.get("type") in present_types:
                    # Save only a summary (not base64) in the scratchpad
                    summary = observation.get("message", "")
                    session_context.update_react_scratchpad(thought, action, summary)
                    await context_manager.add_assistant_message(user_id, session_id, summary)
                    # Return the full observation dict in the response for the frontend
                    return ChatResponse(
                        response=observation,
                        session_id=session_id,
                        tool_usage=[{"tool": action.get("tool"), "arguments": action.get("arguments", {}), "result": observation}],
                        provider="tool",
                        model=observation.get("type")
                    )
                elif is_web_search_result:
                    # Summarize top 3 search results for the user
                    results = observation["search_results"][:3]
                    summary_lines = []
                    for i, r in enumerate(results, 1):
                        title = r.get("title") or r.get("text") or "(no title)"
                        url = r.get("url") or r.get("link") or ""
                        summary_lines.append(f"{i}. {title}\n{url}")
                    summary = "Web search results:\n" + "\n\n".join(summary_lines)
                    session_context.update_react_scratchpad(thought, action, summary)
                    await context_manager.add_assistant_message(user_id, session_id, summary)
                    # Return the full observation dict in the response for the frontend (for possible future UI rendering)
                    return ChatResponse(
                        response={"type": "web_search", "results": results, "summary": summary},
                        session_id=session_id,
                        tool_usage=[{"tool": action.get("tool"), "arguments": action.get("arguments", {}), "result": observation}],
                        provider="tool",
                        model="web_search"
                    )
                # Otherwise, update scratchpad and continue
                summary = str(observation)
                if isinstance(observation, dict) and "base64_data" in observation:
                    summary = observation.get("message", "")
                session_context.update_react_scratchpad(thought, action, summary)

            else:
                final_response = "The agent could not complete the task within the allowed number of steps."

            await context_manager.add_assistant_message(user_id, session_id, final_response)
            return ChatResponse(response=final_response, session_id=session_id)

        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error in ReAct agent: {e}")

    # --- Plan-and-Execute Agent Logic (Public LLM only) ---
    logger.info("--- Running in Plan-and-Execute Mode (Public LLM) ---")
    try:
        current_task = context_manager.get_current_task(session_id)

        if not current_task or current_task.is_finished():
            tools = tool_service.get_tool_definitions()

            # The planner prompt now includes the user's query.
            # We pass this directly to the public LLM.
            planner_prompt = get_planner_prompt(request.message, tools)

            logger.info("Generating a new plan...")
            plan_response = await llm_manager.generate_response(
                prompt=planner_prompt, # Pass the structured prompt as a single string
                provider=LLMProvider.GEMINI, # Use the public LLM for planning
                history=[], # History is now part of the prompt
                max_tokens=1024,
                temperature=0.2  # Lower temperature for planning
            )
            plan_steps = parse_plan_from_response(plan_response.content)

            if not plan_steps:
                logger.info("No plan generated. Responding directly.")
                direct_response = await llm_manager.generate_response(
                    prompt=request.message,
                    provider=LLMProvider.GEMINI, # Use public LLM for direct answers
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

                # If the result is a present-to-user type, return immediately (like ReAct)
                present_types = {"image", "video", "clarification"}
                if isinstance(result, dict) and result.get("type") in present_types:
                    # Save only a summary (not base64) in the scratchpad/history
                    summary = result.get("message", "")
                    await context_manager.add_assistant_message(user_id, session_id, summary)
                    context_manager.set_current_task(session_id, current_task)
                    # Return the full result dict in the response for the frontend
                    return ChatResponse(
                        response=result,
                        session_id=session_id,
                        tool_usage=tool_usage,
                        provider="tool",
                        model=result.get("type")
                    )
            except Exception as e:
                error_message = f"Error executing tool {step['tool']}: {e}"
                logger.error(error_message)
                current_task.complete_step({"error": error_message})
                tool_usage.append({"tool": step["tool"], "arguments": arguments, "result": {"error": error_message}})
                break

        context_manager.set_current_task(session_id, current_task)

        summary_prompt_str = get_summarizer_prompt(current_task.original_query, current_task)
        final_response = await llm_manager.generate_response(
            provider=LLMProvider.GEMINI, # Use public LLM for summarizing
            prompt=summary_prompt_str,
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
