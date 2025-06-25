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
from fastapi.responses import JSONResponse, StreamingResponse
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

def get_knowledge_service(request: Request) -> Any:
    """Dependency injector for the KnowledgeService."""
    return getattr(request.app.state, 'knowledge_service', None)


# --- Intent Detection Helper ---
def detect_intent(user_message: str) -> str:
    """
    Returns one of: 'local_llm', 'public_llm', 'web_search', 'multimedia', 'ambiguous'
    """
    msg = user_message.lower()
    # Explicit local LLM
    if re.search(r"(use local|local llm|private | pai | private model|no web search|no tools|local knowledge only)", msg):
        return "local_llm"
    # Explicit public LLM
    if re.search(r"(use public|public llm|gemini|gpt|openai|more powerful|external model)", msg):
        return "public_llm"
    # Explicit web search
    if re.search(r"(search the web|find latest|current news|recent|lookup|google|duckduckgo|search for)", msg):
        return "web_search"
    # Multimedia
    if re.search(r"(image|picture|photo|screenshot|diagram|graph|chart|video|audio|sound|voice)", msg):
        return "multimedia"
    # If ambiguous, let LLM decide
    return "ambiguous"

def get_planner_prompt(user_query: str, tools: List[Dict[str, Any]], rag_context: str = "") -> str:
    """Creates the prompt to ask the LLM to generate a plan."""
    tool_list = json.dumps(tools, indent=2)
    
    context_section = ""
    if rag_context:
        context_section = f"""**Knowledge Base Context:**
You have the following information from a private knowledge base. Use it to inform your plan if relevant.
{rag_context}
"""

    system_prompt = f"""You are a master planning agent. Your goal is to create a step-by-step plan to fulfill the user\'s request.
You have the following tools available:
{tool_list}

{context_section}

- Based on the user\'s query, create a JSON object with a 'plan' key.
- The 'plan' must be a list of steps. Each step is a dictionary with 'tool' and 'arguments'.
- Each step must have a unique 'step_id' starting from 0.
- If a step requires the output of a previous step, use a placeholder like '<step_0_output>' corresponding to the step_id.
- If the query is simple and doesn\'t require any tools (e.g., "hello"), respond with an empty plan: {{"plan": []}}.
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

def get_react_prompt(user_query: str, tools: List[Dict[str, Any]], history: List[Dict[str, str]], scratchpad: str, rag_context: str = "") -> str:
    """Creates the prompt for the ReAct agent."""
    tool_list = json.dumps(tools, indent=2)
    
    # Simplified history formatting
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])


    context_section = ""
    if rag_context:
        context_section = f"""**Knowledge Base Context:**
You have the following information from a private knowledge base. Use it to answer the user's query if relevant.
{rag_context}
"""

    # More robust, explicit prompt for local LLMs
    prompt = f"""You are a helpful, step-by-step AI assistant that can use tools to solve user queries. You must always output BOTH a Reasoning and an Action in the following format:

**Reasoning:** [your reasoning here]
**Action:**
```json
{{
  "tool": "tool_name",
  "arguments": {{ ... }}
}}
```

Instructions for tool use:
- Only use the web_search tool if you do NOT know the answer or if the information is likely to be recent, up-to-date, or not in your training data. If you know the answer, respond directly using your own knowledge.
- Use the call_public_llm tool for complex reasoning, creative tasks, or when the user requests a more powerful or public LLM.
- If the user explicitly says to use the local LLM, you must answer directly using your own knowledge and NOT use any tools.
- If you are unsure, use the clarification tool to ask the user for more information.
- Never invent tools not in the list. If you have the final answer, use the 'finish' tool.

**Tools:**
You have access to the following tools. Only use these tools.
{tool_list}

{context_section}

**Conversation History:**
{formatted_history}

**Instructions:**
To answer the query, you must follow a cycle of Reasoning, Action, Observation.
1. **Reasoning:** Summarize your reasoning about the user's query and decide which tool to use (if any) to answer it.
2. **Action:** Output a single JSON object for the tool you want to use. The JSON must have a "tool" key and an "arguments" key, and must be inside a markdown code block (```json ... ```).
3. **FINISH:** When you have the final answer, use the special tool "finish" with the "answer" argument.

**Your Internal Monologue (Previous Steps):**
This is your scratchpad. It shows your previous reasoning, actions, and their results. Use it to decide your next step.
{scratchpad}

**New Task:**
User Query: {user_query}

Now, generate your next reasoning and action based on the user query and your internal monologue.

**Reasoning:**"""
    return prompt


def get_summarizer_prompt(user_query: str, task: MultiStepTask, rag_context: str = "") -> str:
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

    context_section = ""
    if rag_context:
        context_section = f"""**Knowledge Base Context:**
The following information was retrieved from a private knowledge base. Use it to help formulate your final answer.
{rag_context}
"""

    system_prompt = "You are a helpful assistant. Your task is to synthesize the results of a multi-step operation into a single, coherent, and user-friendly response of about 300 words. The user will provide their original query and the results of each step. Do not just list the results; explain what was done and what the outcome is. If an image was generated, mention it. If a search was performed, summarize the findings."
    
    user_prompt = f"""Original Query: {user_query}

{context_section}

Execution Results:
{full_summary}
"""
    # Combine into a single string for the LLM
    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    # Stream 

    async def stream_response():
        async for chunk in llm_manager.generate_stream(
            prompt=prompt,
            provider=LLMProvider.LLAMACPP,
            history=history,
            temperature=0.7,
            max_tokens=2048
        ):
            if chunk is not None and str(chunk).strip() != "":
                yield str(chunk) + "\n"

    return StreamingResponse(stream_response(), media_type="text/plain")
# --- Core Chat Endpoint: Single-Cycle ReAct with Streaming for 'finish' ---
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_manager: LLMProvider = Depends(get_llm_manager),
    context_manager: ContextManager = Depends(get_context_manager),
    tool_service: ToolService = Depends(get_tool_service),
    knowledge_service: Any = Depends(get_knowledge_service),
    max_turns: int = 1,  # Configurable for future multi-cycle support
):
    """
    Main chat endpoint.
    - Single-cycle ReAct: One reasoning-action-observation per request (default).
    - Streams LLM output for 'finish' actions if answer is long.
    - Skips ReAct/tool use for simple queries (intent detection).
    - max_turns param allows future multi-cycle support.
    """
    intent = detect_intent(request.message)
    user_id = request.user_id
    session_id = request.session_id or context_manager.create_session()
    if not context_manager.get_context(session_id):
        context_manager.create_session(session_id)
    await context_manager.add_user_message(user_id, session_id, request.message)

    # --- RAG Integration: Retrieve context from Knowledge Base ---
    rag_context = ""
    if knowledge_service:
        try:
            if knowledge_service.list_documents():
                retriever = knowledge_service.get_retriever()
                if retriever:
                    docs = await retriever.aget_relevant_documents(request.message)
                    if docs:
                        rag_context += "\n\n--- Knowledge Base Context ---\n"
                        for doc in docs:
                            source = doc.metadata.get('source', 'N/A').split('/')[-1]
                            rag_context += f"Source: {source}\nContent: {doc.page_content}\n---\n"
        except Exception as e:
            logger.error(f"Error retrieving documents from knowledge base: {e}")


    # --- Streaming Mode: Direct Model Call ---
    if intent in ["local_llm", "public_llm"] or request.use_public_llm:
        provider = LLMProvider.LLAMACPP if intent == "local_llm" else LLMProvider.GEMINI
        history = await context_manager.get_conversation_history(user_id, session_id, limit=10)
        prompt = request.message
        async def stream_llm():
            async for chunk in llm_manager.generate_stream(
                prompt=prompt,
                provider=provider,
                history=history,
                temperature=0.7,
                max_tokens=2048
            ):
                if chunk is not None and str(chunk).strip() != "":
                    yield str(chunk)
        return StreamingResponse(stream_llm(), media_type="text/plain")

    # --- ReAct Mode: Tool Use, Web Search, Multimedia, Ambiguous ---
    try:
        session_context = context_manager.get_context(session_id)
        if not session_context:
            return JSONResponse(status_code=404, content={"error": "Session context not found."})
        history = await context_manager.get_conversation_history(user_id, session_id, limit=10)
        tools = tool_service.get_tool_definitions()
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
        scratchpad = session_context.get_react_scratchpad()
        react_prompt = get_react_prompt(request.message, tools, history, scratchpad, rag_context)
        import time
        logger.info("ReAct: Generating action (LLAMACPP)...")
        start_time = time.time()
        llm_response = await llm_manager.generate_response(
            prompt=react_prompt,
            provider=LLMProvider.LLAMACPP,
            temperature=0.2,
            max_tokens=1024,
            stop=["\nObservation:", "\n**Observation:**", "Observation:"]
        )
        elapsed = time.time() - start_time
        logger.info(f"[ReAct Debug] LLAMACPP LLM call completed in {elapsed:.2f} seconds.")

        # Extract thought and action from the response
        raw_text = llm_response.content
        logger.info(f"[ReAct Debug] LLMProvider.LLAMACPP raw response (before parsing):\n---\n{raw_text}\n---")
        logger.debug(f"LLM Raw Response for ReAct Action (LLAMACPP):\n---\n{raw_text}\n---")

        # Regex to handle optional markdown bolding
        thought_match = re.search(r"\*\*?Thought:\*\*?\s*(.*?)\s*\*\*?Action:\*\*?", raw_text, re.DOTALL)
        action_match = re.search(r"\*\*?Action:\*\*?\s*```json\s*(.*?)\s*```", raw_text, re.DOTALL)

        if not thought_match or not action_match:
            # Fallback for models that might just output the JSON action
            try:
                action_json = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
                if not action_json:
                    raise AttributeError("No JSON action block found")
                action_json_str = action_json.group(1)
                action = json.loads(action_json_str)
                if 'parameters' in action and 'arguments' not in action:
                    action['arguments'] = action.pop('parameters')
                pre_json = raw_text[:action_json.start()].strip()
                pre_json_lines = [line.strip() for line in pre_json.splitlines() if line.strip() and line.strip() != '---']
                thought = "Reasoning was not explicitly generated, proceeding with action."
                for i in range(len(pre_json_lines) - 1, 0, -1):
                    if re.match(r"\*\*?Action:?\*\*?", pre_json_lines[i], re.IGNORECASE):
                        candidate = pre_json_lines[i-1]
                        if candidate:
                            thought = candidate
                        break
                logger.warning("Could not parse reasoning, but found a valid action JSON.")
            except (AttributeError, json.JSONDecodeError):
                logger.error(f"[ReAct Ambiguous] Could not parse reasoning/action from response: {raw_text}")
                # Instead of error, treat as ambiguous/LLM confusion, send message to frontend
                ambiguous_message = "Sorry, I couldn't understand the last step. Could you clarify or rephrase your request?"
                await context_manager.add_assistant_message(user_id, session_id, ambiguous_message)
                # Add debug info for investigation
                logger.debug(f"[ReAct Ambiguous] Sending ambiguous message to frontend. Raw LLM output: {raw_text}")
                return ChatResponse(
                    response=ambiguous_message,
                    session_id=session_id
                )
        else:
            thought = thought_match.group(1).strip()
            action_str = action_match.group(1).strip()
            try:
                action = json.loads(action_str)
                if 'parameters' in action and 'arguments' not in action:
                    action['arguments'] = action.pop('parameters')
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON for action: {action_str}")
                final_response = "Sorry, the agent generated an invalid action. Could you rephrase your request?"
                await context_manager.add_assistant_message(user_id, session_id, final_response)
                return JSONResponse(status_code=500, content={"error": final_response})

        logger.info(f"Thought: {thought}")
        logger.info(f"Action: {action}")

        # 2. ACT: Execute the action (single cycle)
        if action.get("tool") == "finish":
            final_response = action.get("arguments", {}).get("answer", "Task completed.")
            # If the answer is a string and long, stream it directly; else, return as normal
            if isinstance(final_response, str) and len(final_response) > 300:
                from services.llm_manager import LLMManager
                session_context.clear_react_state()
                await context_manager.add_assistant_message(user_id, session_id, "[streamed answer]")
                return StreamingResponse(LLMManager.stream_answer(final_response), media_type="text/plain")
            else:
                if isinstance(final_response, dict):
                    if "message" in final_response:
                        final_response = final_response["message"]
                    else:
                        final_response = str(final_response)
                session_context.clear_react_state()
                await context_manager.add_assistant_message(user_id, session_id, final_response)
                return ChatResponse(
                    response=final_response,
                    session_id=session_id
                )

        try:
            if action.get("tool") == "call_public_llm":
                public_prompt = action.get("arguments", {}).get("query", "")
                observation = await llm_manager.generate_response(
                    prompt=public_prompt,
                    provider=LLMProvider.LLAMACPP,
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

        logger.info(f"Observation: {observation}")

        present_types = {"image", "video", "clarification"}
        is_web_search_result = (
            isinstance(observation, dict)
            and "search_results" in observation
            and isinstance(observation["search_results"], list)
        )

        if isinstance(observation, dict) and observation.get("type") in present_types:
            summary = observation.get("message", "")
            session_context.update_react_scratchpad(thought, action, summary)
            await context_manager.add_assistant_message(user_id, session_id, summary)
            return ChatResponse(
                response=observation,
                session_id=session_id,
                tool_usage=[{"tool": action.get("tool"), "arguments": action.get("arguments", {}), "result": observation}],
                provider="tool",
                model=observation.get("type")
            )
        elif is_web_search_result:
            results = observation["search_results"][:5]
            summary_lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title") or r.get("text") or "(no title)"
                url = r.get("url") or r.get("link") or ""
                summary_lines.append(f"{i}. {title}\n{url}")
            summary = "Web search results:\n" + "\n\n".join(summary_lines)
            session_context.update_react_scratchpad(thought, action, summary)
            await context_manager.add_assistant_message(user_id, session_id, summary)
            return ChatResponse(
                response={"type": "web_search", "results": results, "summary": summary},
                session_id=session_id,
                tool_usage=[{"tool": action.get("tool"), "arguments": action.get("arguments", {}), "result": observation}],
                provider="tool",
                model="web_search"
            )
        # Otherwise, update scratchpad and return observation as string
        summary = str(observation)
        if isinstance(observation, dict) and "base64_data" in observation:
            summary = observation.get("message", "")
        session_context.update_react_scratchpad(thought, action, summary)
        await context_manager.add_assistant_message(user_id, session_id, summary)
        return ChatResponse(
            response=summary,
            session_id=session_id,
            tool_usage=[{"tool": action.get("tool"), "arguments": action.get("arguments", {}), "result": observation}],
            provider="tool",
            model=action.get("tool")
        )
    except Exception as e:
        logger.error(f"Error in single-cycle ReAct agent (LLAMACPP): {e}", exc_info=True)
        # Always return JSON error, never HTML
        return JSONResponse(status_code=500, content={"error": f"Error in ReAct agent (LLAMACPP): {str(e)}"})

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

# --- Local LLM Only Streaming Chat Route ---
@router.post("/chat/local/stream")
async def local_chat_stream(
    request: ChatRequest,
    llm_manager: LLMProvider = Depends(get_llm_manager),
    context_manager: ContextManager = Depends(get_context_manager),
    knowledge_service: Any = Depends(get_knowledge_service),
):
    """
    Local LLM only: Uses only local LLM (LLAMACPP), with chat history and RAG context, no tool use, streams response.
    """
    user_id = request.user_id
    session_id = request.session_id or context_manager.create_session()
    if not context_manager.get_context(session_id):
        context_manager.create_session(session_id)
    await context_manager.add_user_message(user_id, session_id, request.message)

    # Gather RAG context if available
    rag_context = ""
    if knowledge_service:
        try:
            if knowledge_service.list_documents():
                retriever = knowledge_service.get_retriever()
                if retriever:
                    docs = await retriever.aget_relevant_documents(request.message)
                    if docs:
                        rag_context += "\n\n--- Knowledge Base Context ---\n"
                        for doc in docs:
                            source = doc.metadata.get('source', 'N/A').split('/')[-1]
                            rag_context += f"Source: {source}\nContent: {doc.page_content}\n---\n"
        except Exception as e:
            logger.error(f"Error retrieving documents from knowledge base: {e}")


    history = await context_manager.get_conversation_history(user_id, session_id, limit=10)
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = f"""You are a helpful assistant with access to local knowledge and private documents.\n{rag_context}\n\nConversation history:\n{formatted_history}\n\nUser: {request.message}\nAssistant:"""

    async def stream_response():
        async for chunk in llm_manager.generate_stream(
            prompt=prompt,
            provider=LLMProvider.LLAMACPP,
            history=history,
            temperature=0.7,
            max_tokens=2048
        ):
            if chunk is not None and str(chunk).strip() != "":
                yield str(chunk)

    return StreamingResponse(stream_response(), media_type="text/plain")

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
