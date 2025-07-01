# Aura-PAI Agent Architecture Design

## 1. Core Challenge: Stateful, Multi-Step Conversations

The primary goal of Aura-PAI is to handle complex user requests that require multiple steps, tool usage, and access to external information (e.g., "Find recent news about NASDAQ, summarize the key points, and tell me if now is a good time to invest."). This requires a sophisticated agent architecture that can manage state, reason about a problem, and execute a plan.

A central challenge is the trade-off between personalization and privacy. A more personalized AI needs to remember user context and history, but this raises privacy concerns. Aura-PAI's architecture is designed to address this by giving the user control over their data and how it is used.

## 2. Agent Implementation Strategy: Feature-Flagged Evolution

To ensure stability while innovating, we are adopting a feature-flag-driven approach. The application supports two distinct agent architectures, controlled by the `AGENT_MODE` setting in `core/config.py`.

-   `AGENT_MODE = "Plan-and-Execute"`: (Default, Stable) Uses the original, robust agent that creates a full plan upfront and executes it step-by-step.
-   `AGENT_MODE = "ReAct"`: (In Development) Uses the next-generation hierarchical agent that combines a local, privacy-focused LLM with a powerful public LLM in a dynamic, step-by-step reasoning loop.

This strategy allows for the safe, parallel development and testing of the new ReAct agent without disrupting the currently deployed and working `Plan-and-Execute` agent.

## 3. Architecture 1: The Plan-and-Execute Agent

This is the current, stable architecture for handling multi-step tasks.

### Core Workflow

The workflow consists of three main stages: Decomposition (Planning), State Management, and Execution.

#### a. Decomposition (Planning)

-   **Goal:** An initial LLM call creates a complete, structured plan in JSON format.
-   **Prompting:** The LLM is instructed to act as a planner, breaking the user's request into a sequence of tool calls.

#### b. State Management (Context Manager)

-   **Goal:** To track the progress of the multi-step plan.
-   **Implementation:** A `MultiStepTask` object is stored in the `ConversationContext` to hold the plan, the current step, and the results of each action.

#### c. Execution Loop (Orchestrator)

-   **Goal:** To execute the plan step-by-step.
-   **Logic:** The chat endpoint in `api/routes/chat.py` iterates through the plan, calls the necessary tools via `tool_service.py`, and stores the results until the plan is complete. A final summary is then generated.

## 4. Architecture 2: The Hierarchical ReAct Agent (In Development)

This is the next evolution of our architecture, designed for greater efficiency, adaptability, and a clearer separation of private and public data processing.

### Core Concept: Hierarchical LLM Roles

This model uses two distinct types of LLMs:

-   **Local/Private LLM (Personal Orchestrator):** A local model that manages the direct user interaction. It operates on a **ReAct (Reason-Act)** framework, deciding the single next best action in a loop. It has access to private, long-term user history and acts as a smart, privacy-aware router.
-   **Remote/Public LLM (Expert Specialist):** A powerful, public model (e.g., Gemini) treated as a stateless "tool." It handles complex reasoning and knowledge-intensive tasks, receiving only curated, context-specific prompts from the local orchestrator.

### Core Loop: Think -> Act -> Observe

Instead of a rigid upfront plan, the Local LLM engages in a dynamic loop:

1.  **Think (Reason):** Decide the single next best action (e.g., "I need to search the web for reviews").
2.  **Act (Execute):** Execute that one action (e.g., call the `web_search` tool or call the Remote LLM tool).
3.  **Observe (Get Result):** Receive the result and add it to the context as an "observation."
4.  **Repeat:** The loop continues until the Local LLM determines it has enough information to provide a final answer.

### User Interaction Workflow

The user is given explicit control over which model to use:

1.  **`Send` (Default Interaction):** The query goes to the **Local LLM**, which begins its ReAct loop. It may use its own tools or decide to formulate a prompt for the **Remote LLM**.
2.  **`Public Send` (Explicit Bypass):** The query is sent **directly** to the **Remote LLM** for a powerful, one-shot, stateless response.

### 4.1 Intent-Driven Tool Invocation (Robust ReAct Design)

**Motivation:**

-   Local LLMs are often unreliable at generating perfectly formatted tool-call JSON, and may hallucinate parameters or produce malformed output.
-   To maximize reliability, the backend should handle deterministic, accurate, and stateful tool invocation based on the LLM's interpreted intent, not on raw LLM-generated tool call code.

**How it works:**

-   The Local LLM's job is to identify the user's intent and the tool to use (e.g., `{"tool": "generate_image", "intent": "a picture of a chicken"}` or `{"tool": "web_search", "query": "latest news on NASDAQ"}`).
-   The backend parses this intent, validates and fills in required parameters, and constructs the correct tool call.
-   The backend manages all parameter defaults, schema validation, and stateful execution, ensuring robust and predictable behavior.
-   This approach is more reliable, easier to debug, and safer, especially with local LLMs. It also allows the backend to enforce parameter schemas and handle tool chaining and state management more accurately.

**Example:**

-   LLM output: `{"tool": "generate_image", "intent": "a picture of a chicken"}`
-   Backend logic:
    -   Recognizes the tool as `generate_image`.
    -   Maps `intent` to the `prompt` parameter, sets `num_images=1` by default.
    -   Calls `generate_image(prompt="a picture of a chicken", num_images=1)`.
    -   Handles any errors or missing parameters deterministically.

**Benefits:**

-   No more brittle JSON parsing or hallucinated parameters.
-   Backend is always in control of tool invocation and state.
-   LLM can focus on reasoning and intent, not on perfect syntax.
