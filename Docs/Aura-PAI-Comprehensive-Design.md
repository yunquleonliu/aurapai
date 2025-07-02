# Aura-PAI Comprehensive Design Document

## 1. Overview
Aura-PAI is a privacy-compliant, robust, and unified conversational AI platform supporting both web and mobile clients. It features persistent, searchable chat history using ChromaDB, tool invocation (including image generation/interpretation), and a responsive, modern UI.

### 1.1. Target Audience
- **Non-technical users ("Average Joys"):** Individuals with no background in prompt engineering, software development, or AI model training.
- **Privacy-conscious users:** Those who value their data privacy and are willing to invest in a secure, local-first solution.

### 1.2. High-Level Architecture
The backend employs a modular, layered architecture with FastAPI as the central orchestrator, decoupling the client interface, AI logic, data retrieval, and external tool interactions.

```
+---------------------------+
|   Client (Web UI, etc.)   |
+---------------------------+
             |
             v
+---------------------------+
|    FastAPI Backend        |
| (Python Application Layer)|
+---------------------------+
| - API Endpoints           |
| - Agentic Loop            |
| - Context Management      |
+---------------------------+
      |          |        |
      v          v        v
+----------+ +-------+ +----------+
| LLM      | | RAG   | | Tools    |
| (Local/  | |(Chroma| | (Web,    |
| Public)  | | DB)   | | etc.)    |
+----------+ +-------+ +----------+
```

The backend API server runs on port **8001** by default. All API endpoints below assume `http://<host>:8001` as the base URL.

---

## 2. Core Architectural Challenge: Stateful, Multi-Step Conversations
The primary goal of Aura-PAI is to handle complex user requests that require multiple steps, tool usage, and access to external information (e.g., "Find recent news about NASDAQ, summarize the key points, and tell me if now is a good time to invest."). This requires a sophisticated agent architecture that can manage state, reason about a problem, and execute a plan.

A central challenge is the trade-off between personalization and privacy. A more personalized AI needs to remember user context and history, but this raises privacy concerns. Aura-PAI's architecture is designed to address this by giving the user control over their data and how it is used.

---

## 3. Implemented Features and Capabilities (As of June 2025)

### 3.1. Core Features
- **Local LLM Integration:** Seamless integration with local `llama.cpp` servers (hosting models like LLaVA + Mixtral), with an extensible design to support public LLMs (OpenAI, Gemini).
- **Retrieval-Augmented Generation (RAG):** Integrated with ChromaDB for indexing and semantic search over local codebases and documents.
- **Multi-modal Capabilities:** Supports both image generation (text-to-image) and image interpretation (image-to-text).
- **Tool Integration:** Natively supports web search (DuckDuckGo), URL fetching, and other tools.
- **Stateful Conversation Management:** Manages session history, context window, and memory optimization.

### 3.2. AI Capabilities & Tooling
The platform supports a range of AI capabilities orchestrated through a JSON-based tool-calling protocol:
- `search_and_fetch`: Performs a web search, fetches content, and provides a summary.
- `image_generation`: Generates an image from a textual description.
- `interpret_image`: Analyzes the content of an image and provides a textual description.
- `web_search`: Conducts a real-time web search.
- `url_fetch`: Fetches and extracts content from a given URL.

---

## 4. Agent Implementation Strategy: Feature-Flagged Evolution
To ensure stability while innovating, we are adopting a feature-flag-driven approach. The application supports two distinct agent architectures, controlled by the `AGENT_MODE` setting in `core/config.py`.

-   `AGENT_MODE = "Plan-and-Execute"`: (Default, Stable) Uses the original, robust agent that creates a full plan upfront and executes it step-by-step. **This mode is intended for use with powerful, public LLMs (e.g., Gemini).**
-   `AGENT_MODE = "ReAct"`: (In Development) Uses the next-generation hierarchical agent that combines a local, privacy-focused LLM with a powerful public LLM in a dynamic, step-by-step reasoning loop. **This is the only mode suitable for local LLMs.**

This strategy allows for the safe, parallel development and testing of the new ReAct agent without disrupting the currently deployed and working `Plan-and-Execute` agent.

---

## 5. Architecture 1: The Plan-and-Execute Agent (For Public LLMs)
This is the current, stable architecture for handling multi-step tasks with powerful public LLMs.

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

---

## 6. Architecture 2: The Hierarchical ReAct Agent (For Local LLMs)
This is the next evolution of our architecture, designed for greater efficiency, adaptability, and a clearer separation of private and public data processing. It is the **only** architecture used for local LLMs.

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

### Intent-Driven Tool Invocation (Robust ReAct Design)
**Motivation:** Local LLMs are often unreliable at generating perfectly formatted tool-call JSON. To maximize reliability, the backend handles deterministic, accurate, and stateful tool invocation based on the LLM's interpreted intent, not on raw LLM-generated tool call code.

**How it works:**
-   The Local LLM's job is to identify the user's intent and the tool to use (e.g., `{"tool": "generate_image", "intent": "a picture of a chicken"}`).
-   The backend parses this intent, validates and fills in required parameters, and constructs the correct tool call.
-   This approach is more reliable, easier to debug, and safer, especially with local LLMs.

---

## 7. Core Workflows

### 7.1. Query-Intent-Routing-Output Workflow
This workflow ensures that user queries are handled consistently and predictably.

1.  **Intent Detection:** The backend uses an LLM to classify the user's intent (e.g., `image_generation`, `web_search`, `text_generation`).
2.  **Strict Routing:** The request is routed to the appropriate service based on the detected intent. There is no "fall-through" logic; a request to generate an image will only go to the image generation service.
3.  **Bound Output:** The output type is strictly tied to the intent. An image generation request returns an image, and a text generation request returns text.

### 7.2. Internet-Enhanced RAG Workflow
This workflow allows the LLM to access external information via tools in a closed loop.

1.  **Tool-Calling Prompt:** The LLM is prompted to produce a JSON object specifying the tool and parameters (e.g., `{"action": "search_and_fetch", "query": "latest AI news"}`).
2.  **Backend Execution:** The backend parses the JSON and executes the specified tool.
3.  **Result Injection & Summarization:** The tool's output is injected back into the LLM's context, and the LLM is prompted again to produce a final, summarized answer for the user.

---

## 8. Data Management and Privacy

### 8.1. Chat History Storage
- **Unified Storage:** All chat history is stored in a dedicated **ChromaDB** collection, separate from the RAG knowledge base.
- **Data Model:** Each message is a document containing a session ID, timestamp, content, and type (user, assistant, image, etc.).
- **Privacy First:** Data is stored locally by default. Users have the right to view, export, and delete their history at any time.

### 8.2. Frontend History Management
- **Responsive UI:** The UI uses a side-by-side layout on desktop (session list + chat view) and a tabbed layout on mobile.
- **Features:** Users can search history by keyword, filter by date, and delete or export entire sessions.

---

## 9. Frontend and Multi-Device Integration
- **Web Frontend:** A fully implemented web client provides support for text chat, image uploads, and rendering of generated images.
- **Mobile/Other Clients (Planned):** The backend API is designed to be consumed by future clients, including native Android/iOS apps and a VS Code extension.

---

## 10. API Endpoints (Summary)
- `POST /api/v1/chat` — Main chat endpoint (text, tool, or image input). Supports both ReAct (local LLM) and Plan-and-Execute (public LLM) modes, routed automatically.
- `POST /api/v1/tools/interpret-image` — Image upload/analysis
- `POST /api/v1/chat/generate-image` — Generate image from prompt
- `GET /api/v1/chat/history` — List/search chat history (with keyword, session_id)
- `DELETE /api/v1/chat/history` — Delete all or per-session history
- `GET /api/v1/chat/history/export` — Export session history
- `GET /api/v1/tools` — List available tools and schemas

---

## 11. Known Issues & Next Steps
- **Tool invocation via natural language intent is in progress.** The system currently relies on more explicit tool-calling formats.
- **Enhance multi-tool collaboration and error handling.** Improve the agent's ability to sequence multiple tools and recover from failures.
- **Improve API documentation and test coverage.**
- **Ensure that Plan-and-Execute agent logic is only used for public LLMs, and ReAct is the only mode for local LLMs.** Update code and documentation as needed.

---

## 12. Contributors
- Backend: [Yunqu Leon Liu yunqu.liu@gmail.lcom]
- Frontend: [Yunqu Leon Liu yunqu.liu@gmail.com]
- Design/Docs: [Yunqu Leon Liu yunqu.liu@gmail.com]

---

_Last updated: 2025-07-01_
