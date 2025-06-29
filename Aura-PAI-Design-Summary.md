# Aura-PAI Backend Design Document

## Overview
Aura-PAI is a privacy-compliant, robust, and unified conversational AI platform supporting both web and mobile clients. It features persistent, searchable chat history using ChromaDB, tool invocation (including image generation/interpretation), and a responsive, modern UI.

---

> **Note:** The backend API server runs on port **8001** by default. All API endpoints below assume `http://<host>:8001` as the base URL.

## Recent Enhancements (June 2025)

### 1. Chat History Management
- **ChromaDB** is now used for persistent chat history (replacing in-memory storage).
- **API Endpoints**: Added/updated endpoints for chat history retrieval, search, deletion (per-session and all), and export.
- **Privacy**: All history is stored server-side with privacy/access control stubs; users can delete/export their data at any time.
- **Frontend**: New sidebar/tab UI for history, with search/filter, session listing, and responsive design for web/mobile.

### 2. Tool Integration & Intent Detection
- **Tool API**: Tools (including image generation/interpretation) are exposed via backend API and listed in the UI.
- **Intent Detection**: LLM-based intent detection routes user queries to tools or chat as appropriate. Natural language triggers for tool invocation are planned.
- **Tool Definitions**: `ToolService.get_tool_definitions` now returns real tool schemas for frontend/tool search.

### 3. Image Generation/Interpretation
- **Endpoints**: `/api/v1/chat/generate-image` and `/api/v1/chat/generate-images` support single/multi-image generation from prompts or menu images.
- **Frontend**: Users can attach images, generate images from prompts, or generate dish images from menu photos. Results are shown inline with chat.

### 4. UI/UX Improvements
- **Unified UI**: Responsive layout with sidebar (desktop) and tab (mobile) for chat history.
- **History UI**: Compact, side-by-side layout for session list and message list. Search/filter and per-session delete/export.
- **Privacy Notice**: Clear notice in UI; all history can be deleted/exported.

---

## Enhancement Plan (Q3 2025)

1. **Expose Image Generation/Interpretation as Tools**
   - API endpoints and backend logic for image tools.
   - Tool schemas returned in `/api/v1/tools`.
   - UI: Tool search/invocation from chat.

2. **List All Chat Sessions in UI**
   - Fetch all sessions from ChromaDB.
   - UI: Sidebar/tab lists all sessions, with search/filter.

3. **Redesign History/Search UI**
   - Compact, side-by-side layout for session/message list.
   - Improved search/filter UX.

4. **Enhance Intent Detection for Tool Invocation**
   - Support natural language triggers (not just explicit tool names).
   - Document and test intent detection workflow.

5. **Testing & Verification**
   - Test all enhancements (API, UI, privacy, tool invocation).
   - Update API docs and user documentation.

---

## API Endpoints (Summary)
- `POST /api/v1/chat` — Main chat endpoint (text, tool, or image input)
- `POST /api/v1/chat/image` — Image upload/analysis
- `POST /api/v1/chat/generate-image` — Generate image from prompt
- `POST /api/v1/chat/generate-images` — Generate multiple images from menu
- `GET /api/v1/chat/history` — List/search chat history (with keyword, session_id)
- `DELETE /api/v1/chat/history` — Delete all or per-session history
- `GET /api/v1/chat/history/export` — Export session history
- `GET /api/v1/tools` — List available tools and schemas

---

## Privacy & Access Control
- All chat history is stored server-side (ChromaDB) and can be deleted/exported by the user.
- Access control stubs are in place for future multi-user support.
- Privacy principles are documented and visible in the UI.

---

## Known Issues & Next Steps
- Tool invocation via natural language intent is in progress.
- Some tool APIs (e.g., image tools) are being refactored for unified invocation.
- UI/UX for history and tool search will be further improved.

---

## Contributors
- Backend: [Yunqu Leon Liu yunqu.liu@gmail.lcom]
- Frontend: [Yunqu Leon Liu yunqu.liu@gmail.com]
- Design/Docs: [Yunqu Leon Liu yunqu.liu@gmail.com]

---

_Last updated: 2025-06-29_
