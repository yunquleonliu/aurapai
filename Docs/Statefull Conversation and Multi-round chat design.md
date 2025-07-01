# Identify the intention with statefull and multi round conversation.

## Introduction

The LLM model is stateless. The conversational application built around it is stateful, and it maintains this state by managing the conversation history and feeding it back to the model on every turn. Remember we have an open question part?

Here is the issue/value of so called privacy. if our program remember more about user's privacy and chat history, we can manage a better service, but with more privacy, the user is more concerned. that is the motivate of building aura-pai, an Pai for anyoneLLM (not anything LLM). 

the central tension in creating a truly personal AI: the trade-off between personalization and privacy.

 • To provide a highly effective, stateful, multi-turn service (like checking reservations), the AI needs to remember context, preferences, and previous interactions. This is user data.
 
  • The more data it remembers, the more useful it becomes, but also the greater the user's privacy concerns.  The goal of Aura-PAI, as you put it, is to be a "PAI for anyone" (a Personal AI), which means navigating this trade-off is not just a technical challenge, but the core value proposition. The user needs to be in control of what their "Personal" AI knows about them

## What

如何让AI在“帮我查下某些网站的预订/预约情况”这类复杂需求下，真正完成多轮工具调用、信息聚合与总结？

“多轮回合式”智能体编排与真实信息获取，是Auro-PAI面临的核心挑战之一，需持续探索。

## Architecture: The Plan-and-Execute Agent

To address the open question of handling complex, multi-step user requests, we propose moving from a simple, reactive "intent -> tool" model to a proactive, multi-step **Plan-and-Execute Agent** architecture. This approach turns the LLM from a simple answer-bot into a reasoning and planning engine that uses the tools we provide.

### Core Workflow

The workflow consists of three main stages:

#### 1. Decomposition (Planning)

-   **Goal:** Instead of trying to find a single intent, the first LLM call is to create a structured plan.
-   **Mechanism:** We use a specific system prompt to instruct the LLM to act as a planner.
    -   *System Prompt Example:* "You are a helpful planning assistant. Your job is to break down a user's request into a series of discrete, ordered steps that can be executed by tools. The available tools are [`web_search`, `url_fetch`, `summarize_text`]. Respond with a JSON object containing a 'plan' which is a list of steps. Each step must have a 'tool' and 'arguments'."
-   **Output:** The LLM returns a machine-readable JSON plan.
    -   *Example for "Find camping sites near Ottawa, check their reservation status, and give me a recommendation":*
        ```json
        {
          "plan": [
            { "tool": "web_search", "arguments": {"query": "camping sites near Ottawa"} },
            { "tool": "url_fetch", "arguments": {"urls": ["<placeholder_for_url_1>", "<placeholder_for_url_2>"]} },
            { "tool": "summarize_text", "arguments": {"text": "<placeholder_for_fetched_content>", "question": "What is the reservation status and availability?"} }
          ]
        }
        ```

#### 2. State Management (Context Manager)

-   **Goal:** To track the progress of the multi-step plan.
-   **Mechanism:** The `ContextManager` is enhanced to store a stateful task object instead of a simple string.
-   **Implementation:**
    -   A `MultiStepTask` class is introduced in `core/context_manager.py`.
    -   This class instance stores the original `plan`, tracks the `current_step`, and accumulates the `results` of each completed step.
    -   When the plan is generated, this `MultiStepTask` object is created and saved in the user's `ConversationContext`.

#### 3. Execution Loop (Orchestrator)

-   **Goal:** To execute the plan step-by-step.
-   **Mechanism:** The main chat endpoint (in `api/routes/chat.py`) acts as an orchestrator, running an execution loop.
-   **Loop Logic:**
    1.  **Check for Active Task:** On a new user message, check if a `MultiStepTask` is in progress.
    2.  **Get Next Step:** Retrieve the next action from the plan (e.g., `{"tool": "web_search", ...}`).
    3.  **Execute Tool:** Call the corresponding service (e.g., `tool_service.py`) to execute the tool.
    4.  **Update State:** Save the tool's output (e.g., a list of URLs) into the `step_results` of the `MultiStepTask` and advance the `current_step`.
    5.  **Populate Placeholders:** Use the results from completed steps to fill in arguments for future steps.
    6.  **Repeat:** Continue the loop until all steps are complete.
    7.  **Final Summary:** Once the plan is finished, make a final call to the LLM, providing all the collected evidence and the user's original query to generate a comprehensive answer.

### How This Solves the Core Problem

-   **No Hardcoding:** The LLM generates the plan dynamically. We only code the execution loop and the atomic tools.
-   **Handles Ambiguity:** The loop can pause and ask the user for clarification if a step is ambiguous or requires user input, then resume the plan.
-   **Manages Complex Context:** The `MultiStepTask` object in the `ContextManager` provides robust, explicit state tracking for the entire operation.
-   **Preserves Privacy:** The state is task-specific and ephemeral. We can configure it to be cleared after the task is complete, respecting the user's privacy choices for long-term history.
