# Private Knowledge Management (`Private NotebookLM`)

This document outlines the functionality of the Private Knowledge Management feature, a core component of Aura-PAI that allows users to create and manage a private knowledge base for Retrieval-Augmented Generation (RAG).

## Overview

The Private Knowledge Management feature, also known as "Private NotebookLM," enables users to upload their own documents (PDFs, text files, markdown, and various code files) to create a secure, private knowledge base. This knowledge base is then used by the chat agent to provide more accurate, context-aware answers based on the user's own data.

## Key Features

*   **Secure Document Upload:** Users can upload documents through a simple interface in the application's sidebar. The supported file types are `.pdf`, `.md`, `.txt`, `.py`, `.js`, `.html`, and `.css`.
*   **Document Management:** Uploaded documents are listed in the "Knowledge" tab, where users can easily see and delete them.
*   **Vector Embedding and Storage:** When a document is uploaded, it is chunked, converted into vector embeddings, and stored in a local ChromaDB vector database. This process is handled by the `KnowledgeService`.
*   **Retrieval-Augmented Generation (RAG):** When a user sends a message, the system retrieves the most relevant information from the knowledge base and injects it as context into the LLM's prompt. This allows the agent to answer questions based on the content of the uploaded documents.
*   **Seamless Integration:** The entire process is seamlessly integrated into the chat interface. The user interacts with the agent as usual, and the agent automatically leverages the knowledge base when needed.

## How It Works

1.  **Frontend (UI):**
    *   A new "Knowledge" tab is available in the sidebar.
    *   Users can click the "Upload Document" button to select a file from their local machine.
    *   The list of uploaded documents is displayed, with an option to delete each one.

2.  **Backend (`KnowledgeService`):**
    *   Handles document uploads, processing, and storage.
    *   Uses `langchain` for document loading and text splitting.
    *   Uses `sentence-transformers` to generate embeddings.
    *   Uses `ChromaDB` for storing and retrieving vector embeddings.

3.  **Chat Integration (`chat.py`):**
    *   Before processing a user's chat message, the system queries the `KnowledgeService` to find relevant documents.
    *   The retrieved text is added to the LLM's prompt as "Knowledge Base Context."
    *   The LLM uses this context to generate a more informed and accurate response.

## Benefits

*   **Context-Aware Responses:** The agent can answer questions about specific information contained in the user's private documents.
*   **Reduced Hallucinations:** By grounding the agent's responses in a specific knowledge base, the likelihood of the LLM generating incorrect or fabricated information is reduced.
*   **Privacy:** All documents and data are stored locally, ensuring that sensitive information remains private.
