
// Simple markdown rendering using marked.js (or fallback to basic)
import { renderMarkdown } from './markdown.js';

document.addEventListener("DOMContentLoaded", () => {
    const publicSendButton = document.getElementById("public-send-button");
    // --- Public Send Button Handler ---
    if (publicSendButton) {
        publicSendButton.addEventListener("click", async () => {
            const message = input.value.trim();
            if (!message) return;
            appendMessage(message, "user");
            input.value = "";
            showTypingIndicator();
            sendButton.style.display = 'none';
            stopButton.style.display = 'flex';
            try {
                const payload = {
                    message: message,
                    session_id: sessionId || null,
                    user_id: "default_user",
                    use_public_llm: true
                };
                const response = await fetch('/api/v1/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                removeTypingIndicator();
                stopButton.style.display = 'none';
                sendButton.style.display = 'flex';
                if (!response.ok) {
                    // Show the actual error message from the network response
                    let errorMsg = `HTTP ${response.status}`;
                    const contentType = response.headers.get('content-type') || '';
                    if (contentType.includes('application/json')) {
                        try {
                            const errorData = await response.json();
                            errorMsg = errorData.error || errorData.detail || JSON.stringify(errorData) || errorMsg;
                        } catch (e) {}
                    } else {
                        try {
                            const text = await response.text();
                            errorMsg = text;
                        } catch (e) {}
                    }
                    // Show the actual error message to the user (and throw for catch block)
                    throw new Error(errorMsg);
                }
                const data = await response.json();
                if (data.session_id) {
                    sessionId = data.session_id;
                    if (!sessionList.querySelector(`[data-session-id="${sessionId}"]`)) {
                        saveSession(sessionId, message);
                    }
                }
                appendMessage(data.response, "assistant");
            } catch (error) {
                console.error("Public LLM request error:", error);
                removeTypingIndicator();
                // Show the real error message (not prefixed with 'Error:')
                appendMessage(error.message, "assistant", true);
                stopButton.style.display = 'none';
                sendButton.style.display = 'flex';
            }
        });
    }
    // WeChat browser detection
    if (navigator.userAgent.toLowerCase().indexOf('micromessenger') > -1) {
        document.body.classList.add('wechat-webview');
    }

    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const messagesContainer = document.getElementById("chat-messages");
    const sendButton = document.getElementById("send-button");
    const stopButton = document.getElementById("stop-button");
    const attachButton = document.getElementById('attach-button');
    const attachmentMenu = document.getElementById('attachment-menu');
    const attachImageButton = document.getElementById('attach-image-button');
    const attachDocumentButton = document.getElementById('attach-document-button');
    const imageInput = document.getElementById("image-input");
    const documentInput = document.getElementById('document-input');
    const imagePreview = document.getElementById("image-preview");
    const previewImg = document.getElementById("preview-img");
    const removeImageButton = document.getElementById("remove-image");
    const sessionList = document.getElementById("session-list");
    const clearHistoryButton = document.getElementById("clear-history-button");
    const menuIcon = document.getElementById('menu-icon');
    const sidebar = document.getElementById('sidebar');

    // --- Knowledge Base UI Elements ---
    const knowledgeTabButton = document.querySelector('.sidebar-tab-button[data-tab="knowledge"]');
    const chatsTabButton = document.querySelector('.sidebar-tab-button[data-tab="chats"]');
    const knowledgeTab = document.getElementById('knowledge-tab');
    const chatsTab = document.getElementById('chats-tab');
    const documentList = document.getElementById('document-list');
    const uploadDocumentSidebarBtn = document.getElementById('upload-document-sidebar-btn');

    let sessionId = null; // To store the session ID for the conversation
    let currentEventSource = null; // To track current streaming connection
    let selectedImage = null; // To store selected image

    // --- Sidebar Toggle Logic ---
    menuIcon.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent click from bubbling to the document
        sidebar.classList.toggle('active');
    });

    document.addEventListener('click', (e) => {
        // If the sidebar is active and the click is outside the sidebar and not on the menu icon
        if (sidebar.classList.contains('active') && !sidebar.contains(e.target) && e.target !== menuIcon) {
            sidebar.classList.remove('active');
        }
    });

    // --- Sidebar Tab Switching ---
    knowledgeTabButton.addEventListener('click', () => switchTab('knowledge'));
    chatsTabButton.addEventListener('click', () => switchTab('chats'));

    function switchTab(tabName) {
        if (tabName === 'knowledge') {
            chatsTab.style.display = 'none';
            knowledgeTab.style.display = 'block';
            chatsTabButton.classList.remove('active');
            knowledgeTabButton.classList.add('active');
            loadDocuments();
        } else {
            knowledgeTab.style.display = 'none';
            chatsTab.style.display = 'block';
            knowledgeTabButton.classList.remove('active');
            chatsTabButton.classList.add('active');
        }
    }

    // Display a welcome message
    appendMessage("Hello! I am Aura-PAI. How can I assist you today? You can also attach images for visual analysis.", "assistant");

    loadSessions();

    clearHistoryButton.addEventListener("click", () => {
        // Remove the current session and its history from localStorage and UI
        if (!sessionId) return;
        let sessions = JSON.parse(localStorage.getItem("chat_sessions")) || [];
        sessions = sessions.filter(s => s.id !== sessionId);
        localStorage.setItem("chat_sessions", JSON.stringify(sessions));
        // Remove from UI
        sessionList.innerHTML = "";
        loadSessions();
        messagesContainer.innerHTML = "";
        sessionId = null;
        appendMessage("Hello! I am Aura-PAI. How can I assist you today? You can also attach images for visual analysis.", "assistant");
    });

    function loadSessions() {
        const sessions = JSON.parse(localStorage.getItem("chat_sessions")) || [];
        sessionList.innerHTML = "";
        sessions.forEach(session => {
            const li = document.createElement("li");
            li.textContent = session.name;
            li.dataset.sessionId = session.id;
            li.addEventListener("click", () => {
                loadChatHistory(session.id);
            });
            sessionList.appendChild(li);
        });
    }

    // --- Patch: Use localStorage for chat history loading ---
    function loadChatHistory(sid) {
    sessionId = sid;
    messagesContainer.innerHTML = "";
    // Fetch chat history from the correct backend API endpoint
    fetch(`/api/v1/chat/sessions/${encodeURIComponent(sid)}/history`)
        .then(resp => resp.json())
        .then(data => {
            if (!data.history || data.history.length === 0) {
                appendMessage("No history found for this session.", "assistant", true);
                return;
            }
            data.history.forEach(msg => {
                // Map backend message fields (role, content) to appendMessage signature
                appendMessage(msg.content, msg.role, false, null, false);
            });
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        })
        .catch(err => {
            console.error('Error loading history:', err);
            appendMessage("Error loading history from server.", "assistant", true);
        });
    }

    // --- Enhanced: Save and load per-session chat history in localStorage ---
    function saveSession(sid, message) {
        let sessions = JSON.parse(localStorage.getItem("chat_sessions")) || [];
        let session = sessions.find(s => s.id === sid);
        if (!session) {
            session = { id: sid, name: message.substring(0, 30), history: [] };
            sessions.push(session);
        }
        localStorage.setItem("chat_sessions", JSON.stringify(sessions));
        loadSessions();
    }

    // Save a message to the current session's history
    function saveMessageToSession(msgObj) {
        if (!sessionId) return;
        let sessions = JSON.parse(localStorage.getItem("chat_sessions")) || [];
        let session = sessions.find(s => s.id === sessionId);
        if (!session) {
            session = { id: sessionId, name: "Session " + sessionId.substring(0, 8), history: [] };
            sessions.push(session);
        }
        session.history = session.history || [];
        session.history.push(msgObj);
        localStorage.setItem("chat_sessions", JSON.stringify(sessions));
    }

    // Attach button click handler
    attachButton.addEventListener("click", () => {
        // Toggle the visibility of the attachment menu
        if (attachmentMenu.style.display === 'block') {
            attachmentMenu.style.display = 'none';
        } else {
            attachmentMenu.style.display = 'block';
        }
    });

    // Close the attachment menu if clicking outside of it
    document.addEventListener('click', (event) => {
        const isClickInside = attachmentMenu.contains(event.target) || attachButton.contains(event.target);
        if (!isClickInside) {
            attachmentMenu.style.display = 'none';
        }
    });

    // --- Attachment Menu Logic ---
    attachImageButton.addEventListener('click', () => {
        imageInput.click();
        attachmentMenu.style.display = 'none';
    });

    attachDocumentButton.addEventListener('click', () => {
        documentInput.click();
        attachmentMenu.style.display = 'none';
    });

    documentInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleDocumentUpload(file);
        }
    });

    // Sidebar upload button - trigger the main document input
    uploadDocumentSidebarBtn.addEventListener('click', () => {
        documentInput.click();
    });

    // Image input change handler
    imageInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            selectedImage = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'flex';
            };
            reader.readAsDataURL(file);
        }
    });

    // Remove image button handler
    removeImageButton.addEventListener("click", () => {
        selectedImage = null;
        imagePreview.style.display = 'none';
        imageInput.value = '';
    });

    // --- Knowledge Base Functions ---
    async function handleDocumentUpload(file) {
        const formData = new FormData();
        formData.append("file", file);

        appendMessage(`Uploading document: ${file.name}...`, "system");

        try {
            const response = await fetch('/api/v1/knowledge/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Failed to upload document.");
            }

            const result = await response.json();
            appendMessage(`Successfully uploaded and processed ${result.filename}.`, "system");
            loadDocuments(); // Refresh the document list

        } catch (error) {
            console.error("Document upload error:", error);
            appendMessage(`Error uploading document: ${error.message}`, "system", true);
        }
    }

    async function loadDocuments() {
        try {
            const response = await fetch('/api/v1/knowledge/list');
            if (!response.ok) {
                throw new Error("Failed to fetch document list.");
            }
            const data = await response.json();
            const documents = data.documents || [];
            documentList.innerHTML = ''; // Clear existing list
            if (documents.length === 0) {
                documentList.innerHTML = '<li>No documents found.</li>';
                return;
            }
            documents.forEach(doc => {
                const li = document.createElement('li');
                li.textContent = doc.filename || doc; // Handle both object and string formats
                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = 'Delete';
                deleteBtn.className = 'delete-doc-btn';
                deleteBtn.onclick = () => deleteDocument(doc.filename || doc);
                li.appendChild(deleteBtn);
                documentList.appendChild(li);
            });
        } catch (error) {
            console.error("Error loading documents:", error);
            documentList.innerHTML = '<li>Error loading documents.</li>';
        }
    }

    async function deleteDocument(filename) {
        if (!confirm(`Are you sure you want to delete ${filename}?`)) {
            return;
        }
        try {
            const response = await fetch(`/api/v1/knowledge/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Failed to delete document.");
            }
            appendMessage(`Document "${filename}" deleted.`, 'system');
            loadDocuments(); // Refresh list
        } catch (error) {
            console.error("Error deleting document:", error);
            appendMessage(`Error deleting document: ${error.message}`, 'system', true);
        }
    }


    // Stop button handler
    stopButton.addEventListener("click", () => {
        console.log("Stop button clicked!");
        if (currentEventSource) {
            console.log("Closing current event source");
            currentEventSource.close();
            currentEventSource = null;
            stopButton.style.display = 'none';
            sendButton.style.display = 'flex';
            removeTypingIndicator();
        }
    });

    // Allow sending with Enter key, but allow Shift+Enter for new lines
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            form.requestSubmit(); // Trigger form submission
        }
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message && !selectedImage) return;

        // Show user message with image if present
        if (selectedImage) {
            appendMessage(message || "Image attached", "user", false, selectedImage);
        } else {
            appendMessage(message, "user");
        }

        input.value = "";
        showTypingIndicator();

        // Show stop button, hide send button
        sendButton.style.display = 'none';
        stopButton.style.display = 'flex';

        try {
            if (selectedImage) {
                // ...existing code for image upload...
                const reader = new FileReader();
                reader.onload = async (e) => {
                    // ...existing code for image upload...
                };
                reader.readAsDataURL(selectedImage);
            } else {
                // Streaming support for /api/v1/chat
                const payload = {
                    message: message,
                    session_id: sessionId || null,
                    user_id: "default_user"
                };
                const response = await fetch('/api/v1/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'text/plain,application/json'
                    },
                    body: JSON.stringify(payload)
                });

                // If streaming (text/plain), use stream reader
                const contentType = response.headers.get('content-type') || '';
                console.log('[Streaming] Response content-type:', contentType);
                if (contentType.startsWith('text/plain')) {
                    // Streaming response (SSE/JSON lines)
                    removeTypingIndicator();
                    stopButton.style.display = 'flex'; // Show stop button during streaming
                    sendButton.style.display = 'none';
                    let messageDiv = createMessageDiv('assistant');
                    let textSpan = document.createElement('span');
                    messageDiv.appendChild(textSpan);
                    // Add streaming indicator
                    let streamingIndicator = document.createElement('span');
                    streamingIndicator.className = 'streaming-indicator';
                    streamingIndicator.textContent = ' [streaming...]';
                    textSpan.appendChild(streamingIndicator);
                    messagesContainer.appendChild(messageDiv);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    const reader = response.body.getReader();
                    let decoder = new TextDecoder();
                    let done = false;
                    let fullText = '';
                    let chunkCount = 0;
                    let buffer = '';
                    try {
                        while (!done) {
                            const { value, done: doneReading } = await reader.read();
                            done = doneReading;
                            if (value) {
                                const chunk = decoder.decode(value, { stream: !done });
                                buffer += chunk;
                                // Split buffer into lines
                                let lines = buffer.split(/\r?\n/);
                                // Keep last line in buffer if not done
                                if (!done && buffer[buffer.length-1] !== '\n') {
                                    buffer = lines.pop();
                                } else {
                                    buffer = '';
                                }
                                for (let line of lines) {
                                    if (line.startsWith('data:')) {
                                        try {
                                            const jsonStr = line.slice(5).trim();
                                            if (jsonStr) {
                                                const obj = JSON.parse(jsonStr);
                                                let token = '';
                                                if (obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content) {
                                                    token = obj.choices[0].delta.content;
                                                } else if (obj.content) {
                                                    token = obj.content;
                                                } else if (typeof obj === 'string') {
                                                    token = obj;
                                                }
                                                if (token) {
                                                    fullText += token;
                                                    textSpan.innerHTML = renderMarkdown(fullText);
                                                    textSpan.appendChild(streamingIndicator);
                                                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                                    chunkCount++;
                                                }
                                            }
                                        } catch (e) {
                                            // Not valid JSON, show raw line and preserve newlines
                                            fullText += line.replace(/^data:/, '') + '\n';
                                            textSpan.innerHTML = renderMarkdown(fullText);
                                            textSpan.appendChild(streamingIndicator);
                                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                            chunkCount++;
                                        }
                                    } else if (line.trim()) {
                                        // Show any non-empty non-data line as raw text and preserve newlines
                                        fullText += line + '\n';
                                        textSpan.innerHTML = renderMarkdown(fullText);
                                        textSpan.appendChild(streamingIndicator);
                                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                        chunkCount++;
                                    }
                                }
                            }
                        }
                        if (streamingIndicator.parentNode) streamingIndicator.parentNode.removeChild(streamingIndicator);
                    } catch (err) {
                        console.error('[Streaming] Error while reading stream:', err);
                        if (streamingIndicator.parentNode) streamingIndicator.parentNode.removeChild(streamingIndicator);
                        appendMessage('Error: Streaming failed. ' + err.message, 'assistant', true);
                    }
                    if (chunkCount === 0) {
                        appendMessage('No data received from stream or received non-JSON data.', 'assistant', true);
                    }
                    return;
                } else {
                    console.warn('[Streaming] Not a streaming response. Content-Type:', contentType);
                }

                // Otherwise, fallback to old logic (JSON response)
                removeTypingIndicator();
                stopButton.style.display = 'none';
                sendButton.style.display = 'flex';

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || "An unknown error occurred with the chat endpoint.");
                }
                const data = await response.json();
                if (data.session_id) {
                    sessionId = data.session_id;
                    if (!sessionList.querySelector(`[data-session-id="${sessionId}"]`)) {
                        saveSession(sessionId, message);
                    }
                }
                // PATCH: Render image or video if present in tool_usage
                if (
                    data.tool_usage &&
                    data.tool_usage.length > 0 &&
                    data.tool_usage[0].result
                ) {
                    const result = data.tool_usage[0].result;
                    if (result.type === "image") {
                        let imgHtml = "";
                        if (result.image_url) {
                            imgHtml = `<img src="${result.image_url}" alt="${result.message || 'Generated image'}" class="message-image">`;
                        } else if (result.base64_data) {
                            imgHtml = `<img src="data:image/png;base64,${result.base64_data}" alt="${result.message || 'Generated image'}" class="message-image">`;
                        } else {
                            imgHtml = `<span>[Image data unavailable]</span>`;
                        }
                        let htmlContent = imgHtml;
                        if (result.message) {
                            htmlContent += `<div class="image-caption">${result.message}</div>`;
                        }
                        appendMessage(htmlContent, "assistant", false, null, true);
                        return;
                    }
                    if (result.type === "video" && result.video_url) {
                        appendMessage(
                            `<video controls src="${result.video_url}" style="max-width: 400px; border-radius: 8px; margin: 5px;"></video>`,
                            "assistant",
                            false,
                            null,
                            true
                        );
                        if (result.message) {
                            appendMessage(result.message, "assistant");
                        }
                        return;
                    }
                    if (result.type === "clarification" && result.message) {
                        appendMessage(result.message, "assistant");
                        return;
                    }
                }
                // PATCH: Render image if present in data.response (for ReAct agent direct tool return)
                if (
                    data.response &&
                    typeof data.response === "object"
                ) {
                    if (data.response.type === "image") {
                        let imgHtml = "";
                        if (data.response.image_url) {
                            imgHtml = `<img src="${data.response.image_url}" alt="${data.response.message || 'Generated image'}" class="message-image">`;
                        } else if (data.response.base64_data) {
                            imgHtml = `<img src="data:image/png;base64,${data.response.base64_data}" alt="${data.response.message || 'Generated image'}" class="message-image">`;
                        } else {
                            imgHtml = `<span>[Image data unavailable]</span>`;
                        }
                        let htmlContent = imgHtml;
                        if (data.response.message) {
                            htmlContent += `<div class="image-caption">${data.response.message}</div>`;
                        }
                        appendMessage(htmlContent, "assistant", false, null, true);
                        return;
                    }
                    if (data.response.type === "web_search" && Array.isArray(data.response.results || data.response.search_results)) {
                        const results = data.response.results || data.response.search_results;
                        let htmlContent = `<div class="web-search-results"><b>Web search results:</b><ul style='padding-left: 18px;'>`;
                        results.slice(0, 5).forEach(r => {
                            htmlContent += `<li style='margin-bottom: 8px;'><a href="${r.url}" target="_blank" rel="noopener">${r.title || r.url}</a><br><span style='font-size: 0.95em; color: #666;'>${r.snippet || ''}</span></li>`;
                        });
                        htmlContent += `</ul></div>`;
                        appendMessage(htmlContent, "assistant", false, null, true);
                        return;
                    }
                }
                // Default: render text response
                appendMessage(data.response, "assistant");
            }
        } catch (error) {
            console.error("Error:", error);
            removeTypingIndicator();
            appendMessage(`Error: ${error.message}`, "assistant", true);
            stopButton.style.display = 'none';
            sendButton.style.display = 'flex';
        }
    });

    function createMessageDiv(sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", `${sender}-message`);
        return messageDiv;
    }

function appendMessage(text, sender, isError = false, imageFile = null, isHtml = false) {
    const messageDiv = createMessageDiv(sender);
    if (isError) {
        messageDiv.style.color = "red";
    }

    // Add a class for system messages to style them differently
    if (sender === 'system') {
        messageDiv.classList.add('system-message');
    }

    // If there's an image, add it first
    if (imageFile) {
        messageDiv.classList.add("has-image");
        const img = document.createElement("img");
        img.classList.add("message-image");
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(imageFile);
        messageDiv.appendChild(img);
    }

    // Add text or HTML content
    if (text) {
        if (isHtml) {
            // If isHtml is true, still try to render markdown for assistant, else show as HTML
            if (sender === "assistant") {
                const mdSpan = document.createElement("span");
                mdSpan.innerHTML = renderMarkdown(text);
                messageDiv.appendChild(mdSpan);
            } else {
                const htmlSpan = document.createElement("span");
                htmlSpan.innerHTML = text;
                messageDiv.appendChild(htmlSpan);
            }
        } else if (sender === "assistant") {
            // Always render markdown for assistant messages
            const mdSpan = document.createElement("span");
            mdSpan.innerHTML = renderMarkdown(text);
            messageDiv.appendChild(mdSpan);
        } else {
            const textSpan = document.createElement("span");
            textSpan.textContent = text;
            messageDiv.appendChild(textSpan);
        }
    }

    // Option 1: right-align user, left-align assistant (already handled by CSS)
    // Option 2: shorten bubble to content (handled by CSS width: fit-content)
    // Optionally, add a class for very short messages for better appearance
    if (sender === "user" && text && text.length < 8) {
        messageDiv.classList.add("short-user-message");
    }
    if (sender === "assistant" && text && text.length < 8) {
        messageDiv.classList.add("short-assistant-message");
    }

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight; // Auto-scroll
}

    function showTypingIndicator() {
        const typingDiv = document.createElement("div");
        typingDiv.id = "typing-indicator";
        typingDiv.classList.add("message", "assistant-message");
        typingDiv.innerHTML = `<span></span><span></span><span></span>`;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Add CSS for the typing indicator dots
        const style = document.createElement('style');
        style.innerHTML = `
            #typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #8e8e93;
                margin: 0 2px;
                animation: bounce 1.4s infinite both;
            }
            #typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
            #typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1.0); }
            }
        `;
        document.head.appendChild(style);
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById("typing-indicator");
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }


    // (Image generation button and function removed)

    // ...existing code...
});
