document.addEventListener("DOMContentLoaded", () => {
    const publicSendButton = document.getElementById("public-send-button");
    // --- Public Send Button Handler ---
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
                const errorData = await response.json();
                throw new Error(errorData.detail || "An unknown error occurred with the public LLM endpoint.");
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
            appendMessage(`Error: ${error.message}`, "assistant", true);
            stopButton.style.display = 'none';
            sendButton.style.display = 'flex';
        }
    });
    // WeChat browser detection
    if (navigator.userAgent.toLowerCase().indexOf('micromessenger') > -1) {
        document.body.classList.add('wechat-webview');
    }

    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const messagesContainer = document.getElementById("chat-messages");
    const sendButton = document.getElementById("send-button");
    const stopButton = document.getElementById("stop-button");
    const attachButton = document.getElementById("attach-button");
    const imageInput = document.getElementById("image-input");
    const imagePreview = document.getElementById("image-preview");
    const previewImg = document.getElementById("preview-img");
    const removeImageButton = document.getElementById("remove-image");
    // const generateButton = document.getElementById("generate-button"); // Removed
    const sessionList = document.getElementById("session-list");
    const clearHistoryButton = document.getElementById("clear-history-button");
    const menuIcon = document.getElementById('menu-icon');
    const sidebar = document.getElementById('sidebar');

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
        imageInput.click();
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
        console.log("Showing stop button, hiding send button");
        sendButton.style.display = 'none';
        stopButton.style.display = 'flex';

        try {
            if (selectedImage) {
                // Handle image upload (non-streaming for now)
                const formData = new FormData();
                formData.append('message', message || 'Please analyze this image');
                formData.append('image', selectedImage);
                formData.append('session_id', sessionId || '');
                formData.append('include_rag', 'true');
                formData.append('include_tools', 'true');
                formData.append('temperature', '0.7');

                const response = await fetch('/api/v1/chat/image', {
                    method: 'POST',
                    body: formData
                });

                removeTypingIndicator();
                stopButton.style.display = 'none';
                sendButton.style.display = 'flex';

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || "An unknown error occurred.");
                }

                const data = await response.json();
                sessionId = data.session_id;
                
                // Check if this is an image generation response
                if (data.metadata && data.metadata.type === "image_generation" && data.metadata.images) {
                    // Handle image generation response
                    let imagesHtml = '<div class="generated-images">';
                    data.metadata.images.forEach((imageData, index) => {
                        const imageUrl = `data:image/png;base64,${imageData.base64_data}`;
                        imagesHtml += `
                            <div class="generated-image">
                                <img src="${imageUrl}" alt="Generated image ${index + 1}" style="max-width: 300px; border-radius: 8px; margin: 5px;">
                                <p><small>Model: ${imageData.model} | Size: ${Math.round(imageData.size_bytes / 1024)}KB</small></p>
                            </div>`;
                    });
                    imagesHtml += '</div>';
                    // Append as HTML instead of text
                    const htmlContent = data.response + imagesHtml;
                    const htmlMsgDiv = createMessageDiv("assistant");
                    htmlMsgDiv.innerHTML = htmlContent;
                    messagesContainer.appendChild(htmlMsgDiv);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                } else {
                    appendMessage(data.response, "assistant");
                }

                // Clear image after sending
                selectedImage = null;
                imagePreview.style.display = 'none';
                imageInput.value = '';

            } else {
                // Handle text-only request (non-streaming)
                const payload = {
                    message: message,
                    session_id: sessionId || null,
                    user_id: "default_user" 
                };

                try {
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
                        // Prefer image_url, fallback to base64_data
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
                    // Handle image
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
                    // Handle web_search
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

                } catch (error) {
                    console.error("Chat request error:", error);
                    removeTypingIndicator();
                    appendMessage(`Error: ${error.message}`, "assistant", true);
                    stopButton.style.display = 'none';
                    sendButton.style.display = 'flex';
                }
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
                const htmlSpan = document.createElement("span");
                htmlSpan.innerHTML = text;
                messageDiv.appendChild(htmlSpan);
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

    // === Chat History UI & Logic ===
    const historySidebar = document.getElementById("history-sidebar");
    const historyTabBtn = document.getElementById("history-tab-btn");
    const historyTab = document.getElementById("history-tab");
    const historySearchInput = document.getElementById("history-search");
    const historyDeleteBtn = document.getElementById("history-delete-btn");
    const historyExportBtn = document.getElementById("history-export-btn");

    // Responsive: show sidebar on desktop, tab on mobile
    function updateHistoryLayout() {
        // Don't apply this logic in the WeChat webview, let CSS handle it
        if (document.body.classList.contains('wechat-webview')) {
            return;
        }
        if (window.innerWidth < 700) {
            historySidebar.style.display = "none";
            historyTabBtn.style.display = "block";
        } else {
            historySidebar.style.display = "block";
            historyTabBtn.style.display = "none";
            historyTab.style.display = "none";
        }
    }
    window.addEventListener("resize", updateHistoryLayout);
    document.addEventListener("DOMContentLoaded", updateHistoryLayout);

    // Show history tab on mobile
    historyTabBtn.addEventListener("click", () => {
        historyTab.style.display = "block";
    });

    // Hide history tab on mobile when back button clicked
    const historyTabBack = document.getElementById("history-tab-back");
    historyTabBack.addEventListener("click", () => {
        historyTab.style.display = "none";
    });

    // Fetch and render chat history sessions
    async function loadChatHistorySessions(keyword = "") {
        const resp = await fetch(`/api/v1/chat/history?limit=100&keyword=${encodeURIComponent(keyword)}`);
        const data = await resp.json();
        const sessions = groupMessagesBySession(data.history);
        renderHistorySessions(sessions);
    }

    // Group messages by session for sidebar/tab
    function groupMessagesBySession(messages) {
        const sessions = {};
        messages.forEach(msg => {
            if (!sessions[msg.session_id]) sessions[msg.session_id] = [];
            sessions[msg.session_id].push(msg);
        });
        return sessions;
    }

    // Render session list in sidebar/tab
    function renderHistorySessions(sessions) {
        const container = document.getElementById("history-session-list");
        container.innerHTML = "";
        Object.entries(sessions).forEach(([sessionId, msgs]) => {
            const li = document.createElement("li");
            li.textContent = `会话 ${sessionId.slice(0, 8)} (${msgs.length}条)`;
            li.onclick = () => showSessionMessages(sessionId, msgs);
            container.appendChild(li);
        });
    }

    // Show messages for a session
    function showSessionMessages(sessionId, msgs) {
        const container = document.getElementById("history-message-list");
        container.innerHTML = "";
        msgs.forEach(msg => {
            const div = document.createElement("div");
            div.className = `history-msg ${msg.message_type}`;
            div.innerHTML = `<span class='msg-role'>${msg.message_type}</span>: <span class='msg-content'>${msg.content}</span> <span class='msg-time'>${new Date(msg.timestamp*1000).toLocaleString()}</span>`;
            container.appendChild(div);
        });
        // Show delete/export buttons for this session
        historyDeleteBtn.onclick = () => deleteSessionHistory(sessionId);
        historyExportBtn.onclick = () => exportSessionHistory(sessionId);
    }

    // Search/filter handler
    historySearchInput.addEventListener("input", (e) => {
        loadChatHistorySessions(e.target.value);
    });

    // Delete all history
    document.getElementById("history-delete-all-btn").onclick = async () => {
        if (confirm("确定要删除全部历史记录吗？此操作不可恢复。")) {
            await fetch("/api/v1/chat/history", { method: "DELETE" });
            loadChatHistorySessions();
        }
    };

    // Delete session history
    async function deleteSessionHistory(sessionId) {
        if (confirm("确定要删除该会话的所有消息吗？")) {
            await fetch(`/api/v1/chat/history?session_id=${sessionId}`, { method: "DELETE" });
            loadChatHistorySessions();
        }
    }

    // Export session history
    async function exportSessionHistory(sessionId) {
        const resp = await fetch(`/api/v1/chat/history/export?session_id=${sessionId}`);
        const data = await resp.json();
        const blob = new Blob([JSON.stringify(data.export_data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `session_${sessionId}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // Privacy notice
    const privacyNotice = document.getElementById("privacy-notice");
    privacyNotice.textContent = "所有历史仅本地存储，可随时删除。";

    // === Tool Invocation UI ===
    // Fetch and display available tools
    async function loadAvailableTools() {
        const resp = await fetch('/api/v1/tools');
        const data = await resp.json();
        renderToolList(data.tools);
    }

    function renderToolList(tools) {
        let toolPanel = document.getElementById('tool-panel');
        if (!toolPanel) {
            toolPanel = document.createElement('div');
            toolPanel.id = 'tool-panel';
            toolPanel.className = 'tool-panel';
            document.querySelector('.chat-container').prepend(toolPanel);
        }
        toolPanel.innerHTML = '<h3>工具箱</h3>';
        tools.forEach(tool => {
            const btn = document.createElement('button');
            btn.className = 'tool-btn';
            btn.textContent = tool.name + ' - ' + tool.description;
            btn.onclick = () => showToolInvokeForm(tool);
            toolPanel.appendChild(btn);
        });
    }

    function showToolInvokeForm(tool) {
        let formDiv = document.getElementById('tool-invoke-form');
        if (!formDiv) {
            formDiv = document.createElement('div');
            formDiv.id = 'tool-invoke-form';
            formDiv.className = 'tool-invoke-form';
            document.body.appendChild(formDiv);
        }
        formDiv.innerHTML = `<h4>Invoke Tool: ${tool.name}</h4>`;
        tool.parameters.forEach(param => {
            formDiv.innerHTML += `<label>${param.name}: <input type='text' id='tool-param-${param.name}' placeholder='${param.description}'></label><br>`;
        });
        formDiv.innerHTML += `<button id='tool-invoke-btn'>Invoke</button> <button id='tool-cancel-btn'>Cancel</button>`;
        formDiv.style.display = 'block';
        document.getElementById('tool-invoke-btn').onclick = async () => {
            const params = {};
            tool.parameters.forEach(param => {
                params[param.name] = document.getElementById('tool-param-' + param.name).value;
            });
            await invokeTool(tool.name, params);
            formDiv.style.display = 'none';
        };
        document.getElementById('tool-cancel-btn').onclick = () => {
            formDiv.style.display = 'none';
        };
    }

    async function invokeTool(toolName, params) {
        appendMessage(`Invoking tool: ${toolName}...`, 'user');
        showTypingIndicator();
        const resp = await fetch('/api/v1/tools/invoke', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool_name: toolName, params })
        });
        removeTypingIndicator();
        const data = await resp.json();
        if (data.success) {
            appendMessage(JSON.stringify(data.result, null, 2), 'assistant');
        } else {
            appendMessage(`Tool error: ${data.error}`, 'assistant', true);
        }
    }

    // Add a button to open the tool panel
    let toolPanelBtn = document.getElementById('tool-panel-btn');
    if (!toolPanelBtn) {
        toolPanelBtn = document.createElement('button');
        toolPanelBtn.id = 'tool-panel-btn';
        toolPanelBtn.textContent = '🧰 工具箱';
        toolPanelBtn.className = 'tool-panel-btn';
        document.querySelector('.chat-header').appendChild(toolPanelBtn);
    }
    toolPanelBtn.onclick = () => {
        let toolPanel = document.getElementById('tool-panel');
        if (toolPanel) {
            toolPanel.style.display = (toolPanel.style.display === 'none' ? 'block' : 'none');
        }
    };
    // Load tools on startup
    loadAvailableTools();

    // === List All Chat Sessions (ChromaDB) ===
    async function loadAllChatSessions() {
        const resp = await fetch('/api/v1/chat/history?limit=1000');
        const data = await resp.json();
        const sessions = groupMessagesBySession(data.history);
        renderHistorySessions(sessions);
    }
    // Optionally, call loadAllChatSessions() on startup or via a button
    // loadAllChatSessions();

    // === Enhanced History/Search UI ===
    // Compact, side-by-side layout for session/message list
    // Add a toggle to switch between all sessions and filtered/search view
    let historyMode = 'all'; // 'all' or 'search'
    const historyModeToggle = document.createElement('button');
    historyModeToggle.id = 'history-mode-toggle';
    historyModeToggle.textContent = '全部会话';
    historyModeToggle.className = 'history-mode-toggle';
    document.querySelector('.history-header').appendChild(historyModeToggle);

    historyModeToggle.onclick = () => {
        historyMode = (historyMode === 'all') ? 'search' : 'all';
        historyModeToggle.textContent = (historyMode === 'all') ? '全部会话' : '搜索结果';
        if (historyMode === 'all') {
            loadAllChatSessions();
        } else {
            loadChatHistorySessions(historySearchInput.value);
        }
    };

    // Improve search UX: auto-switch to search mode on input
    historySearchInput.addEventListener('input', (e) => {
        if (e.target.value.trim()) {
            historyMode = 'search';
            historyModeToggle.textContent = '搜索结果';
            loadChatHistorySessions(e.target.value);
        } else {
            historyMode = 'all';
            historyModeToggle.textContent = '全部会话';
            loadAllChatSessions();
        }
    });

    // Compact layout: show session list and message list side by side (desktop)
    // Add CSS class for compact layout (handled in CSS)
    document.getElementById('history-sidebar').classList.add('compact-history');

    // --- Remove old global chat history logic ---
    // On page load, load the most recent session's history (if any)
    function loadLastSession() {
        const sessions = JSON.parse(localStorage.getItem("chat_sessions")) || [];
        if (sessions.length > 0) {
            const lastSession = sessions[sessions.length - 1];
            loadChatHistory(lastSession.id);
        } else {
            messagesContainer.innerHTML = "";
            appendMessage("Hello! I am Aura-PAI. How can I assist you today? You can also attach images for visual analysis.", "assistant");
        }
    }
    // Call this instead of loadChatHistoryFromLocal on page load
    loadLastSession();

    // Remove old saveChatHistory and loadChatHistoryFromLocal logic
    // (No longer needed, as we use per-session storage)

    // More robust sidebar toggle logic using event delegation
    document.addEventListener('click', function(event) {
        const sidebar = document.getElementById('sidebar');
        const menuIcon = document.getElementById('menu-icon');

        // Check if the click is on the menu icon or one of its children (the spans)
        if (menuIcon.contains(event.target)) {
            sidebar.classList.toggle('active');
            return; // Stop further processing
        }

        // Check if the click is outside the sidebar and the sidebar is active
        if (!sidebar.contains(event.target) && sidebar.classList.contains('active')) {
            sidebar.classList.remove('active');
        }
    });
});
