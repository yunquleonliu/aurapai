document.addEventListener("DOMContentLoaded", () => {
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

    let sessionId = null; // To store the session ID for the conversation
    let currentEventSource = null; // To track current streaming connection
    let selectedImage = null; // To store selected image

    // Display a welcome message
    appendMessage("Hello! I am Auro-PAI. How can I assist you today? You can also attach images for visual analysis.", "assistant");

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
                appendMessage(data.response, "assistant");

                // Clear image after sending
                selectedImage = null;
                imagePreview.style.display = 'none';
                imageInput.value = '';

            } else {
                // Handle text-only streaming
                const params = new URLSearchParams({
                    message: message,
                    session_id: sessionId || '',
                    include_rag: true,
                    include_tools: true,
                    temperature: 0.7,
                });

                currentEventSource = new EventSource(`/api/v1/chat/stream?${params.toString()}`);
                let assistantMessageDiv = null;
                let fullResponse = "";

                // Listener for the session_start event
                currentEventSource.addEventListener('session_start', function(event) {
                    const data = JSON.parse(event.data);
                    sessionId = data.session_id;
                    console.log("Session started:", sessionId);
                });

                // Listener for incoming message chunks
                currentEventSource.addEventListener('message', function(event) {
                    removeTypingIndicator();
                    const data = JSON.parse(event.data);

                    if (!assistantMessageDiv) {
                        assistantMessageDiv = createMessageDiv("assistant");
                        messagesContainer.appendChild(assistantMessageDiv);
                    }

                    fullResponse += data.content;
                    assistantMessageDiv.textContent = fullResponse;
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                });

                // Listener for the end of the stream
                currentEventSource.addEventListener('end', function(event) {
                    console.log("Stream ended.");
                    currentEventSource.close();
                    currentEventSource = null;
                    stopButton.style.display = 'none';
                    sendButton.style.display = 'flex';
                });

                // Listener for any errors from the stream
                currentEventSource.addEventListener('error', function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        console.error("Stream error:", data.error);
                        removeTypingIndicator();
                        appendMessage(`Server Error: ${data.error}`, "assistant", true);
                    } catch (e) {
                        console.error("Stream parse error:", e);
                    }
                    currentEventSource.close();
                    currentEventSource = null;
                    stopButton.style.display = 'none';
                    sendButton.style.display = 'flex';
                });

                // General error handler for the EventSource connection itself
                currentEventSource.onerror = function(err) {
                    console.error("EventSource failed:", err);
                    removeTypingIndicator();
                    if (!fullResponse) {
                        appendMessage("Error: Could not connect to the server for streaming.", "assistant", true);
                    }
                    if (currentEventSource) {
                        currentEventSource.close();
                        currentEventSource = null;
                    }
                    stopButton.style.display = 'none';
                    sendButton.style.display = 'flex';
                };
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

    function appendMessage(text, sender, isError = false, imageFile = null) {
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

        // Add text content
        if (text) {
            const textSpan = document.createElement("span");
            textSpan.textContent = text;
            messageDiv.appendChild(textSpan);
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
});
