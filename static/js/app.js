document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const messagesContainer = document.getElementById("chat-messages");
    const sendButton = document.getElementById("send-button");

    let sessionId = null; // To store the session ID for the conversation

    // Display a welcome message
    appendMessage("Hello! I am Auro-PAI. How can I assist you today?", "assistant");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message) return;

        appendMessage(message, "user");
        input.value = "";
        showTypingIndicator();

        try {
            // Use EventSource for streaming from the GET endpoint
            const params = new URLSearchParams({
                message: message,
                session_id: sessionId || '',
                include_rag: true,
                include_tools: true,
                temperature: 0.7,
            });

            const eventSource = new EventSource(`/api/v1/chat/stream?${params.toString()}`);
            let assistantMessageDiv = null;
            let fullResponse = "";

            // Listener for the session_start event
            eventSource.addEventListener('session_start', function(event) {
                const data = JSON.parse(event.data);
                sessionId = data.session_id;
                console.log("Session started:", sessionId);
            });

            // Listener for incoming message chunks
            eventSource.addEventListener('message', function(event) {
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
            eventSource.addEventListener('end', function(event) {
                console.log("Stream ended.");
                eventSource.close();
            });

            // Listener for any errors from the stream
            eventSource.addEventListener('error', function(event) {
                const data = JSON.parse(event.data);
                console.error("Stream error:", data.error);
                removeTypingIndicator();
                appendMessage(`Server Error: ${data.error}`, "assistant", true);
                eventSource.close();
            });

            // General error handler for the EventSource connection itself
            eventSource.onerror = function(err) {
                console.error("EventSource failed:", err);
                removeTypingIndicator();
                // Avoid showing a generic error if a specific one was already sent
                if (!fullResponse) {
                    appendMessage("Error: Could not connect to the server for streaming.", "assistant", true);
                }
                eventSource.close();
            };

        } catch (error) {
            console.error("Error:", error);
            removeTypingIndicator();
            appendMessage(`Error: ${error.message}`, "assistant", true);
        }
    });

    function createMessageDiv(sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", `${sender}-message`);
        return messageDiv;
    }

    function appendMessage(text, sender, isError = false) {
        const messageDiv = createMessageDiv(sender);
        if (isError) {
            messageDiv.style.color = "red";
        }
        messageDiv.textContent = text;
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
