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
    const generateButton = document.getElementById("generate-button");
    const generateDishesButton = document.getElementById("generate-dishes-button");

    let sessionId = null; // To store the session ID for the conversation
    let currentEventSource = null; // To track current streaming connection
    let selectedImage = null; // To store selected image

    // Display a welcome message
    appendMessage("Hello! I am Aura-PAI. How can I assist you today? You can also attach images for visual analysis.", "assistant");

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

    // Add generate images button handler
    const generateImagesButton = document.getElementById("generate-images-button");
    if (generateImagesButton) {
        generateImagesButton.addEventListener("click", () => {
            if (selectedImage) {
                generateImagesFromMenu();
            } else {
                alert("Please attach a menu image first!");
            }
        });
    }

    // Generate images from menu function
    async function generateImagesFromMenu() {
        if (!selectedImage) return;

        const formData = new FormData();
        formData.append("message", "Generate images for all dishes in this menu");
        formData.append("image", selectedImage);
        formData.append("style", "photorealistic");
        formData.append("max_dishes", "5");

        try {
            // Show loading message
            appendMessage("🎨 Generating images for menu dishes...", "assistant");
            
            const response = await fetch("/api/v1/chat/generate-images", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                // Display generated images
                displayGeneratedImages(result);
            } else {
                appendMessage(`❌ Failed to generate images: ${result.error || 'Unknown error'}`, "assistant");
            }
        } catch (error) {
            console.error("Generation error:", error);
            appendMessage(`❌ Error generating images: ${error.message}`, "assistant");
        }
    }

    // Display generated images function
    function displayGeneratedImages(result) {
        const container = document.createElement("div");
        container.className = "generated-images-container";
        
        let html = `
            <div class="message assistant">
                <h3>🎨 Generated Dish Images</h3>
                <p>Generated ${result.generated_images.length} images in ${result.style} style:</p>
                <div class="images-grid">
        `;

        result.generated_images.forEach((image, index) => {
            if (image.success) {
                // Always prefer image_url if present, otherwise use base64
                let imgSrc = image.image_url ? image.image_url : (image.image_base64 ? `data:image/png;base64,${image.image_base64}` : '');
                html += `
                    <div class="dish-image-card">
                        <img src="${imgSrc}" 
                             alt="${image.dish_name || 'Generated image'}" 
                             class="generated-dish-image">
                        <div class="dish-info">
                            <h4>${image.dish_name || `Image ${index + 1}`}</h4>
                            <p>Provider: ${image.provider || ''}</p>
                        </div>
                    </div>
                `;
            } else {
                html += `
                    <div class="dish-image-card error">
                        <div class="error-placeholder">❌</div>
                        <div class="dish-info">
                            <h4>${image.dish_name || `Image ${index + 1}`}</h4>
                            <p class="error">Failed: ${image.error}</p>
                        </div>
                    </div>
                `;
            }
        });

        html += `
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        messagesContainer.appendChild(container);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

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

    // Generate image button handler
    generateButton.addEventListener("click", () => {
        const message = input.value.trim();
        if (!message) {
            alert("Please enter a description for the image you want to generate");
            return;
        }
        generateImage(message);
        input.value = "";
    });

    // Generate dishes button handler (for menu images)
    generateDishesButton.addEventListener("click", () => {
        if (!selectedImage) {
            alert("Please attach a menu image first");
            return;
        }
        generateDishesFromMenu(selectedImage);
    });

    // Image generation functions
    async function generateImage(prompt) {
        console.log("Generating image with prompt:", prompt);
        
        // Show user message
        appendMessage(`Generate image: ${prompt}`, "user");
        showTypingIndicator();
        
        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('session_id', sessionId || '');
            formData.append('provider', ''); // Use default provider
            formData.append('size', '1024x1024');
            formData.append('num_images', '1');

            const response = await fetch('/api/v1/chat/generate-image', {
                method: 'POST',
                body: formData
            });

            removeTypingIndicator();

            if (!response.ok) {
                const errorData = await response.json();
                appendMessage(`Error generating image: ${errorData.detail || 'Unknown error'}`, "assistant");
                return;
            }

            const data = await response.json();
            sessionId = data.session_id;

            // Display the generated images
            if (data.images && data.images.length > 0) {
                // Show the text message first
                appendMessage(data.message, "assistant");
                // Then show each image as a separate HTML message
                data.images.forEach((imageData, index) => {
                    const imageUrl = `data:image/png;base64,${imageData.base64_data}`;
                    appendMessage(`<img src=\"${imageUrl}\" alt=\"Generated image ${index + 1}\" style=\"max-width: 300px; border-radius: 8px; margin: 5px;\">`, "assistant", false, null, true);
                });
            } else {
                appendMessage("No images were generated.", "assistant");
            }
        } catch (error) {
            console.error("Image generation error:", error);
            removeTypingIndicator();
            appendMessage("Error generating image. Please try again.", "assistant");
        }
    }

    async function generateDishesFromMenu(menuImage) {
        console.log("Generating dishes from menu image");
        
        // Show user message
        appendMessage("Generate dishes from menu image", "user", false, menuImage);
        showTypingIndicator();
        
        try {
            const formData = new FormData();
            formData.append('message', 'Generate images of the dishes from this menu');
            formData.append('image', menuImage);
            formData.append('style', 'photorealistic');
            formData.append('max_dishes', '5');

            const response = await fetch('/api/v1/chat/generate-images', {
                method: 'POST',
                body: formData
            });

            removeTypingIndicator();

            if (!response.ok) {
                const errorData = await response.json();
                appendMessage(`Error generating dish images: ${errorData.detail || 'Unknown error'}`, "assistant");
                return;
            }

            const data = await response.json();

            // Display the generated dish images
            if (data.generated_images && data.generated_images.length > 0) {
                let imagesHtml = '<div class="generated-dishes">';
                data.generated_images.forEach((imageData, index) => {
                    const imageUrl = `data:image/png;base64,${imageData.base64_data}`;
                    imagesHtml += `
                        <div class="generated-dish">
                            <img src="${imageUrl}" alt="Generated dish ${index + 1}" style="max-width: 200px; border-radius: 8px; margin: 5px;">
                            <p><strong>${imageData.dish_name || `Dish ${index + 1}`}</strong></p>
                            <p><small>${imageData.description || ''}</small></p>
                        </div>
                    `;
                });
                imagesHtml += '</div>';
                
                appendMessage(data.message + imagesHtml, "assistant", false, null, true);
            } else {
                appendMessage("No dish images were generated.", "assistant");
            }
        } catch (error) {
            console.error("Dish generation error:", error);
            removeTypingIndicator();
            appendMessage("Error generating dish images. Please try again.", "assistant");
        }
    }
});
