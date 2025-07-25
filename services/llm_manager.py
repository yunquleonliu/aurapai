"""
LLM Manager for Auro-PAI Platform
=================================

Manages communication with various LLM providers including local llama.cpp,
OpenAI, and Gemini. Provides unified interface for LLM interactions.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from enum import Enum

import google.generativeai as genai

from core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    LLAMACPP = "llamacpp"
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMResponse:
    """Standardized LLM response format."""
    
    def __init__(self, content: str, provider: str, model: str, usage: Optional[Dict] = None):
        self.content = content
        self.provider = provider
        self.model = model
        self.usage = usage or {}
        self.metadata = {}


class LLMManager:
    @staticmethod
    async def stream_answer(answer: str, chunk_size: int = 128):
        """
        Async generator to stream a pre-generated answer string in chunks.
        """
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i+chunk_size]
    """Manages LLM interactions across different providers."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_providers: List[LLMProvider] = []
        self.default_provider = LLMProvider.LLAMACPP
        self.gemini_model: Optional[genai.GenerativeModel] = None
        self.gemini_image_model: Optional[genai.GenerativeModel] = None
        
    async def initialize(self):
        """Initialize the LLM manager and check available providers."""
        logger.info("Initializing LLM Manager...")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=settings.LLAMACPP_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Configure Gemini
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                # Use latest supported Gemini models for text and image
                # Override config if old/deprecated model is set
                gemini_model_name = getattr(settings, 'GEMINI_MODEL', None) or 'gemini-1.5-flash'
                gemini_image_model_name = getattr(settings, 'GEMINI_IMAGE_MODEL', None) or 'gemini-1.5-flash'
                # If config uses deprecated model, force upgrade
                if 'pro-vision' in gemini_image_model_name or 'pro' in gemini_image_model_name:
                    gemini_image_model_name = 'gemini-1.5-flash'
                if 'pro' in gemini_model_name:
                    gemini_model_name = 'gemini-1.5-flash'
                self.gemini_model = genai.GenerativeModel(gemini_model_name)
                self.gemini_image_model = genai.GenerativeModel(gemini_image_model_name)
                logger.info(f"Gemini API configured successfully. Text model: {gemini_model_name}, Image model: {gemini_image_model_name}")
            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")
        
        # Check available providers
        await self._check_providers()
        
        if not self.available_providers:
            raise RuntimeError("No LLM providers available")
        
        logger.info(f"LLM Manager initialized with providers: {[p.value for p in self.available_providers]}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        logger.info("LLM Manager cleaned up")
    
    async def _check_providers(self):
        """Check which LLM providers are available."""
        self.available_providers = []
        
        # Check llama.cpp
        if await self._check_llamacpp():
            self.available_providers.append(LLMProvider.LLAMACPP)
        
        # Check OpenAI
        if settings.OPENAI_API_KEY:
            self.available_providers.append(LLMProvider.OPENAI)
        
        # Check Gemini
        if settings.GEMINI_API_KEY:
            self.available_providers.append(LLMProvider.GEMINI)
    
    async def _check_llamacpp(self) -> bool:
        """Check if llama.cpp server is available."""
        health_url = f"{settings.LLAMACPP_SERVER_URL}/health"
        logger.info(f"Checking llama.cpp server health at: {health_url}")
        try:
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    logger.info("llama.cpp server is available.")
                    return True
                else:
                    logger.error(f"llama.cpp server returned status {response.status} from {health_url}")
                    return False
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Could not connect to llama.cpp server at {health_url}. Connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while checking llama.cpp server: {e}")
            return False

    async def generate_stream(
        self, 
        provider: LLMProvider,
        prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Gateway for generating a streaming response from a specified LLM provider.

        Args:
            prompt (str): The user's prompt.
            provider (LLMProvider): The provider to use for generation.
            history (Optional[List[Dict]], optional): Conversation history. Defaults to None.
            temperature (float, optional): The generation temperature. Defaults to 0.7.
            max_tokens (int, optional): Max tokens to generate. Defaults to 2048.
            stop (Optional[List[str]], optional): Stop sequences. Defaults to None.

        Yields:
            AsyncGenerator[str, None]: A stream of response tokens.
        """
        if provider == LLMProvider.LLAMACPP:
            async for chunk in self._generate_llamacpp_stream(prompt, history, temperature, max_tokens, stop, **kwargs):
                yield chunk
        elif provider == LLMProvider.GEMINI:
            if not self.gemini_model:
                raise ValueError("Gemini provider is not configured.")
            
            # Check for image generation keywords
            is_image_generation = False
            if prompt and any(keyword in prompt.lower() for keyword in ["generate image", "create a picture", "draw"]):
                is_image_generation = True

            # Remove is_image_generation from kwargs if present, and set it explicitly
            is_image_generation = kwargs.pop('is_image_generation', False)
            async for chunk in self._generate_gemini_stream(prompt, history, temperature, max_tokens, stop, is_image_generation=is_image_generation, **kwargs):
                yield chunk
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def _generate_gemini_stream(
        self,
        prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        is_image_generation: bool = False,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Gemini. Logs and yields raw Gemini output for debugging image generation."""
        logger.debug(f"Generating stream from Gemini with prompt: {prompt[:100] if prompt else ''}...")
        try:
            model_to_use = self.gemini_image_model if is_image_generation else self.gemini_model
            if not model_to_use:
                raise ValueError("The requested Gemini model is not configured.")

            # Note: Gemini API uses 'temperature' and 'max_output_tokens'.
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop
            )

            chat_history = []
            if history:
                for message in history:
                    role = "model" if message.get("role") == "assistant" else "user"
                    chat_history.append({"role": role, "parts": [message.get("content", "")]})

            final_prompt = prompt if prompt and not (history and history[-1]['content'] == prompt) else prompt

            chat = model_to_use.start_chat(history=chat_history)
            response = await asyncio.to_thread(
                chat.send_message,
                final_prompt,
                stream=True,
                generation_config=generation_config
            )

            # --- DEBUG: Log and yield raw Gemini output for image generation ---
            for chunk in response:
                logger.info(f"Gemini raw chunk: {repr(chunk)}")
                # Try to yield both text and any possible image fields
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                # If Gemini returns images as base64 or URLs, log and yield them
                if hasattr(chunk, 'candidates'):
                    logger.info(f"Gemini chunk.candidates: {repr(chunk.candidates)}")
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content'):
                            logger.info(f"Gemini candidate.content: {repr(candidate.content)}")
                        if hasattr(candidate, 'images'):
                            logger.info(f"Gemini candidate.images: {repr(candidate.images)}")
                        # Try to yield image data if present
                        if hasattr(candidate, 'images') and candidate.images:
                            for img in candidate.images:
                                logger.info(f"Gemini image: {repr(img)}")
                                yield str(img)

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            yield f"Error: Could not get response from Gemini. Details: {e}"

    async def _generate_llamacpp_stream(
        self,
        prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the llama.cpp server."""
        if not self.session:
            raise RuntimeError("LLMManager is not initialized. Call initialize() first.")

        completion_url = f"{settings.LLAMACPP_SERVER_URL}/v1/chat/completions"
        
        # Construct messages payload
        messages = []
        if history:
            messages.extend(history)
        
        # Append the prompt as a user message only if it's not already the last message in history
        if prompt and (not messages or messages[-1].get('content') != prompt):
            # Check if the prompt is a JSON string representing a list of messages
            try:
                prompt_messages = json.loads(prompt)
                if isinstance(prompt_messages, list):
                    # It's a structured prompt, use it as the message list
                    messages = prompt_messages
                else:
                    # It's a simple string, append it
                    messages.append({"role": "user", "content": prompt})
            except json.JSONDecodeError:
                # It's just a regular string prompt
                messages.append({"role": "user", "content": prompt})

        payload = {
            "stream": True,  # Enable streaming to get tokens as they are generated
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "messages": messages if messages else [{"role": "user", "content": prompt}]
        }
        
        # For /v1/chat/completions, always send 'messages' array (do not send 'prompt')
        # No changes needed here; payload already has 'messages' as required by OpenAI API

        logger.info(f"[LLAMACPP DEBUG] Sending payload to llama.cpp at {completion_url}:")
        logger.info(json.dumps(payload, indent=2))
        logger.info("------------------------------------")

        try:
            async with self.session.post(completion_url, json=payload) as response:
                logger.info(f"[LLAMACPP DEBUG] llama.cpp server response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"llama.cpp server returned status {response.status}: {error_text}")
                    yield f"Error: llama.cpp server returned status {response.status}"
                    return

                # Stream each chunk as-is, with minimal processing
                async for chunk in response.content:
                    if not chunk:
                        continue
                    yield chunk.decode("utf-8")
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Could not connect to llama.cpp server at {completion_url}. Connection error: {e}")
            yield f"Error: Could not connect to llama.cpp server."
        except Exception as e:
            logger.error(f"An unexpected error occurred during llama.cpp stream generation: {e}")
            yield f"Error: An unexpected error occurred."

    async def generate_response(
        self,
        provider: LLMProvider,
        prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Gateway for generating a non-streaming response from a specified LLM provider.
        Forces non-streaming (stream: false) mode to check if LLM output is more structured for ReAct parsing.
        """
        is_image_generation = kwargs.pop('is_image_generation', False)
        # --- Force non-streaming mode for this call ---
        if provider == LLMProvider.LLAMACPP:
            if not self.session:
                raise RuntimeError("LLMManager is not initialized. Call initialize() first.")
            completion_url = f"{settings.LLAMACPP_SERVER_URL}/v1/chat/completions"
            messages = []
            if history:
                messages.extend(history)
            if prompt and (not messages or messages[-1].get('content') != prompt):
                try:
                    prompt_messages = json.loads(prompt)
                    if isinstance(prompt_messages, list):
                        messages = prompt_messages
                    else:
                        messages.append({"role": "user", "content": prompt})
                except json.JSONDecodeError:
                    messages.append({"role": "user", "content": prompt})
            payload = {
                "stream": False,  # Force non-streaming
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": stop or [],
                "messages": messages if messages else [{"role": "user", "content": prompt}]
            }
            logger.info(f"[LLAMACPP DEBUG] Sending NON-STREAM payload to llama.cpp at {completion_url}:")
            logger.info(json.dumps(payload, indent=2))
            logger.info("------------------------------------")
            content = ""
            try:
                async with self.session.post(completion_url, json=payload) as response:
                    logger.info(f"[LLAMACPP DEBUG] llama.cpp server response status: {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"llama.cpp server returned status {response.status}: {error_text}")
                        content = f"Error: llama.cpp server returned status {response.status}"
                    else:
                        result = await response.json()
                        # OpenAI compatible: result['choices'][0]['message']['content']
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            except Exception as e:
                logger.error(f"Error in non-streaming llama.cpp completion: {e}")
                content = f"Error: {e}"
            model_name = settings.LLAMACPP_MODEL_NAME
        elif provider == LLMProvider.GEMINI:
            # For Gemini, fallback to streaming aggregation as before
            full_response = []
            async for chunk in self.generate_stream(
                provider, prompt, history, temperature, max_tokens, stop, is_image_generation=is_image_generation, **kwargs
            ):
                full_response.append(chunk)
            content = "".join(chunk for chunk in full_response if chunk is not None)
            model_name = settings.GEMINI_MODEL
        else:
            # Default: aggregate streaming
            full_response = []
            async for chunk in self.generate_stream(
                provider, prompt, history, temperature, max_tokens, stop, is_image_generation=is_image_generation, **kwargs
            ):
                full_response.append(chunk)
            content = "".join(chunk for chunk in full_response if chunk is not None)
            model_name = ""
        return LLMResponse(
            content=content,
            provider=provider.value,
            model=model_name
        )
