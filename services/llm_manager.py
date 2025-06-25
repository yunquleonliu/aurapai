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
    """Manages LLM interactions across different providers."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_providers: List[LLMProvider] = []
        self.default_provider = LLMProvider.LLAMACPP
        
    async def initialize(self):
        """Initialize the LLM manager and check available providers."""
        logger.info("Initializing LLM Manager...")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=settings.LLAMACPP_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
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
        try:
            async with self.session.get(f"{settings.LLAMACPP_SERVER_URL}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"llama.cpp server not available: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
        health_status = {
            "available_providers": [p.value for p in self.available_providers],
            "default_provider": self.default_provider.value,
            "providers": {}
        }
        
        for provider in self.available_providers:
            if provider == LLMProvider.LLAMACPP:
                is_healthy = await self._check_llamacpp()
                health_status["providers"]["llamacpp"] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "url": settings.LLAMACPP_SERVER_URL
                }
            elif provider == LLMProvider.OPENAI:
                health_status["providers"]["openai"] = {
                    "status": "configured",
                    "model": settings.OPENAI_MODEL
                }
            elif provider == LLMProvider.GEMINI:
                health_status["providers"]["gemini"] = {
                    "status": "configured",
                    "model": settings.GEMINI_MODEL
                }
        
        return health_status
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Generate a response using the specified or default provider."""
        
        if provider is None:
            provider = self.default_provider
        
        if provider not in self.available_providers:
            raise ValueError(f"Provider {provider.value} is not available")
        
        try:
            if provider == LLMProvider.LLAMACPP:
                return await self._generate_llamacpp(messages, temperature, max_tokens)
            elif provider == LLMProvider.OPENAI:
                return await self._generate_openai(messages, temperature, max_tokens)
            elif provider == LLMProvider.GEMINI:
                return await self._generate_gemini(messages, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {provider.value}")
                
        except Exception as e:
            logger.error(f"Error generating response with {provider.value}: {e}")
            raise
    
    async def _generate_llamacpp(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> LLMResponse:
        """Generate response using llama.cpp server."""
        
        # Convert messages to llama.cpp format
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens or 512,
            "stop": ["</s>", "[INST]", "[/INST]"],
            "stream": False
        }
        
        url = f"{settings.LLAMACPP_SERVER_URL}/completion"
        
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"llama.cpp error: {error_text}")
            
            result = await response.json()
            content = result.get("content", "").strip()
            
            return LLMResponse(
                content=content,
                provider="llamacpp",
                model=settings.LLAMACPP_MODEL_NAME,
                usage={
                    "prompt_tokens": result.get("tokens_evaluated", 0),
                    "completion_tokens": result.get("tokens_predicted", 0)
                }
            )
    
    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        url = "https://api.openai.com/v1/chat/completions"
        
        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"OpenAI API error: {error_text}")
            
            result = await response.json()
            content = result["choices"][0]["message"]["content"]
            
            return LLMResponse(
                content=content,
                provider="openai",
                model=settings.OPENAI_MODEL,
                usage=result.get("usage", {})
            )
    
    async def _generate_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> LLMResponse:
        """Generate response using Google Gemini API."""
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens or 512
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent"
        params = {"key": settings.GEMINI_API_KEY}
        
        async with self.session.post(url, json=payload, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Gemini API error: {error_text}")
            
            result = await response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            return LLMResponse(
                content=content,
                provider="gemini",
                model=settings.GEMINI_MODEL,
                usage=result.get("usageMetadata", {})
            )
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt format for llama.cpp."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)
    
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        
        if provider is None:
            provider = self.default_provider
        
        if provider not in self.available_providers:
            raise ValueError(f"Provider {provider.value} is not available")
        
        if provider == LLMProvider.LLAMACPP:
            async for chunk in self._stream_llamacpp(messages, temperature, max_tokens):
                yield chunk
        else:
            # For non-streaming providers, yield the full response
            response = await self.generate_response(messages, provider, temperature, max_tokens)
            yield response.content
    
    async def _stream_llamacpp(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> AsyncGenerator[str, None]:
        """Stream response from llama.cpp server."""
        
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens or 512,
            "stop": ["</s>", "[INST]", "[/INST]"],
            "stream": True
        }
        
        url = f"{settings.LLAMACPP_SERVER_URL}/completion"
        
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"llama.cpp streaming error: {error_text}")
            
            async for line in response.content:
                if line:
                    try:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            data = json.loads(line_text[6:])
                            if 'content' in data:
                                yield data['content']
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
