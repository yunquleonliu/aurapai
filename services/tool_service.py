"""
Tool Service for Auro-PAI Platform
==================================

Provides external tool capabilities including web search and URL fetching
for the AI to gather external information.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import re
from datetime import datetime

from services.llm_manager import LLMManager, LLMProvider

try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    logging.warning("DuckDuckGo search not available. Install duckduckgo-search package.")

from core.config import settings

logger = logging.getLogger(__name__)


class WebSearchResult:
    """Represents a web search result."""
    
    def __init__(self, title: str, url: str, snippet: str, source: str = "unknown"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "timestamp": self.timestamp
        }


class URLFetchResult:
    """Represents the result of fetching content from a URL."""
    
    def __init__(self, url: str, content: str, title: str = "", status_code: int = 200):
        self.url = url
        self.content = content
        self.title = title
        self.status_code = status_code
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "content": self.content,
            "title": self.title,
            "status_code": self.status_code,
            "timestamp": self.timestamp,
            "content_length": len(self.content)
        }


class ToolService:
    """Manages external tool capabilities for AI assistance."""
    
    def __init__(self, llm_manager: LLMManager):
        self.session: Optional[aiohttp.ClientSession] = None
        self.ddgs: Optional[DDGS] = None
        self.llm_manager = llm_manager
        
        # This is a placeholder. In a real scenario, you would have a more robust
        # mechanism for defining and registering tools.
        self.tools = {
            "clarification": {
                "function": self.clarification_tool,
                "description": "Ask the user for clarification when the agent is unsure or needs more information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The clarifying question to ask the user."}
                    },
                    "required": ["question"]
                }
            },
            "web_search": {
                "function": self.web_search,
                "description": "Searches the web for information on a given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"]
                }
            },
            "search_and_fetch": {
                "function": self.search_and_fetch,
                "description": "Search web and fetch content from top results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string."},
                        "fetch_top_results": {"type": "integer", "description": "Number of top results to fetch content from (default: 2)"},
                        "max_search_results": {"type": "integer", "description": "Maximum search results (default: 5)"}
                    },
                    "required": ["query"]
                }
            },
            "generate_image": {
                "function": self.generate_image,
                "description": "Generates an image based on a textual description (prompt). This tool uses a powerful public model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "A detailed description of the image to generate."},
                        "num_images": {"type": "integer", "description": "Number of images to generate", "default": 1}
                    },
                    "required": ["prompt"]
                }
            },
            "call_public_llm": {
                "function": self.call_public_llm,
                "description": "Calls a powerful public LLM (like Gemini) for complex reasoning, creative tasks, or up-to-date information. Use this when the local model is insufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query or prompt to send to the public LLM."}
                    },
                    "required": ["query"]
                }
            }
        }

    async def initialize(self):
        """Initialize the tool service."""
        logger.info("Initializing Tool Service...")
        
        # Create HTTP session for URL fetching
        timeout = aiohttp.ClientTimeout(total=settings.URL_FETCH_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                'User-Agent': 'Auro-PAI/1.0 (AI Assistant)'
            }
        )
        
        # Initialize DuckDuckGo search if available
        if DUCKDUCKGO_AVAILABLE:
            self.ddgs = DDGS()
            logger.info("DuckDuckGo search initialized")
        
        logger.info("Tool Service initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on tool service."""
        capabilities = {
            "web_search": settings.WEB_SEARCH_ENABLED and DUCKDUCKGO_AVAILABLE,
            "url_fetch": True,
            "search_engine": settings.WEB_SEARCH_ENGINE if settings.WEB_SEARCH_ENABLED else None
        }
        
        return {
            "status": "healthy",
            "capabilities": capabilities
        }
    
    async def web_search(
        self,
        query: str,
        max_results: int = 5,
        safe_search: bool = True
    ) -> List[WebSearchResult]:
        """Perform web search using configured search engine."""
        if not settings.WEB_SEARCH_ENABLED:
            raise RuntimeError("Web search is disabled")
        
        logger.info(f"Performing web search: '{query}'")
        
        try:
            if settings.WEB_SEARCH_ENGINE == "duckduckgo" and DUCKDUCKGO_AVAILABLE:
                return await self._search_duckduckgo(query, max_results, safe_search)
            else:
                raise RuntimeError(f"Search engine '{settings.WEB_SEARCH_ENGINE}' not available")
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise
    
    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        safe_search: bool
    ) -> List[WebSearchResult]:
        """Search using DuckDuckGo."""
        try:
            # Run search in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def search():
                safesearch = 'strict' if safe_search else 'off'
                return list(self.ddgs.text(query, max_results=max_results, safesearch=safesearch))
            
            results = await loop.run_in_executor(None, search)
            
            search_results = []
            for result in results:
                search_result = WebSearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    source='duckduckgo'
                )
                search_results.append(search_result)
            
            logger.info(f"DuckDuckGo search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            raise
    
    async def fetch_url_content(
        self,
        url: str,
        extract_text: bool = True,
        max_size: int = None
    ) -> URLFetchResult:
        """Fetch content from a URL."""
        if max_size is None:
            max_size = settings.URL_FETCH_MAX_SIZE
        
        logger.info(f"Fetching URL: {url}")
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        
        try:
            async with self.session.get(url) as response:
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size:
                    raise ValueError(f"Content too large: {content_length} bytes")
                
                # Read content with size limit
                content = await response.read()
                if len(content) > max_size:
                    content = content[:max_size]
                
                # Decode content
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content.decode('latin-1', errors='ignore')
                
                # Extract readable text if requested
                if extract_text:
                    text_content = self._extract_text(text_content)
                
                # Extract title
                title = self._extract_title(text_content) if extract_text else ""
                
                result = URLFetchResult(
                    url=url,
                    content=text_content,
                    title=title,
                    status_code=response.status
                )
                
                logger.info(f"Successfully fetched URL: {url} ({len(text_content)} chars)")
                return result
                
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            raise
    
    def _extract_text(self, html_content: str) -> str:
        """Extract readable text from HTML content."""
        # Simple HTML text extraction
        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content."""
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up title
            title = re.sub(r'\s+', ' ', title)
            return title
        return ""
    
    async def search_and_fetch(
        self,
        query: str,
        fetch_top_results: int = 2,
        max_search_results: int = 5
    ) -> Dict[str, Any]:
        """Perform search and fetch content from top results."""
        logger.info(f"Search and fetch for: '{query}'")
        
        # Perform search
        search_results = await self.web_search(query, max_search_results)
        
        # Fetch content from top results
        fetched_content = []
        for i, result in enumerate(search_results[:fetch_top_results]):
            try:
                url_content = await self.fetch_url_content(result.url)
                fetched_content.append({
                    "search_result": result.to_dict(),
                    "content": url_content.to_dict()
                })
            except Exception as e:
                logger.warning(f"Failed to fetch content from {result.url}: {e}")
                continue
        
        return {
            "query": query,
            "search_results": [r.to_dict() for r in search_results],
            "fetched_content": fetched_content,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available tools."""
        tools = {}
        
        if settings.WEB_SEARCH_ENABLED:
            tools["web_search"] = {
                "description": "Search the web for information",
                "parameters": {
                    "query": "Search query string",
                    "max_results": "Maximum number of results (default: 5)",
                    "safe_search": "Enable safe search (default: true)"
                },
                "available": DUCKDUCKGO_AVAILABLE,
                "engine": settings.WEB_SEARCH_ENGINE
            }
        
        tools["fetch_url"] = {
            "description": "Fetch content from a specific URL",
            "parameters": {
                "url": "URL to fetch",
                "extract_text": "Extract readable text from HTML (default: true)",
                "max_size": f"Maximum content size in bytes (default: {settings.URL_FETCH_MAX_SIZE})"
            },
            "available": True
        }
        
        tools["search_and_fetch"] = {
            "description": "Search web and fetch content from top results",
            "parameters": {
                "query": "Search query string",
                "fetch_top_results": "Number of top results to fetch content from (default: 2)",
                "max_search_results": "Maximum search results (default: 5)"
            },
            "available": settings.WEB_SEARCH_ENABLED and DUCKDUCKGO_AVAILABLE
        }
        
        return {
            "available_tools": list(tools.keys()),
            "tools": tools
        }
    
    def get_tool_definitions(self):
        """Return a summary of available tools and their parameters for LLM prompt injection."""
        tools = []
        if getattr(settings, 'WEB_SEARCH_ENABLED', False):
            tools.append({
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": [
                    {"name": "query", "type": "string", "description": "Search query string"},
                    {"name": "max_results", "type": "int", "description": "Maximum number of results (default: 5)"},
                    {"name": "safe_search", "type": "bool", "description": "Enable safe search (default: true)"}
                ]
            })
        tools.append({
            "name": "fetch_url",
            "description": "Fetch content from a specific URL",
            "parameters": [
                {"name": "url", "type": "string", "description": "URL to fetch"},
                {"name": "extract_text", "type": "bool", "description": "Extract readable text from HTML (default: true)"},
                {"name": "max_size", "type": "int", "description": f"Maximum content size in bytes (default: {getattr(settings, 'URL_FETCH_MAX_SIZE', 1024*1024)})"}
            ]
        })
        tools.append({
            "name": "search_and_fetch",
            "description": "Search web and fetch content from top results",
            "parameters": [
                {"name": "query", "type": "string", "description": "Search query string"},
                {"name": "fetch_top_results", "type": "int", "description": "Number of top results to fetch content from (default: 2)"},
                {"name": "max_search_results", "type": "int", "description": "Maximum search results (default: 5)"}
            ]
        })
        # --- Add image generation/interpretation tools ---
        tools.append({
            "name": "generate_image",
            "description": "Generate an image from a text prompt using a powerful AI image model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "A detailed text prompt for the image generation."},
                    "num_images": {"type": "integer", "description": "Number of images to generate (default: 1)"}
                },
                "required": ["prompt"]
            }
        })
        tools.append({
            "name": "interpret_image",
            "description": "Interpret or analyze an uploaded image (OCR, description, etc).",
            "parameters": [
                {"name": "image_base64", "type": "string", "description": "Base64-encoded image data."},
                {"name": "task", "type": "string", "description": "Analysis task, e.g. 'describe', 'ocr', 'detect_objects' (optional)"}
            ]
        })
        tools.append(
            {
                "name": "clarification",
                "description": "Asks the user for more information when the query is ambiguous or incomplete.",
                "arguments": {
                    "question": {"type": "string", "description": "The question to ask the user."}
                }
            }
        )
        return tools

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Dynamically invoke a tool by its name. Standardizes image tool output."""
        logger.info(f"Invoking tool: {tool_name} with args: {arguments}")

        # Intent-driven invocation: handle nested or intent-based tool calls
        if isinstance(tool_name, dict):
            tool_name_flat = tool_name.get('name')
            arguments = tool_name.get('parameters', {})
            tool_name = tool_name_flat
        elif isinstance(arguments, dict) and 'intent' in arguments:
            if tool_name == 'generate_image':
                arguments = {'prompt': arguments['intent'], 'num_images': 1}
            elif tool_name == 'web_search':
                arguments = {'query': arguments['intent'], 'max_results': 5, 'safe_search': True}

        if tool_name not in self.tools:
            raise NotImplementedError(f"Tool '{tool_name}' is not implemented or defined.")

        tool_method = self.tools[tool_name].get("function")
        if not tool_method:
            raise NotImplementedError(f"Tool '{tool_name}' does not have a function defined.")

        try:
            result = await tool_method(**arguments) if asyncio.iscoroutinefunction(tool_method) else tool_method(**arguments)

            # Standardize image tool output
            if tool_name == 'generate_image':
                # If result is a list of GeneratedImage or dicts, take the first
                if isinstance(result, list) and result:
                    img = result[0]
                    # If it's a GeneratedImage object, convert to dict
                    if hasattr(img, 'to_dict'):
                        img = img.to_dict()
                    # Build a data URL for frontend
                    base64_data = img.get('base64_data')
                    image_url = f"data:image/png;base64,{base64_data}" if base64_data else None
                    return {
                        "status": "success",
                        "type": "image",
                        "image_url": image_url,
                        "base64_data": base64_data,
                        "message": f"Successfully generated image for: {img.get('prompt', '')}"
                    }
                # If already a dict with image_url, just return
                if isinstance(result, dict) and result.get('type') == 'image':
                    return result

            # Standardize web_search tool output
            if tool_name == 'web_search':
                # If result is a list of WebSearchResult or dicts, convert to dict
                if isinstance(result, list) and result and hasattr(result[0], 'to_dict'):
                    search_results = [r.to_dict() for r in result]
                elif isinstance(result, list) and result and isinstance(result[0], dict):
                    search_results = result
                else:
                    search_results = []
                return {
                    "type": "web_search",
                    "search_results": search_results,
                    "query": arguments.get("query", ""),
                    "timestamp": datetime.utcnow().isoformat()
                }

            return result
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}", exc_info=True)
            raise

    async def clarification_tool(self, question: str) -> Dict[str, str]:
        """A tool that represents asking for clarification. Returns structured response."""
        return {"status": "success", "type": "clarification", "message": question}

    async def generate_image(self, prompt: str, num_images: int = 1) -> Dict:
        """Generates an image using Hugging Face Inference API (Stable Diffusion)."""
        logger.info(f"Generating {num_images} image(s) with prompt: '{prompt}' via Hugging Face API")
        if not settings.HUGGINGFACE_API_KEY:
            logger.warning("No Hugging Face API key set. Cannot generate image.")
            return {"status": "error", "type": "image", "message": "Hugging Face API key not set. Please configure it in backend.", "image_url": None}
        try:
            import httpx
            # Use a free, community model for image generation (confirmed working)
            api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
            payload = {"inputs": prompt}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    # Hugging Face returns image bytes directly
                    image_bytes = response.content
                    import base64
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_url = f"data:image/png;base64,{image_b64}"
                    return {"status": "success", "type": "image", "image_url": image_url, "message": f"Successfully generated image for: {prompt}"}
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} {response.text}")
                    return {"status": "error", "type": "image", "message": f"Hugging Face API error: {response.text}", "image_url": None}
        except Exception as e:
            logger.error(f"Error generating image for prompt '{prompt}': {e}", exc_info=True)
            return {"status": "error", "type": "image", "message": f"Failed to generate image. Details: {str(e)}", "image_url": None}

    async def generate_video(self, prompt: str) -> Dict:
        """Generates a video using Hugging Face Inference API (text-to-video)."""
        logger.info(f"Generating video with prompt: '{prompt}' via Hugging Face API")
        if not settings.HUGGINGFACE_API_KEY:
            logger.warning("No Hugging Face API key set. Cannot generate video.")
            return {"status": "error", "type": "video", "message": "Hugging Face API key not set. Please configure it in backend.", "video_url": None}
        try:
            import httpx
            api_url = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"
            headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
            payload = {"inputs": prompt}
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    # Hugging Face returns video bytes directly
                    video_bytes = response.content
                    import base64
                    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
                    video_url = f"data:video/mp4;base64,{video_b64}"
                    return {"status": "success", "type": "video", "video_url": video_url, "message": f"Successfully generated video for: {prompt}"}
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} {response.text}")
                    return {"status": "error", "type": "video", "message": f"Hugging Face API error: {response.text}", "video_url": None}
        except Exception as e:
            logger.error(f"Error generating video for prompt '{prompt}': {e}", exc_info=True)
            return {"status": "error", "type": "video", "message": f"Failed to generate video. Details: {str(e)}", "video_url": None}

    async def call_public_llm(self, query: str) -> str:
        """Calls the public LLM via the LLMManager."""
        logger.info(f"Calling public LLM with query: '{query}'")
        try:
            response = await self.llm_manager.generate_response(
                provider=LLMProvider.GEMINI,
                prompt=query
            )
            return response.content
        except Exception as e:
            logger.error(f"Error calling public LLM for query '{query}': {e}")
            return {"error": str(e)}
