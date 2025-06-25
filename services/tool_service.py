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
import requests

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
    async def search_and_fetch(self, query: str, fetch_top_results: int = 2, max_search_results: int = 5, summarize: bool = False):
        """Perform search and fetch content from top results, streaming each result as soon as it is fetched. Optionally summarize using public LLM if requested (never local LLM)."""
        import json
        from fastapi.responses import StreamingResponse
        logger.info(f"Streaming search and fetch for: '{query}' (Google Custom Search)")
        search_results = await self.web_search(query, max_search_results)

        async def result_generator():
            for i, result in enumerate(search_results[:fetch_top_results]):
                try:
                    url_content = await self.fetch_url_content(result.url)
                    # Defensive: If fetch failed or returned HTML, yield error JSON
                    if not isinstance(url_content, dict) or url_content.get("status") == "error":
                        yield json.dumps({
                            "search_result": result.to_dict(),
                            "error": url_content.get("message") if isinstance(url_content, dict) else "Fetch failed or returned non-JSON content"
                        }) + '\n'
                        continue
                    # If content looks like HTML (starts with <!DOCTYPE or <html), treat as error
                    content_text = url_content.get("content", "")
                    if isinstance(content_text, str) and content_text.strip().lower().startswith("<!doctype"):
                        yield json.dumps({
                            "search_result": result.to_dict(),
                            "error": "Fetched content is HTML, not plain text."
                        }) + '\n'
                        continue
                    # Summarize only if requested, using public LLM only (never local)
                    summary_text = None
                    if summarize:
                        title = url_content.get("title", "")
                        if content_text:
                            try:
                                prompt = f"Summarize the following article. Title: {title}\nContent: {content_text[:2000]}"
                                summary = await self.call_public_llm(prompt)
                                summary_text = summary if isinstance(summary, str) else str(summary)
                            except Exception as llm_e:
                                summary_text = f"Error during summarization: {llm_e}"
                        else:
                            summary_text = "No content to summarize."
                    # Always yield fetched content and summary (if any) in a single JSON object
                    result_obj = {
                        "search_result": result.to_dict(),
                        "content": url_content
                    }
                    if summarize:
                        result_obj["summary"] = summary_text
                    yield json.dumps(result_obj) + '\n'
                except Exception as e:
                    yield json.dumps({
                        "search_result": result.to_dict(),
                        "error": str(e)
                    }) + '\n'
        return StreamingResponse(result_generator(), media_type='application/json')
    
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
                "description": "Searches the web for information ONLY if the answer is not known or is likely to be recent, up-to-date, or not in the LLM's training data. Do NOT use for general knowledge or common facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"]
                }
            },
            "fetch_url": {
                "function": self.fetch_url_content,
                "description": "Fetches the content of a given URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch the content from."},
                        "extract_text": {"type": "boolean", "description": "Extract readable text from HTML (default: true)"}
                    },
                    "required": ["url"]
                }
            },
            "search_and_fetch": {
                "function": self.search_and_fetch,
                "description": "Search web and fetch content from top results. Use ONLY if the answer is not known or is likely to be recent or not in the LLM's training data.",
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
            "interpret_image": {
                "function": self.interpret_image,
                "description": "Interprets or analyzes an uploaded image (OCR, description, etc) using the local Mixtral+LLaVA model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string", "description": "Base64-encoded image data."},
                        "task": {"type": "string", "description": "Analysis task, e.g. 'describe', 'ocr', 'detect_objects' (optional)"}
                    },
                    "required": ["image_base64"]
                }
            },
            "call_public_llm": {
                "function": self.call_public_llm,
                "description": "Calls a powerful public LLM (like Gemini) for complex reasoning, creative tasks, or when the user requests a more powerful or public LLM. Use this when the local model is insufficient or the user requests it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query or prompt to send to the public LLM."}
                    },
                    "required": ["query"]
                }
            }
        }
    async def interpret_image(self, image_base64: str, task: str = "describe") -> Dict[str, Any]:
        """Interprets or analyzes an uploaded image using the local Mixtral+LLaVA model."""
        logger.info(f"Interpreting image with task '{task}' using Mixtral+LLaVA.")
        try:
            # Compose prompt for vision model
            prompt = f"[IMAGE]\nTask: {task or 'describe'}"
            # Call local LLMManager with vision capability (assume LLAMACPP is Mixtral+LLaVA)
            response = await self.llm_manager.generate_response(
                provider=LLMProvider.LLAMACPP,
                prompt=prompt,
                image_base64=image_base64,
                task=task
            )
            return {
                "status": "success",
                "type": "image_interpretation",
                "task": task,
                "result": response.content
            }
        except Exception as e:
            logger.error(f"Error interpreting image: {e}", exc_info=True)
            return {
                "status": "error",
                "type": "image_interpretation",
                "task": task,
                "message": f"Failed to interpret image. Details: {str(e)}"
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
        """Perform web search using Google Custom Search API."""
        if not settings.WEB_SEARCH_ENABLED:
            raise RuntimeError("Web search is disabled")
        logger.info(f"Performing web search (Google Custom Search): '{query}'")
        try:
            results = self.google_custom_search(query, max_results)
            search_results = []
            for result in results:
                search_result = WebSearchResult(
                    title=result.get('title', ''),
                    url=result.get('link', ''),
                    snippet=result.get('snippet', ''),
                    source='google_custom_search'
                )
                search_results.append(search_result)
            logger.info(f"Google Custom Search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Google Custom Search failed: {e}")
            raise
    
    # DuckDuckGo search is deprecated and replaced by Google Custom Search
    
    async def fetch_url_content(
        self,
        url: str,
        extract_text: bool = True,
        max_size: int = None
    ) -> dict:
        """Fetch content from a URL and always return a JSON-serializable dict."""
        if max_size is None:
            max_size = settings.URL_FETCH_MAX_SIZE
        logger.info(f"Fetching URL: {url}")
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"status": "error", "message": "Invalid URL format", "url": url}
        try:
            async with self.session.get(url) as response:
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size:
                    return {"status": "error", "message": f"Content too large: {content_length} bytes", "url": url}
                content = await response.read()
                if len(content) > max_size:
                    content = content[:max_size]
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = content.decode('latin-1', errors='ignore')
                if extract_text:
                    text_content = self._extract_text(text_content)
                title = self._extract_title(text_content) if extract_text else ""
                result = URLFetchResult(
                    url=url,
                    content=text_content,
                    title=title,
                    status_code=response.status
                )
                logger.info(f"Successfully fetched URL: {url} ({len(text_content)} chars)")
                return result.to_dict()
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {"status": "error", "message": f"Failed to fetch URL: {str(e)}", "url": url}
    
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
                if isinstance(result, list) and result:
                    img = result[0]
                    if hasattr(img, 'to_dict'):
                        img = img.to_dict()
                    base64_data = img.get('base64_data')
                    image_url = f"data:image/png;base64,{base64_data}" if base64_data else None
                    return {
                        "status": "success",
                        "type": "image",
                        "image_url": image_url,
                        "base64_data": base64_data,
                        "message": f"Successfully generated image for: {img.get('prompt', '')}"
                    }
                if isinstance(result, dict) and result.get('type') == 'image':
                    return result

            # Standardize web_search tool output
            if tool_name == 'web_search':
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

            # Standardize fetch_url tool output
            if tool_name == 'fetch_url':
                if isinstance(result, dict):
                    return result
                elif hasattr(result, 'to_dict'):
                    return result.to_dict()
                else:
                    return {"status": "error", "message": "Unknown fetch_url result type"}

            return result
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}", exc_info=True)
            return {"status": "error", "message": f"Tool invocation failed: {str(e)}"}

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

    def google_custom_search(self, query, max_results=5):
        url = "https://www.googleapis.com/customsearch/v1"
        api_key = getattr(settings, "GOOGLE_API_KEY", None)
        cse_id = getattr(settings, "GOOGLE_CSE_ID", None)
        if not api_key or not cse_id:
            raise RuntimeError("Google Custom Search API key or CSE ID not set in settings.")
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": max_results
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "source": "google_custom_search"
                })
            logger.info(f"Custom Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Custom Search error: {e}")
            raise
