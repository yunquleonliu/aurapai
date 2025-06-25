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
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.ddgs: Optional[DDGS] = None
        
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
        logger.info("Tool Service cleaned up")
    
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
