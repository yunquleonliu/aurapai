"""
Tools API routes for Auro-PAI Platform
======================================

Handles external tool operations including web search and URL fetching.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSearchRequest(BaseModel):
    """Web search request model."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    safe_search: bool = Field(default=True, description="Enable safe search")


class URLFetchRequest(BaseModel):
    """URL fetch request model."""
    url: HttpUrl = Field(..., description="URL to fetch")
    extract_text: bool = Field(default=True, description="Extract readable text from HTML")
    max_size: Optional[int] = Field(default=None, ge=1024, description="Maximum content size in bytes")


class SearchAndFetchRequest(BaseModel):
    """Combined search and fetch request model."""
    query: str = Field(..., description="Search query")
    fetch_top_results: int = Field(default=2, ge=1, le=5, description="Number of top results to fetch content from")
    max_search_results: int = Field(default=5, ge=1, le=20, description="Maximum search results")


class WebSearchResult(BaseModel):
    """Web search result model."""
    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Page URL")
    snippet: str = Field(..., description="Page snippet/description")
    source: str = Field(..., description="Search engine source")
    timestamp: str = Field(..., description="Search timestamp")


class URLFetchResult(BaseModel):
    """URL fetch result model."""
    url: str = Field(..., description="Fetched URL")
    content: str = Field(..., description="Page content")
    title: str = Field(..., description="Page title")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Fetch timestamp")
    content_length: int = Field(..., description="Content length in characters")


class SearchAndFetchResult(BaseModel):
    """Combined search and fetch result model."""
    query: str = Field(..., description="Original search query")
    search_results: List[WebSearchResult] = Field(..., description="Search results")
    fetched_content: List[Dict[str, Any]] = Field(..., description="Fetched content from top results")
    timestamp: str = Field(..., description="Operation timestamp")


@router.post("/tools/search", response_model=List[WebSearchResult])
async def web_search(request: WebSearchRequest, req: Request):
    """Perform web search using configured search engine."""
    try:
        tool_service = req.app.state.tool_service
        
        # Perform search
        results = await tool_service.web_search(
            query=request.query,
            max_results=request.max_results,
            safe_search=request.safe_search
        )
        
        # Convert to response format
        search_results = [
            WebSearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                source=result.source,
                timestamp=result.timestamp
            )
            for result in results
        ]
        
        logger.info(f"Web search completed: query='{request.query}', results={len(search_results)}")
        return search_results
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/fetch", response_model=URLFetchResult)
async def fetch_url(request: URLFetchRequest, req: Request):
    """Fetch content from a specific URL."""
    try:
        tool_service = req.app.state.tool_service
        
        # Fetch URL content
        result = await tool_service.fetch_url_content(
            url=str(request.url),
            extract_text=request.extract_text,
            max_size=request.max_size
        )
        
        # Convert to response format
        fetch_result = URLFetchResult(
            url=result.url,
            content=result.content,
            title=result.title,
            status_code=result.status_code,
            timestamp=result.timestamp,
            content_length=len(result.content)
        )
        
        logger.info(f"URL fetch completed: url='{request.url}', content_length={len(result.content)}")
        return fetch_result
        
    except Exception as e:
        logger.error(f"URL fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/search-and-fetch", response_model=SearchAndFetchResult)
async def search_and_fetch(request: SearchAndFetchRequest, req: Request):
    """Perform web search and fetch content from top results."""
    try:
        tool_service = req.app.state.tool_service
        
        # Perform search and fetch
        result = await tool_service.search_and_fetch(
            query=request.query,
            fetch_top_results=request.fetch_top_results,
            max_search_results=request.max_search_results
        )
        
        # Convert search results to response format
        search_results = [
            WebSearchResult(**search_result)
            for search_result in result["search_results"]
        ]
        
        response = SearchAndFetchResult(
            query=result["query"],
            search_results=search_results,
            fetched_content=result["fetched_content"],
            timestamp=result["timestamp"]
        )
        
        logger.info(f"Search and fetch completed: query='{request.query}', fetched={len(result['fetched_content'])}")
        return response
        
    except Exception as e:
        logger.error(f"Search and fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/available")
async def get_available_tools(req: Request):
    """Get information about available tools."""
    try:
        tool_service = req.app.state.tool_service
        tools_info = await tool_service.get_available_tools()
        return tools_info
    except Exception as e:
        logger.error(f"Error getting available tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/health")
async def tools_health_check(req: Request):
    """Perform health check on tool service."""
    try:
        tool_service = req.app.state.tool_service
        health = await tool_service.health_check()
        return health
    except Exception as e:
        logger.error(f"Tools health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Individual tool endpoints for more granular access

@router.get("/tools/search/engines")
async def get_search_engines():
    """Get available search engines."""
    return {
        "available_engines": ["duckduckgo"],
        "current_engine": "duckduckgo",
        "description": "DuckDuckGo provides privacy-focused web search"
    }


@router.post("/tools/validate-url")
async def validate_url(url: HttpUrl):
    """Validate a URL without fetching content."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(str(url))
        
        return {
            "valid": bool(parsed.scheme and parsed.netloc),
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
            "url": str(url)
        }
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid URL: {e}")


@router.get("/tools/usage-stats")
async def get_tool_usage_stats(req: Request):
    """Get tool usage statistics (placeholder for future implementation)."""
    # This would typically be implemented with a usage tracking system
    return {
        "message": "Tool usage statistics not yet implemented",
        "available_tools": ["web_search", "fetch_url", "search_and_fetch"],
        "note": "Usage tracking can be added in future versions"
    }
