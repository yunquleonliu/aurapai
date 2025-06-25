"""
RAG API routes for Auro-PAI Platform
====================================

Handles RAG (Retrieval-Augmented Generation) operations including
document indexing and knowledge base search.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter()


class IndexRequest(BaseModel):
    """Document indexing request model."""
    directory_path: str = Field(..., description="Path to directory to index")
    file_patterns: Optional[List[str]] = Field(default=None, description="File patterns to include")


class SearchRequest(BaseModel):
    """RAG search request model."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    file_filter: Optional[str] = Field(default=None, description="Filter by file pattern")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")


class SearchResult(BaseModel):
    """Search result model."""
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source file path")
    similarity: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Search response model."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")


class IndexStatus(BaseModel):
    """Indexing status model."""
    status: str = Field(..., description="Indexing status")
    indexed_files: int = Field(..., description="Number of files indexed")
    skipped_files: int = Field(..., description="Number of files skipped")
    errors: List[str] = Field(default_factory=list, description="Indexing errors")
    total_documents: int = Field(..., description="Total documents in collection")


@router.post("/rag/index", response_model=IndexStatus)
async def index_directory(request: IndexRequest, req: Request, background_tasks: BackgroundTasks):
    """Index a directory for RAG retrieval."""
    try:
        rag_service = req.app.state.rag_service
        
        # Validate directory path
        if not os.path.exists(request.directory_path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory_path}")
        
        if not os.path.isdir(request.directory_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.directory_path}")
        
        # Start indexing
        logger.info(f"Starting indexing of directory: {request.directory_path}")
        
        result = await rag_service.index_directory(
            directory_path=request.directory_path,
            file_patterns=request.file_patterns
        )
        
        response = IndexStatus(
            status="completed",
            indexed_files=result["indexed_files"],
            skipped_files=result["skipped_files"],
            errors=result["errors"],
            total_documents=result["total_documents"]
        )
        
        logger.info(f"Indexing completed: {response.dict()}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/index/file")
async def index_file(file_path: str, req: Request):
    """Index a single file."""
    try:
        rag_service = req.app.state.rag_service
        
        # Validate file path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
        
        # Index the file
        success = await rag_service.index_file(file_path)
        
        return {
            "status": "completed" if success else "skipped",
            "file_path": file_path,
            "indexed": success
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload", response_model=IndexStatus)
async def upload_and_index(req: Request, file: UploadFile = File(...)):
    """Upload and index a file."""
    try:
        rag_service = req.app.state.rag_service
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Index the temporary file
            success = await rag_service.index_file(tmp_file_path)
            
            response = IndexStatus(
                status="completed" if success else "skipped",
                indexed_files=1 if success else 0,
                skipped_files=0 if success else 1,
                errors=[],
                total_documents=await _get_collection_count(rag_service)
            )
            
            return response
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_file_path}: {e}")
        
    except Exception as e:
        logger.error(f"Upload and index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, req: Request):
    """Search indexed documents using RAG."""
    try:
        import time
        start_time = time.time()
        
        rag_service = req.app.state.rag_service
        
        # Perform search
        results = await rag_service.search(
            query=request.query,
            max_results=request.max_results,
            file_filter=request.file_filter,
            similarity_threshold=request.similarity_threshold
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Format results
        search_results = [
            SearchResult(
                content=result["content"],
                source=result["source"],
                similarity=result["similarity"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        response = SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            execution_time_ms=round(execution_time, 2)
        )
        
        logger.info(f"Search completed: query='{request.query}', results={len(search_results)}, time={execution_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/context")
async def get_context_for_query(query: str, req: Request, max_length: int = 2000, max_results: int = 3):
    """Get formatted context for a query."""
    try:
        rag_service = req.app.state.rag_service
        
        context = await rag_service.get_context_for_query(
            query=query,
            max_context_length=max_length,
            max_results=max_results
        )
        
        return context
        
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/stats")
async def get_collection_stats(req: Request):
    """Get statistics about the RAG collection."""
    try:
        rag_service = req.app.state.rag_service
        
        stats = await rag_service.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/collection")
async def clear_collection(req: Request):
    """Clear the entire RAG collection."""
    try:
        rag_service = req.app.state.rag_service
        
        # Clear collection by recreating it
        collection_name = rag_service.collection.name
        rag_service.client.delete_collection(collection_name)
        rag_service.collection = rag_service.client.create_collection(collection_name)
        
        # Clear indexed files tracking
        rag_service.indexed_files.clear()
        
        return {
            "status": "success",
            "message": "Collection cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Collection clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/health")
async def rag_health_check(req: Request):
    """Perform health check on RAG service."""
    try:
        rag_service = req.app.state.rag_service
        health = await rag_service.health_check()
        return health
    except Exception as e:
        logger.error(f"RAG health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_collection_count(rag_service) -> int:
    """Get the current count of documents in the collection."""
    try:
        return rag_service.collection.count()
    except Exception:
        return 0
