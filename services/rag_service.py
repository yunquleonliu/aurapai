"""
RAG Service for Auro-PAI Platform
=================================

Retrieval-Augmented Generation service using ChromaDB for vector storage
and retrieval of relevant context from local codebase and documents.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import aiofiles

from core.config import settings

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a document chunk for RAG."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"{self.metadata.get('source', 'unknown')}_{content_hash[:8]}"


class RAGService:
    """Manages retrieval-augmented generation using ChromaDB."""
    
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.indexed_files: Dict[str, str] = {}  # file_path -> hash
        
    async def initialize(self):
        """Initialize the RAG service."""
        logger.info("Initializing RAG Service...")
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT,
                settings=ChromaSettings(allow_reset=True)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(settings.CHROMADB_COLLECTION_NAME)
                logger.info(f"Using existing collection: {settings.CHROMADB_COLLECTION_NAME}")
            except Exception:
                self.collection = self.client.create_collection(settings.CHROMADB_COLLECTION_NAME)
                logger.info(f"Created new collection: {settings.CHROMADB_COLLECTION_NAME}")
                
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            logger.info("Falling back to in-memory ChromaDB")
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(settings.CHROMADB_COLLECTION_NAME)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        logger.info("RAG Service initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources."""
        # ChromaDB client cleanup is handled automatically
        logger.info("RAG Service cleaned up")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on RAG service."""
        try:
            if self.collection:
                count = self.collection.count()
                return {
                    "status": "healthy",
                    "collection_name": settings.CHROMADB_COLLECTION_NAME,
                    "document_count": count,
                    "embedding_model": settings.EMBEDDING_MODEL
                }
            else:
                return {"status": "unhealthy", "error": "Collection not initialized"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def index_directory(self, directory_path: str, file_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Index all supported files in a directory."""
        logger.info(f"Indexing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        indexed_count = 0
        skipped_count = 0
        errors = []
        
        for root, dirs, files in os.walk(directory_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'dist', 'build']]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()
                
                # Check if file type is supported
                if file_ext not in settings.SUPPORTED_FILE_TYPES:
                    continue
                
                try:
                    success = await self.index_file(file_path)
                    if success:
                        indexed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    errors.append(f"{file_path}: {str(e)}")
                    logger.error(f"Error indexing {file_path}: {e}")
        
        result = {
            "indexed_files": indexed_count,
            "skipped_files": skipped_count,
            "errors": errors,
            "total_documents": self.collection.count() if self.collection else 0
        }
        
        logger.info(f"Directory indexing complete: {result}")
        return result
    
    async def index_file(self, file_path: str) -> bool:
        """Index a single file."""
        try:
            # Check if file has changed
            current_hash = await self._get_file_hash(file_path)
            if file_path in self.indexed_files and self.indexed_files[file_path] == current_hash:
                return False  # File unchanged, skip
            
            # Read file content
            content = await self._read_file(file_path)
            if not content.strip():
                return False  # Empty file, skip
            
            # Create chunks
            chunks = self._create_chunks(content, file_path)
            if not chunks:
                return False
            
            # Remove existing chunks for this file
            await self._remove_file_chunks(file_path)
            
            # Generate embeddings and store
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk.content).tolist()
                
                self.collection.add(
                    ids=[chunk.id],
                    embeddings=[embedding],
                    documents=[chunk.content],
                    metadatas=[chunk.metadata]
                )
            
            # Update indexed files record
            self.indexed_files[file_path] = current_hash
            
            logger.debug(f"Indexed file: {file_path} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            raise
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        file_filter: Optional[str] = None,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare search filters
            where_filter = None
            if file_filter:
                where_filter = {"source": {"$regex": file_filter}}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_filter
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'][0] else 1.0
                    
                    # Convert distance to similarity score
                    similarity = 1.0 - distance
                    
                    if similarity >= similarity_threshold:
                        search_results.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity": similarity,
                            "source": metadata.get("source", "unknown")
                        })
            
            logger.debug(f"Search query: '{query}' returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return []
    
    async def get_context_for_query(
        self,
        query: str,
        max_context_length: int = 2000,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """Get formatted context for a query."""
        search_results = await self.search(query, max_results)
        
        context_parts = []
        total_length = 0
        
        for result in search_results:
            content = result["content"]
            source = result["source"]
            
            # Format context entry
            context_entry = f"Source: {source}\n{content}\n"
            
            if total_length + len(context_entry) <= max_context_length:
                context_parts.append(context_entry)
                total_length += len(context_entry)
            else:
                break
        
        return {
            "context": "\n---\n".join(context_parts),
            "sources": [r["source"] for r in search_results],
            "total_results": len(search_results)
        }
    
    def _create_chunks(self, content: str, file_path: str) -> List[DocumentChunk]:
        """Create chunks from file content."""
        chunks = []
        
        # Simple chunking strategy - split by lines and group
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > settings.CHUNK_SIZE and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                metadata = {
                    "source": file_path,
                    "type": "code" if self._is_code_file(file_path) else "text",
                    "chunk_index": len(chunks)
                }
                chunks.append(DocumentChunk(chunk_content, metadata))
                
                # Start new chunk with overlap
                overlap_lines = max(1, settings.CHUNK_OVERLAP // 50)  # Approximate lines for overlap
                current_chunk = current_chunk[-overlap_lines:] if len(current_chunk) > overlap_lines else []
                current_size = sum(len(l) + 1 for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add final chunk if not empty
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            metadata = {
                "source": file_path,
                "type": "code" if self._is_code_file(file_path) else "text",
                "chunk_index": len(chunks)
            }
            chunks.append(DocumentChunk(chunk_content, metadata))
        
        return chunks
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs', '.php'}
        return Path(file_path).suffix.lower() in code_extensions
    
    async def _read_file(self, file_path: str) -> str:
        """Read file content asynchronously."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                return await f.read()
    
    async def _get_file_hash(self, file_path: str) -> str:
        """Get file content hash for change detection."""
        content = await self._read_file(file_path)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _remove_file_chunks(self, file_path: str):
        """Remove existing chunks for a file."""
        try:
            # Query for existing chunks from this file
            existing = self.collection.get(where={"source": file_path})
            if existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except Exception as e:
            logger.debug(f"Error removing existing chunks for {file_path}: {e}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze sources
            sample = self.collection.get(limit=min(100, count))
            sources = set()
            file_types = {}
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
                        ext = Path(metadata['source']).suffix.lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "file_types": file_types,
                "collection_name": settings.CHROMADB_COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
