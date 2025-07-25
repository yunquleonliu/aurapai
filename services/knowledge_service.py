"""
Knowledge Service for Aura-PAI Platform
=======================================

This service manages the lifecycle of documents in the user's private
knowledge base. It handles file ingestion, processing, embedding,
and storage in a ChromaDB vector store for Retrieval-Augmented Generation (RAG).
"""

import logging
import os
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
from typing import List

# Corrected imports for LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.config import settings

logger = logging.getLogger(__name__)

class KnowledgeService:
    """Manages the ingestion and retrieval of documents for the RAG system."""

    def __init__(self):
        """Initializes the KnowledgeService, setting up the vector store connection."""
        try:
            # Initialize the embedding model
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'} # Or 'cuda' if GPU is available
            )

            # Initialize the ChromaDB client
            self.vector_store = Chroma(
                collection_name=settings.CHROMADB_COLLECTION_NAME,
                embedding_function=self.embedding_function,
                persist_directory=settings.CHROMADB_PATH
            )
            logger.info(f"KnowledgeService initialized. Connected to ChromaDB collection '{settings.CHROMADB_COLLECTION_NAME}'.")
        except Exception as e:
            logger.critical(f"Failed to initialize KnowledgeService: {e}", exc_info=True)
            # This is a critical failure, the application might not be able to function correctly
            raise

    def get_retriever(self, search_kwargs={"k": 3}):
        """
        Returns a retriever instance for the vector store.

        Args:
            search_kwargs (dict): A dictionary of arguments to pass to the retriever's search function.
                                  For example, `{"k": 3}` to retrieve the top 3 most similar documents.

        Returns:
            A LangChain retriever instance.
        """
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


    async def add_document_from_path_stream(self, file_path: str, filename: str):
        """
        Async generator version for streaming document processing progress.
        Yields status updates at each major step.
        """
        from fastapi import status as fastapi_status
        from pathlib import Path
        import traceback

        file_extension = Path(filename).suffix.lower()
        if file_extension not in settings.SUPPORTED_FILE_TYPES:
            yield f"Error: File type '{file_extension}' is not supported."
            return

        temp_file_path = Path(file_path)
        try:
            yield f"Loading document: {filename}\n"
            # Load the document based on its file type
            if file_extension == ".pdf":
                loader = PyPDFLoader(str(temp_file_path))
            else:
                loader = TextLoader(str(temp_file_path), encoding="utf-8")
            documents = loader.load()
            yield f"Loaded document. Splitting into chunks...\n"

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            yield f"Split document into {len(chunks)} chunks.\n"

            if not chunks:
                yield f"Error: No content could be extracted from '{filename}'. The file might be empty or in an unsupported format.\n"
                return

            # Add metadata and add to vector store
            for i, chunk in enumerate(chunks, 1):
                chunk.metadata["source"] = filename
                self.vector_store.add_documents([chunk])
                yield f"Embedded chunk {i}/{len(chunks)}\n"

            yield f"Successfully added {len(chunks)} chunks from '{filename}' to the knowledge base.\n"
        except Exception as e:
            tb = traceback.format_exc()
            yield f"Error: Failed to process and add document '{filename}': {e}\n{tb}\n"
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                yield f"Removed temporary file: {temp_file_path}\n"

    def list_documents(self) -> List[dict]:
        """
        Lists all unique documents currently in the knowledge base.

        This is done by querying the metadata of the stored vectors.
        """
        try:
            # The `get()` method without IDs or where_document returns all entries' metadata
            results = self.vector_store.get()
            
            # The metadata is in a list of dictionaries. We want to find unique sources.
            if not results or not results.get('metadatas'):
                return []

            unique_sources = {meta['source'] for meta in results['metadatas'] if 'source' in meta}
            
            # We can enhance this to return more info if needed, like chunk count per doc
            return [{"filename": source, "status": "available"} for source in sorted(list(unique_sources))]
        except Exception as e:
            logger.error(f"Failed to list documents from ChromaDB: {e}", exc_info=True)
            return []

    def delete_document(self, document_filename: str) -> dict:
        """
        Deletes a document and all its associated chunks from the knowledge base.

        Args:
            document_filename (str): The filename of the document to delete.

        Returns:
            A dictionary with the status of the operation.
        """
        try:
            # To delete in Chroma, you need the IDs of the vectors.
            # First, we find all vectors that have the specified source filename in their metadata.
            results = self.vector_store.get(where={"source": document_filename})
            
            ids_to_delete = results.get('ids')

            if not ids_to_delete:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{document_filename}' not found in the knowledge base."
                )

            # Now, delete the vectors using their IDs
            self.vector_store.delete(ids=ids_to_delete)
            logger.info(f"Successfully deleted {len(ids_to_delete)} chunks for document '{document_filename}'.")

            return {
                "filename": document_filename,
                "status": "deleted",
                "chunks_removed": len(ids_to_delete)
            }
        except HTTPException:
            raise # Re-raise HTTPException to preserve status code and detail
        except Exception as e:
            logger.error(f"Failed to delete document '{document_filename}': {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while deleting the document: {str(e)}"
            )
