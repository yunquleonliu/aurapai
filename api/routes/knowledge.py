from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
import os
import shutil
from typing import List

from services.knowledge_service import KnowledgeService
from core.dependencies import get_knowledge_service
from core.config import settings

router = APIRouter(
    prefix="/api/v1/knowledge",
    tags=["Knowledge Base"],
)


# Streaming upload endpoint for local LLM interactivity
@router.post("/upload/stream", status_code=status.HTTP_202_ACCEPTED)
async def upload_document_stream(
    file: UploadFile = File(...),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Upload a document to the personal knowledge base (streaming version).
    Streams status updates as the document is processed for better interactivity with local LLM.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file name provided.")

    if not any(file.filename.lower().endswith(ext) for ext in settings.SUPPORTED_FILE_TYPES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types are: {', '.join(settings.SUPPORTED_FILE_TYPES)}"
        )

    upload_dir = "temp_uploads"
    temp_path = os.path.join(upload_dir, file.filename)
    os.makedirs(upload_dir, exist_ok=True)

    async def event_stream():
        try:
            yield f"Starting upload for: {file.filename}\n"
            # Write file in chunks and yield progress
            with open(temp_path, "wb") as buffer:
                chunk_size = 1024 * 1024  # 1MB
                total_written = 0
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    total_written += len(chunk)
                    yield f"Uploaded {total_written // 1024} KB...\n"
            yield f"File upload complete. Processing file: {file.filename}\n"
            async for status in knowledge_service.add_document_from_path_stream(temp_path, file.filename):
                yield status
        except Exception as e:
            yield f"Error: Failed to process file: {e}\n"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(event_stream(), media_type="text/plain")
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from typing import List

from services.knowledge_service import KnowledgeService
from core.dependencies import get_knowledge_service
from core.config import settings

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Upload a document to the personal knowledge base.

    The system will process the document, chunk it, create embeddings,
    and add it to the RAG vector store for future conversations.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file name provided.")

    if not any(file.filename.lower().endswith(ext) for ext in settings.SUPPORTED_FILE_TYPES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types are: {', '.join(settings.SUPPORTED_FILE_TYPES)}"
        )

    # Create upload directory if it doesn't exist
    upload_dir = "temp_uploads"
    temp_path = os.path.join(upload_dir, file.filename)

    os.makedirs(upload_dir, exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = await knowledge_service.add_document_from_path(temp_path, file.filename)
        return {"filename": file.filename, "message": "File uploaded and processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/list", summary="List Documents in Knowledge Base")
async def list_documents(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """List all documents currently available in the RAG knowledge base."""
    try:
        documents = knowledge_service.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

@router.delete("/{filename}", summary="Delete Document from Knowledge Base")
async def delete_document(
    filename: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """Delete a document and all its associated data from the knowledge base."""
    try:
        knowledge_service.delete_document(filename)
        return {"message": f"Document '{filename}' deleted successfully."}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")