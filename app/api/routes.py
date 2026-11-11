import logging
import uuid
from datetime import datetime, timezone
from pathlib import PureWindowsPath

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from groq import APIError

from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    DocumentInfo,
)
from app.services.document_processing import chunk_text, extract_text

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[dict]:
    return request.app.state.vector_store.list_documents()


@router.post("/documents/upload", response_model=DocumentInfo, status_code=201)
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> dict:
    try:
        settings = request.app.state.settings
        filename = PureWindowsPath((file.filename or "").replace("/", "\\")).name
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if not filename or extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=f"Allowed file types: {', '.join(settings.allowed_extensions)}.",
            )
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
            )
        text = extract_text(filename, content)
        document_id = str(uuid.uuid4())
        chunks = chunk_text(
            text,
            settings.chunk_size,
            settings.chunk_overlap,
            document_id,
            filename,
            datetime.now(timezone.utc).isoformat(),
        )
        request.app.state.vector_store.add_chunks(chunks)
    except HTTPException:
        raise
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        logger.exception("Document could not be indexed")
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        await file.close()
    return next(
        document
        for document in request.app.state.vector_store.list_documents()
        if document["document_id"] == document_id
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request) -> None:
    if not request.app.state.vector_store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document was not found.")


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        history = [message.model_dump() for message in payload.history]
        answer, sources = request.app.state.rag.answer(payload.question, history)
    except RuntimeError as error:
        logger.exception("Chat request could not be completed")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except APIError as error:
        logger.exception("Groq request failed")
        raise HTTPException(
            status_code=502, detail="The Groq provider request failed."
        ) from error
    return ChatResponse(
        answer=answer,
        sources=[
            ChatSource(
                filename=item["filename"],
                chunk=item["chunk"],
                excerpt=item["text"][:500],
            )
            for item in sources
        ],
    )

