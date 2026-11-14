import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import PureWindowsPath
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from groq import APIError
from openai import APIError as OpenAIAPIError

from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    DocumentInfo,
    EvaluationRequest,
)
from app.services.document_processing import ChunkingStrategy, chunk_text, extract_text

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@router.post("/evaluation/run")
def run_evaluation(payload: EvaluationRequest, request: Request) -> dict:
    try:
        return request.app.state.evaluation.run(
            payload.framework,
            [case.model_dump() for case in payload.cases],
        )
    except (RuntimeError, ModuleNotFoundError) as error:
        logger.exception("Chatbot evaluation could not be completed")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (APIError, OpenAIAPIError) as error:
        logger.exception("Evaluation judge provider request failed")
        raise HTTPException(
            status_code=502,
            detail="The configured Groq provider request failed during evaluation.",
        ) from error


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[dict]:
    return request.app.state.vector_store.list_documents()


@router.get("/documents/chunking-options")
def chunking_options(request: Request) -> dict[str, list[str]]:
    settings = request.app.state.settings
    return {
        "embedding_models": [settings.huggingface_model_id],
        "llm_models": [settings.grok_model],
    }


@router.post("/documents/upload", response_model=DocumentInfo, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(default="basic"),
    chunk_size: int | None = Form(default=None, gt=0),
    chunk_overlap: int | None = Form(default=None, ge=0),
    min_chunk_size: int = Form(default=1, gt=0),
    similarity_threshold: float = Form(default=0.55, ge=0, le=1),
    max_chunk_tokens: int = Form(default=512, gt=0),
    chunking_prompt: str = Form(
        default="Prefer coherent, self-contained chunks.", max_length=4000
    ),
    preserve_structure: bool = Form(default=True),
    window_size: int | None = Form(default=None, gt=0),
    window_stride: int | None = Form(default=None, gt=0),
    embedding_model: str | None = Form(default=None),
    llm_model: str | None = Form(default=None),
) -> dict:
    try:
        settings = request.app.state.settings
        selected_embedding_model = embedding_model or settings.huggingface_model_id
        selected_llm_model = llm_model or settings.grok_model
        if selected_embedding_model != settings.huggingface_model_id:
            raise ValueError("Only the configured embedding model is available.")
        if selected_llm_model != settings.grok_model:
            raise ValueError("Only the configured LLM is available.")
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
        resolved_chunk_size = (
            settings.chunk_size if chunk_size is None else chunk_size
        )
        resolved_chunk_overlap = (
            settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        if chunking_strategy == "sliding_window":
            resolved_window_size = window_size or resolved_chunk_size
            resolved_window_stride = (
                window_stride
                if window_stride is not None
                else resolved_window_size - resolved_chunk_overlap
            )
            resolved_chunk_size = resolved_window_size
            resolved_chunk_overlap = resolved_window_size - resolved_window_stride
        else:
            resolved_window_size = window_size
            resolved_window_stride = window_stride
        def encode_texts(texts: list[str]) -> Any:
            return request.app.state.vector_store.model.encode(
                texts, convert_to_numpy=True
            )

        def generate_chunks(messages: list[dict[str, str]]) -> str:
            return request.app.state.grok.generate(
                messages, model=selected_llm_model
            )

        embed_texts: Callable[[list[str]], Any] | None = (
            encode_texts if chunking_strategy == "semantic" else None
        )
        generate: Callable[[list[dict[str, str]]], str] | None = (
            generate_chunks if chunking_strategy == "agentic" else None
        )
        chunks = chunk_text(
            text,
            resolved_chunk_size,
            resolved_chunk_overlap,
            document_id,
            filename,
            datetime.now(timezone.utc).isoformat(),
            strategy=chunking_strategy,
            embed_texts=embed_texts,
            generate=generate,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold,
            max_chunk_tokens=max_chunk_tokens,
            prompt=chunking_prompt,
            preserve_structure=preserve_structure,
            window_size=resolved_window_size,
            window_stride=resolved_window_stride,
            llm_model=selected_llm_model,
        )
        request.app.state.vector_store.add_chunks(chunks)
    except HTTPException:
        raise
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        logger.exception("Document could not be indexed")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except APIError as error:
        logger.exception("Agentic chunking provider request failed")
        raise HTTPException(
            status_code=502, detail="The Groq provider request failed during chunking."
        ) from error
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
        started = time.perf_counter()
        answer, sources = request.app.state.rag.answer(payload.question, history)
        generation_latency_ms = (time.perf_counter() - started) * 1000
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
        generation_latency_ms=generation_latency_ms,
        sources=[
            ChatSource(
                filename=item["filename"],
                chunk=item["chunk"],
                excerpt=item["text"][:500],
            )
            for item in sources
        ],
    )

