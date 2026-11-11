from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.clients.grok import GrokClient
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.repositories.vector_store import VectorStore
from app.services.rag import RAGService

settings = get_settings()
configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    vector_store = VectorStore(
        settings.faiss_dir,
        settings.huggingface_model_id,
        settings.hf_token,
        settings.huggingface_endpoint,
    )
    grok = GrokClient(
        settings.grok_api_key,
        settings.grok_base_url,
        settings.grok_model,
        settings.grok_fallback_model,
    )
    application.state.settings = settings
    application.state.vector_store = vector_store
    application.state.grok = grok
    application.state.rag = RAGService(
        vector_store, grok, settings.retrieval_top_k
    )
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": settings.app_name, "docs": "/docs"}
