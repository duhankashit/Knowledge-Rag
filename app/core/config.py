from functools import lru_cache
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Knowledge Assistant"
    api_prefix: str = "/api/v1"
    grok_api_key: str = ""
    grok_base_url: str = "https://api.groq.com"
    grok_model: str = "qwen/qwen3.8-27b"
    grok_fallback_model: str = "openai/gpt-oss-20b"
    grok_input_cost_per_million: float | None = Field(default=None, ge=0)
    grok_output_cost_per_million: float | None = Field(default=None, ge=0)
    huggingface_model_url: str = "https://huggingface.co/BAAI/bge-small-en-v1.5"
    huggingface_endpoint: str = "https://huggingface.co"
    hf_token: str = ""
    data_dir: Path = Path("data")
    chunk_size: int = Field(default=900, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        model_url = urlparse(self.huggingface_model_url)
        if model_url.scheme != "https" or not model_url.netloc:
            raise ValueError("HUGGINGFACE_MODEL_URL must be an HTTPS URL.")
        endpoint = urlparse(self.huggingface_endpoint)
        if endpoint.scheme != "https" or not endpoint.netloc:
            raise ValueError("HUGGINGFACE_ENDPOINT must be an HTTPS URL.")
        return self

    @property
    def huggingface_model_id(self) -> str:
        parts = urlparse(self.huggingface_model_url).path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(
                "HUGGINGFACE_MODEL_URL must include an organization and model name."
            )
        return "/".join(parts[:2])

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"


@lru_cache
def get_settings() -> Settings:
    return Settings()
