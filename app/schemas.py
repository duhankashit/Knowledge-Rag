from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class ChatSource(BaseModel):
    filename: str
    chunk: int
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    generation_latency_ms: float = Field(ge=0)


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    expected_answer: str | None = Field(default=None, min_length=1, max_length=8000)
    relevant_contexts: list[str] = Field(default_factory=list, max_length=20)
    answer: str | None = Field(default=None, min_length=1, max_length=8000)
    retrieved_contexts: list[str] | None = Field(default=None, max_length=20)
    generation_latency_ms: float | None = Field(default=None, ge=0)


class EvaluationRequest(BaseModel):
    framework: Literal["ragas", "deepeval"]
    cases: list[EvaluationCase] = Field(min_length=1, max_length=20)
