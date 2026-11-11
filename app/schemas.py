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
