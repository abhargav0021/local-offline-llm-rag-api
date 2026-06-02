from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = None
    options: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool
    raw: dict[str, Any]


class EmbedRequest(BaseModel):
    input: str | list[str] = Field(..., description="Text or list of texts to embed.")
    model: str | None = None
    options: dict[str, Any] | None = None


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    raw: dict[str, Any]


class ModelInfo(BaseModel):
    name: str
    modified_at: str | None = None
    size: int | None = None
    digest: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class RagIngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    collection: str


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    model: str | None = None
    options: dict[str, Any] | None = None


class RagContextChunk(BaseModel):
    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any]


class RagQueryResponse(BaseModel):
    answer: str
    model: str
    context: list[RagContextChunk]
    raw: dict[str, Any]
