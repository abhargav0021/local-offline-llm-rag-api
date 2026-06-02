import logging
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status

from app.config import Settings, get_settings
from app.document_loader import DocumentLoadError
from app.logging_config import configure_logging, request_id_context
from app.ollama_client import OllamaClient, OllamaError
from app.rag_pipeline import RagError, RagPipeline
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    ModelsResponse,
    RagIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from app.vector_store import VectorStore

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local LLM Inference API",
    description="A FastAPI wrapper around Ollama for local chat and embeddings.",
    version="0.1.0",
)


def get_ollama_client(settings: Settings = Depends(get_settings)) -> OllamaClient:
    return OllamaClient(settings)


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStore:
    return VectorStore(settings)


def get_rag_pipeline(
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClient = Depends(get_ollama_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> RagPipeline:
    return RagPipeline(settings=settings, ollama_client=ollama_client, vector_store=vector_store)


def ollama_unavailable_response(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": "Ollama is unavailable. Make sure the local Ollama service is running and reachable.",
            "error": str(exc),
        },
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    token = request_id_context.set(request_id)
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "request_failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else None,
                "error": str(exc),
            },
        )
        request_id_context.reset(token)
        raise

    response.headers["X-Request-ID"] = request_id
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else None,
        },
    )
    request_id_context.reset(token)
    return response


@app.get("/health")
async def health(
    settings: Settings = Depends(get_settings),
    client: OllamaClient = Depends(get_ollama_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    services = {"ollama": {"status": "ok"}, "chroma": {"status": "ok"}}

    try:
        await client.health()
    except OllamaError as exc:
        services["ollama"] = {"status": "unavailable", "message": str(exc)}

    try:
        vector_store.health()
    except Exception as exc:
        services["chroma"] = {"status": "unavailable", "message": str(exc)}

    if any(service["status"] != "ok" for service in services.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "degraded",
                "message": "One or more local services are unavailable.",
                "offline_mode": settings.offline_mode,
                "services": services,
            },
        )

    return {"status": "ok", "offline_mode": settings.offline_mode, "services": services}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, client: OllamaClient = Depends(get_ollama_client)) -> ChatResponse:
    try:
        result = await client.chat(request)
    except OllamaError as exc:
        raise ollama_unavailable_response(exc) from exc

    return ChatResponse(
        model=result.get("model", request.model or client.settings.ollama_chat_model),
        message=result["message"],
        done=result.get("done", False),
        raw=result,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, client: OllamaClient = Depends(get_ollama_client)) -> EmbedResponse:
    try:
        result = await client.embed(request)
    except OllamaError as exc:
        raise ollama_unavailable_response(exc) from exc

    embeddings = result.get("embeddings")
    if embeddings is None and "embedding" in result:
        embeddings = [result["embedding"]]

    return EmbedResponse(
        model=result.get("model", request.model or client.settings.ollama_embed_model),
        embeddings=embeddings or [],
        raw=result,
    )


@app.get("/models", response_model=ModelsResponse)
async def models(client: OllamaClient = Depends(get_ollama_client)) -> ModelsResponse:
    try:
        result = await client.models()
    except OllamaError as exc:
        raise ollama_unavailable_response(exc) from exc

    return ModelsResponse(models=result.get("models", []))


@app.post("/rag/ingest", response_model=RagIngestResponse)
async def rag_ingest(
    file: UploadFile = File(...),
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> RagIngestResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        chunks_stored = await pipeline.ingest(
            filename=file.filename or "uploaded-file",
            content=content,
            content_type=file.content_type,
        )
    except DocumentLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RagError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return RagIngestResponse(
        filename=file.filename or "uploaded-file",
        chunks_stored=chunks_stored,
        collection=pipeline.vector_store.collection_name,
    )


@app.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(
    request: RagQueryRequest,
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> RagQueryResponse:
    try:
        answer, model, context, raw = await pipeline.query(request)
    except RagError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return RagQueryResponse(answer=answer, model=model, context=context, raw=raw)
