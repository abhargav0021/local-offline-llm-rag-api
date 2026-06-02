from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_ollama_client, get_rag_pipeline, get_vector_store
from app.ollama_client import OllamaError
from app.rag_pipeline import RagError
from app.schemas import RagContextChunk


class HealthyOllama:
    async def health(self) -> None:
        return None

    async def models(self) -> dict:
        return {"models": []}


class DownOllama:
    async def health(self) -> None:
        raise OllamaError("Unable to reach Ollama at http://ollama:11434")

    async def models(self) -> dict:
        raise OllamaError("Unable to reach Ollama at http://ollama:11434")


class HealthyVectorStore:
    def health(self) -> None:
        return None


class FakeRagPipeline:
    def __init__(self) -> None:
        self.vector_store = SimpleNamespace(collection_name="test_documents")

    async def ingest(self, filename: str, content: bytes, content_type: str | None) -> int:
        assert filename == "sample.txt"
        assert content == b"local rag notes"
        assert content_type == "text/plain"
        return 1

    async def query(self, request):
        return (
            "Use Ollama locally and store embeddings in ChromaDB.",
            "llama3.2",
            [
                RagContextChunk(
                    id="chunk-1",
                    text="Ollama runs local models. ChromaDB stores local embeddings.",
                    score=0.12,
                    metadata={"filename": "sample.txt", "chunk_index": 0},
                )
            ],
            {"message": {"content": "Use Ollama locally and store embeddings in ChromaDB."}},
        )


class FailingRagPipeline(FakeRagPipeline):
    async def query(self, request):
        raise RagError("Ollama is unavailable")


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_ollama_client] = lambda: HealthyOllama()
    app.dependency_overrides[get_vector_store] = lambda: HealthyVectorStore()
    app.dependency_overrides[get_rag_pipeline] = lambda: FakeRagPipeline()
    return TestClient(app)


def test_health_check_reports_local_services(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"
    assert response.json() == {
        "status": "ok",
        "offline_mode": False,
        "services": {
            "ollama": {"status": "ok"},
            "chroma": {"status": "ok"},
        },
    }


def test_rag_ingest_uploads_file_and_returns_chunk_count(client: TestClient):
    response = client.post(
        "/rag/ingest",
        files={"file": ("sample.txt", b"local rag notes", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.txt",
        "chunks_stored": 1,
        "collection": "test_documents",
    }


def test_rag_query_returns_answer_and_context(client: TestClient):
    response = client.post("/rag/query", json={"question": "How does the local demo work?", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Use Ollama locally and store embeddings in ChromaDB."
    assert payload["model"] == "llama3.2"
    assert payload["context"][0]["metadata"]["filename"] == "sample.txt"


def test_bad_input_empty_upload_returns_400(client: TestClient):
    response = client.post(
        "/rag/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_ollama_down_returns_clean_503():
    app.dependency_overrides[get_ollama_client] = lambda: DownOllama()
    app.dependency_overrides[get_vector_store] = lambda: HealthyVectorStore()

    with TestClient(app) as test_client:
        response = test_client.get("/models")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["message"] == "Ollama is unavailable. Make sure the local Ollama service is running and reachable."


def test_rag_query_ollama_failure_returns_503():
    app.dependency_overrides[get_rag_pipeline] = lambda: FailingRagPipeline()

    with TestClient(app) as test_client:
        response = test_client.post("/rag/query", json={"question": "What happens if Ollama is down?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama is unavailable"
