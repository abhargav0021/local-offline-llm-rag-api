import hashlib

from app.config import Settings
from app.document_loader import extract_text
from app.ollama_client import OllamaClient, OllamaError
from app.schemas import ChatRequest, EmbedRequest, RagContextChunk, RagQueryRequest
from app.text_splitter import chunk_text
from app.vector_store import VectorStore


class RagError(RuntimeError):
    pass


class RagPipeline:
    def __init__(self, settings: Settings, ollama_client: OllamaClient, vector_store: VectorStore) -> None:
        self.settings = settings
        self.ollama_client = ollama_client
        self.vector_store = vector_store

    async def ingest(self, filename: str, content: bytes, content_type: str | None) -> int:
        text = extract_text(filename, content)
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise RagError("No text could be extracted from the uploaded file.")

        embeddings = await self._embed(chunks)
        document_hash = hashlib.sha256(content).hexdigest()
        ids = [f"{document_hash}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "filename": filename,
                "content_type": content_type or "application/octet-stream",
                "source_hash": document_hash,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.vector_store.upsert_chunks(ids=ids, chunks=chunks, embeddings=embeddings, metadatas=metadatas)
        return len(chunks)

    async def query(self, request: RagQueryRequest) -> tuple[str, str, list[RagContextChunk], dict]:
        top_k = request.top_k or self.settings.rag_top_k
        question_embedding = (await self._embed(request.question))[0]
        retrieved = self.vector_store.query(question_embedding, top_k=top_k)
        context_chunks = [RagContextChunk(**chunk) for chunk in retrieved]

        context_text = self._format_context(context_chunks)
        chat_request = ChatRequest(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided local document context. "
                        "If the context does not contain the answer, say you do not know."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {request.question}",
                },
            ],
            options=request.options,
        )
        result = await self.ollama_client.chat(chat_request)
        answer = result.get("message", {}).get("content", "")
        model = result.get("model", request.model or self.settings.ollama_chat_model)
        return answer, model, context_chunks, result

    async def _embed(self, input_text: str | list[str]) -> list[list[float]]:
        try:
            result = await self.ollama_client.embed(EmbedRequest(input=input_text, model=self.settings.ollama_embed_model))
        except OllamaError as exc:
            raise RagError(str(exc)) from exc

        embeddings = result.get("embeddings")
        if embeddings is None and "embedding" in result:
            embeddings = [result["embedding"]]
        if not embeddings:
            raise RagError("Ollama did not return embeddings.")
        return embeddings

    @staticmethod
    def _format_context(chunks: list[RagContextChunk]) -> str:
        if not chunks:
            return "No context retrieved."

        formatted = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("filename", "unknown")
            formatted.append(f"[{index}] Source: {source}\n{chunk.text}")
        return "\n\n".join(formatted)
