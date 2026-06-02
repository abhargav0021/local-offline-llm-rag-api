from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=not settings.disable_chroma_telemetry),
        )
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection)

    def upsert_chunks(
        self,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not chunks:
            return

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": documents[index],
                    "metadata": metadatas[index] or {},
                    "score": distances[index] if index < len(distances) else None,
                }
            )
        return chunks

    def health(self) -> None:
        self.collection.count()

    @property
    def collection_name(self) -> str:
        return self.settings.chroma_collection
