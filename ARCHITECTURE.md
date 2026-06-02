# Architecture

## Overview

This project is a local-only LLM and RAG demo. FastAPI exposes the public HTTP API, Ollama runs the chat and embedding models, and ChromaDB stores document vectors on local disk.

The request path is:

1. A client calls the FastAPI service.
2. `/rag/ingest` extracts text from PDF, text, or markdown uploads.
3. The service chunks the extracted text.
4. Chunks are embedded through Ollama using the local embedding model.
5. ChromaDB persists chunk text, metadata, and vectors locally.
6. `/rag/query` embeds the user question, retrieves top matching chunks from ChromaDB, and sends the retrieved context to the local Ollama chat model.

No OpenAI SDK, cloud LLM API, hosted vector database, or external inference service is used.

## Design Decisions

Ollama is used as the model runtime because it provides a simple local REST API and stores model weights on the host or inside the container image. The FastAPI app wraps that API so application callers do not need to know Ollama-specific request formats.

ChromaDB is configured as a persistent local vector store. It writes vectors and metadata to `CHROMA_PATH`, which is mounted as a Docker volume in Compose. Chroma telemetry is disabled by default with `DISABLE_CHROMA_TELEMETRY=true`.

The RAG pipeline is intentionally small and auditable. Text extraction, chunking, embedding, retrieval, and answer generation are separated into clear modules. This makes it easier to review data flow and verify that sensitive content does not leave the local runtime.

The offline deployment path uses prebuilt images:

- `local-llm-api:offline` contains the Python API and all Python dependencies.
- `local-ollama:offline` contains Ollama and pre-pulled model weights.

`docker-compose.offline.yml` does not include the model-pulling init container. It expects those images to already exist on the air-gapped machine.

## Regulated Or Classified Environments

This approach is suitable for production-style demos in regulated or classified environments because inference, embeddings, retrieval, and document storage happen inside the local deployment boundary.

Important properties:

- No cloud model calls are required.
- Uploaded documents are stored only as local ChromaDB records and Docker volume data.
- Model weights can be reviewed, approved, and imported as an image artifact.
- The offline Compose file binds published ports to `127.0.0.1`.
- The offline Compose network is marked `internal: true`, preventing containers on that network from reaching external networks through Docker.
- ChromaDB telemetry is disabled through configuration.

Operational controls still matter. In a real regulated deployment, teams should also apply host firewall rules, image signing, vulnerability scanning, audit logging, access control, and approved media-transfer procedures. This project provides the local architecture pattern, not a complete accreditation package.

## Failure Handling

The API returns clean `503` responses when Ollama is unavailable. The `/health` endpoint probes both Ollama and ChromaDB so operators can distinguish an application problem from a local runtime or storage problem.
