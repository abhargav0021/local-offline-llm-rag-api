# Local Offline LLM RAG API

A FastAPI project for running local chat, embeddings, and retrieval-augmented generation with Ollama and ChromaDB. It is built for portfolio demonstrations around local inference, offline deployment, and constrained-environment AI systems.

## Demo Status

No public hosted demo is configured because this project is designed to run local model infrastructure and a local vector store. The best demo is a local Docker/API walkthrough:

- API docs: `http://localhost:8000/docs`
- Health endpoint: `GET /health`
- RAG flow: `POST /rag/ingest` followed by `POST /rag/query`

Recommended public alternative: add screenshots or a short recorded API demo instead of exposing an unauthenticated public RAG service.

## What It Does

This service wraps a local Ollama runtime with a clean HTTP API:

- `POST /chat` sends chat messages to a local Ollama chat model.
- `POST /embed` generates embeddings with a local Ollama embedding model.
- `GET /models` lists locally available Ollama models.
- `POST /rag/ingest` accepts PDF, text, and markdown uploads, chunks them, embeds them, and stores vectors in local persistent ChromaDB.
- `POST /rag/query` retrieves top matching chunks from ChromaDB and passes them to the local LLM as grounded context.
- `GET /health` checks that Ollama is reachable and ChromaDB is accessible.

The project includes Docker Compose, an offline Compose profile, structured JSON logging with request IDs, pytest coverage, and Makefile shortcuts.

## Preview

Add these screenshots before using the repository as a pinned portfolio project:

- FastAPI Swagger UI at `/docs`.
- Successful `/health` response showing Ollama and ChromaDB status.
- Example `/rag/query` response with retrieved context.

## Why Local LLMs Matter

Many useful AI workflows cannot depend on cloud APIs. Teams working in regulated, classified, disconnected, or bandwidth-constrained environments often need inference to happen inside a controlled boundary.

Local LLM deployment helps with:

- Data control: prompts, documents, and retrieved context stay on local infrastructure.
- Offline operation: models can be pre-pulled, baked into images, and transferred through approved channels.
- Lower integration risk: no external model API keys are required.
- Auditability: the application, model runtime, vector store, and data flow are visible and reviewable.
- Repeatability: the same containerized runtime can be tested before entering an air-gapped environment.

This repository demonstrates those ideas with a compact, inspectable implementation.

## Architecture

```text
                      +---------------------+
                      |      API Client     |
                      | curl / app / browser|
                      +----------+----------+
                                 |
                                 v
                      +---------------------+
                      |       FastAPI       |
                      | request IDs + logs  |
                      +------+-------+------+
                             |       |
                chat/embed   |       | RAG ingest/query
                             |       |
                             v       v
                    +------------+  +--------------------+
                    |  Ollama    |  | Document Pipeline  |
                    | local LLM  |  | parse + chunk      |
                    +-----+------+  +---------+----------+
                          |                   |
                          | embeddings        | chunks + vectors
                          v                   v
                    +-------------------------------------+
                    |      ChromaDB Persistent Store      |
                    |      local vectors + metadata       |
                    +-------------------------------------+
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and regulated-environment notes.

## Project Structure

```text
.
|-- app/
|   |-- config.py
|   |-- document_loader.py
|   |-- logging_config.py
|   |-- main.py
|   |-- ollama_client.py
|   |-- rag_pipeline.py
|   |-- schemas.py
|   |-- text_splitter.py
|   `-- vector_store.py
|-- tests/
|   `-- test_api.py
|-- .github/workflows/test.yml
|-- .env.example
|-- ARCHITECTURE.md
|-- Dockerfile
|-- Dockerfile.ollama-offline
|-- docker-compose.yml
|-- docker-compose.offline.yml
|-- Makefile
|-- pyproject.toml
`-- requirements.txt
```

## Quickstart

Prerequisites:

- Docker and Docker Compose
- Enough disk space for Ollama model weights

Create a local config file:

```bash
cp .env.example .env
```

Start the stack:

```bash
docker compose up --build
```

Or use Make:

```bash
make run
```

The first run pulls the configured Ollama models:

- `llama3.2` for chat
- `nomic-embed-text` for embeddings

API docs are available at:

```text
http://localhost:8000/docs
```

## Configuration

The app reads configuration from environment variables. Use `.env.example` as the template:

```dotenv
APP_HOST=0.0.0.0
APP_PORT=8000
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
REQUEST_TIMEOUT_SECONDS=120
CHROMA_PATH=.chroma
CHROMA_COLLECTION=local_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_TOP_K=4
OFFLINE_MODE=false
DISABLE_CHROMA_TELEMETRY=true
```

Do not commit `.env`. It is intentionally ignored.

## Example API Calls

Set a base URL:

```bash
BASE_URL=http://localhost:8000
```

Health check:

```bash
curl "$BASE_URL/health"
```

List local Ollama models:

```bash
curl "$BASE_URL/models"
```

Chat completion:

```bash
curl -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-chat-001" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain why local LLM inference matters in constrained environments."
      }
    ],
    "options": {
      "temperature": 0.2
    }
  }'
```

Generate embeddings:

```bash
curl -X POST "$BASE_URL/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Local inference keeps model calls inside the deployment boundary."
  }'
```

Ingest a document for RAG:

```bash
curl -X POST "$BASE_URL/rag/ingest" \
  -F "file=@./sample.md"
```

Or:

```bash
make ingest FILE=sample.md
```

Supported upload types:

- PDF
- `.txt`
- `.md`
- `.markdown`

Query ingested documents:

```bash
curl -X POST "$BASE_URL/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What deployment constraints does this project address?",
    "top_k": 4,
    "options": {
      "temperature": 0.1
    }
  }'
```

## Local Python Development

Install Ollama locally, then pull the default models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Create a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local non-Docker development, set:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
```

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

Run the API test suite:

```bash
python -m pytest tests -q
```

Or use the Makefile shortcut:

```bash
make test
```

The tests use FastAPI dependency overrides, so they validate request/response behavior without calling a live Ollama service or cloud API.

## Deployment

Recommended hosting model: local Docker Compose, internal server, or controlled lab environment.

This is not a good fit for Vercel, Netlify, or GitHub Pages because it requires:

- A long-running FastAPI service.
- Ollama model runtime.
- Persistent ChromaDB storage.
- Local model weights and enough disk/RAM for inference.

For a public portfolio demo, use one of these safer options:

- Recorded local demo video.
- Screenshot walkthrough.
- Temporary private Render/Railway-style deployment with authentication and no sensitive documents.

## Known Limitations

- No public live demo is available.
- First startup may take time because Ollama models must be pulled.
- Inference quality and latency depend on the local model and host machine.
- The API is not currently protected with authentication.
- Uploaded document storage is local and not multi-tenant.

## Planned Improvements

- Add authentication or API-key middleware for non-local deployments.
- Add a small checked-in sample document for demo ingestion.
- Add screenshots to the README.
- Add request/response examples for failure cases.
- Add load/performance notes for different local model sizes.

## Offline / Air-Gapped Deployment

The default Compose file includes an `ollama-init` service that pulls models at startup. That is useful for development, but offline deployments need model weights available before the environment is disconnected.

While connected to the internet, build and export offline images:

```bash
docker build -t local-llm-api:offline .

docker build \
  -f Dockerfile.ollama-offline \
  --build-arg OLLAMA_CHAT_MODEL=llama3.2 \
  --build-arg OLLAMA_EMBED_MODEL=nomic-embed-text \
  -t local-ollama:offline .

docker save local-llm-api:offline local-ollama:offline -o local-llm-offline-images.tar
```

Transfer these files through your approved process:

- `local-llm-offline-images.tar`
- `docker-compose.offline.yml`
- `.env` created from `.env.example`

On the offline machine:

```bash
docker load -i local-llm-offline-images.tar
docker compose -f docker-compose.offline.yml up
```

The offline Compose file:

- Uses prebuilt images only
- Does not pull models at startup
- Binds API and Ollama ports to `127.0.0.1`
- Sets `OFFLINE_MODE=true`
- Keeps Chroma telemetry disabled
- Uses an internal Docker network

## Public Repo Hygiene

This repository is configured so public commits should not include local secrets or heavyweight artifacts:

- `.env` and `.env.*` are ignored, except `.env.example`
- ChromaDB local data is ignored
- Ollama data directories are ignored
- Common model weight formats such as `.gguf`, `.safetensors`, `.pt`, `.pth`, `.onnx`, and `.bin` are ignored
- Offline image tarballs are ignored
- GitHub Actions runs pytest on push and pull request

Before publishing, check:

```bash
git status --short
```

Only source files, docs, config templates, and CI files should be staged.

## License

No license file is currently included. Add a license before encouraging reuse outside portfolio review.

## Contact

Maintained by [Bhargav Reddy](https://github.com/abhargav0021). Open an issue for questions or portfolio review feedback.
