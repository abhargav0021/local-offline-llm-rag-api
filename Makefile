APP_HOST ?= 0.0.0.0
APP_PORT ?= 8000
PYTEST := $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)

.PHONY: run test ingest

run:
	docker compose up --build

test:
	$(PYTEST)

ingest:
	@if [ -z "$(FILE)" ]; then echo "Usage: make ingest FILE=sample.txt"; exit 1; fi
	curl -X POST http://localhost:$(APP_PORT)/rag/ingest -F "file=@$(FILE)"
