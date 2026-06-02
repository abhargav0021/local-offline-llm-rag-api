from typing import Any

import httpx

from app.config import Settings
from app.schemas import ChatRequest, EmbedRequest


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    async def chat(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.settings.ollama_chat_model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": False,
        }
        if request.options:
            payload["options"] = request.options

        return await self._post("/api/chat", payload)

    async def embed(self, request: EmbedRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.settings.ollama_embed_model,
            "input": request.input,
        }
        if request.options:
            payload["options"] = request.options

        return await self._post("/api/embed", payload)

    async def models(self) -> dict[str, Any]:
        return await self._get("/api/tags")

    async def health(self) -> None:
        await self._get("/api/tags")

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}{path}")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(self._format_http_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(self._format_http_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc

    @staticmethod
    def _format_http_error(exc: httpx.HTTPStatusError) -> str:
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text
        return f"Ollama returned HTTP {exc.response.status_code}: {detail}"
