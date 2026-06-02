import json
import logging
import time
from contextvars import ContextVar
from typing import Any


request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "request_id": request_id_context.get(),
        }

        for field in ("method", "path", "status_code", "duration_ms", "client", "error"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_local_llm_json", False) for handler in root.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler._local_llm_json = True  # type: ignore[attr-defined]

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
