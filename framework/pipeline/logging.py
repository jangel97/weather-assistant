"""Structured JSON logging for the agent pipeline.

Replaces free-form f-string logs with machine-parseable JSON lines.
Each log entry has a consistent schema with standard fields (ts, level,
logger, event) plus event-specific fields.

Usage::

    from framework.pipeline.logging import log_event, setup_logging

    setup_logging()  # call once at startup (main.py)

    # In request handlers:
    log_event(logger, logging.INFO, request_id=req_id,
              event="tool_executed", tool="search_documents", duration_ms=142)

Query examples with jq::

    # All events for a request
    jq 'select(.request_id == "abc12345")' agent.log

    # Slowest tool calls
    jq 'select(.event == "tool_executed") | {tool, duration_ms}' agent.log

    # Failed requests
    jq 'select(.event == "request_failed")' agent.log
"""

import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    If the record has a ``structured`` attribute (set via ``extra``),
    those fields are merged into the JSON entry.  Otherwise, the
    plain message is included under a ``message`` key — this handles
    logs from third-party libraries (uvicorn, httpx, etc.) that
    don't use ``log_event()``.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": (
                datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
            ),
            "level": record.levelname,
            "logger": record.name,
        }
        if hasattr(record, "structured"):
            entry.update(record.structured)
        else:
            entry["message"] = record.getMessage()
        return json.dumps(entry, default=str)


def setup_logging() -> None:
    """Configure the root logger with JSON output to stdout.

    Also forces uvicorn's loggers to propagate through the root logger
    so that *all* output is valid JSON lines (no mixed plain-text).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Force uvicorn loggers to use our formatter via propagation.
    # Uvicorn sets propagate=False and adds its own handlers; we undo that.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


def log_event(
    logger: logging.Logger,
    level: int,
    request_id: str | None = None,
    stream: bool = False,
    **fields,
) -> None:
    """Emit a structured log event.

    All keyword arguments become fields in the JSON entry.  The
    ``event`` field should always be provided to identify the log type.

    Parameters
    ----------
    logger:
        The module-level logger to emit through.
    level:
        Logging level (e.g. ``logging.INFO``).
    request_id:
        Optional request correlation ID.
    stream:
        Whether this is a streaming request.
    **fields:
        Arbitrary structured fields (event, tool, duration_ms, etc.).
    """
    structured = dict(fields)
    if request_id:
        structured["request_id"] = request_id
    if stream:
        structured["stream"] = True
    logger.log(level, "", extra={"structured": structured})
