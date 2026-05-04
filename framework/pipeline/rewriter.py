"""Two-layer follow-up question rewriting.

When the user sends a follow-up like "what about X?", the routing
layers (which don't see conversation history) can't resolve the reference.
This module uses two focused LLM calls to rewrite follow-ups:

Layer 1 (context extraction): Extracts key entities from the last
assistant message into JSON.

Layer 2 (rewrite with context): Given the extracted JSON + the user's
follow-up, produces a standalone question the routing layers can handle.

This two-layer approach is more reliable than a single-pass rewriter
because each layer has a focused task rather than simultaneously scanning
long responses and rewriting.
"""

import json
import logging
import re
import time
from typing import Dict, Optional, Tuple

from framework.clients import BaseLLMClient
from framework.pipeline_config import PipelineConfig
from framework.models import ChatRequest
from framework.pipeline.logging import log_event

logger = logging.getLogger(__name__)

_IMPERATIVE_STARTERS = (
    "show", "list", "tell", "get", "find", "give", "display", "describe",
    "compare", "count",
)


def _is_valid_rewrite(text: str) -> bool:
    """Check if rewritten text looks like a valid standalone question or request."""
    if text.endswith("?"):
        return True
    words = text.split()
    first_word = words[0].lower() if words else ""
    return first_word in _IMPERATIVE_STARTERS


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_last_messages(history) -> Tuple[Optional[str], Optional[str]]:
    """Return (last_user_content, last_assistant_content) from history."""
    last_user = None
    last_assistant = None
    for msg in reversed(history):
        if msg.role == "assistant" and last_assistant is None:
            last_assistant = msg.content
        elif msg.role == "user" and last_user is None:
            last_user = msg.content
        if last_user is not None and last_assistant is not None:
            break
    return last_user, last_assistant


def _has_entities(context: dict) -> bool:
    """Return True if the extracted context has at least one non-null entity."""
    return any(v is not None for v in context.values())




async def _extract_context(
    llm: BaseLLMClient,
    assistant_content: str,
    trace: Dict,
    config: PipelineConfig,
    request_id: str | None = None,
    stream: bool = False,
) -> Optional[dict]:
    """Layer 1: Extract key entities from the last assistant message."""
    extractor_cfg = config.models.extractor
    rewriter_model = config.models.rewriter.model

    messages = [
        {"role": "system", "content": config.prompts.extractor_system_prompt},
        {"role": "user", "content": f"Assistant message:\n{assistant_content}"},
    ]

    start = time.time()
    response = await llm.chat_completion(
        messages=messages,
        temperature=extractor_cfg.temperature,
        max_tokens=extractor_cfg.max_tokens,
        model=extractor_cfg.model or rewriter_model,
        response_format={"type": "json_object"},
    )
    elapsed = time.time() - start

    raw = (response.choices[0].message.content or "").strip()
    raw = _strip_think_tags(raw)

    effective_model = extractor_cfg.model or rewriter_model
    trace["extractor_ms"] = round(elapsed * 1000)
    trace["extractor_model"] = effective_model or str(llm.model)

    try:
        context = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log_event(
            logger, logging.WARNING, request_id=request_id, stream=stream,
            event="context_extraction_failed",
            raw_output=raw[:200],
            duration_ms=round(elapsed * 1000),
        )
        return None

    if not _has_entities(context):
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="context_empty",
            duration_ms=round(elapsed * 1000),
        )
        return None

    trace["extracted_context"] = context
    log_event(
        logger, logging.INFO, request_id=request_id, stream=stream,
        event="context_extracted",
        context=context,
        duration_ms=round(elapsed * 1000),
        assistant_content_chars=len(assistant_content),
    )
    return context


async def _rewrite_with_context(
    llm: BaseLLMClient,
    context: dict,
    current_message: str,
    previous_user_message: str,
    trace: Dict,
    config: PipelineConfig,
    request_id: str | None = None,
    stream: bool = False,
) -> Optional[str]:
    """Layer 2: Rewrite the follow-up using extracted context."""
    rewriter_cfg = config.models.rewriter

    context_json = json.dumps(
        {k: v for k, v in context.items() if v is not None},
        ensure_ascii=False,
    )
    user_content = (
        f"Context: {context_json}\n"
        f'User asked before: "{previous_user_message}"\n'
        f'Follow-up: "{current_message}"'
    )

    messages = [
        {"role": "system", "content": config.prompts.rewriter_system_prompt},
        {"role": "user", "content": user_content},
    ]

    start = time.time()
    response = await llm.chat_completion(
        messages=messages,
        temperature=rewriter_cfg.temperature,
        max_tokens=rewriter_cfg.max_tokens,
        model=rewriter_cfg.model,
    )
    elapsed = time.time() - start

    rewritten = (response.choices[0].message.content or "").strip()
    rewritten = _strip_think_tags(rewritten)
    rewritten = rewritten.strip("\"'")

    trace["rewriter_ms"] = round(elapsed * 1000)
    trace["rewriter_model"] = rewriter_cfg.model or str(llm.model)

    if rewritten.upper() == "PASS_THROUGH":
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="rewrite_pass_through",
            original=current_message,
            duration_ms=round(elapsed * 1000),
        )
        return None

    if (
        rewritten
        and rewritten != current_message
        and _is_valid_rewrite(rewritten)
        and len(rewritten) < 500
    ):
        return rewritten

    if rewritten and not _is_valid_rewrite(rewritten):
        log_event(
            logger, logging.WARNING, request_id=request_id, stream=stream,
            event="rewrite_discarded",
            output=rewritten[:200],
            duration_ms=round(elapsed * 1000),
        )
    return None


async def rewrite_question(
    llm: BaseLLMClient,
    request: ChatRequest,
    trace: Dict,
    config: PipelineConfig,
    request_id: str | None = None,
    stream: bool = False,
) -> str:
    """Rewrite a follow-up question into a standalone question.

    Uses a two-layer approach:
    1. Extract entities from the last assistant message (JSON output).
    2. Rewrite the follow-up using extracted entities (plain text output).

    Returns the original message unchanged if there is no history or
    no entities to resolve.
    """
    if not request.history:
        return request.message

    last_user, last_assistant = _get_last_messages(request.history)
    if not last_assistant:
        return request.message

    context = await _extract_context(
        llm, last_assistant, trace, config,
        request_id=request_id, stream=stream,
    )

    if context is None:
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="question_unchanged",
            reason="no_entities",
            duration_ms=trace.get("extractor_ms", 0),
        )
        return request.message

    if config.context_sanitizer:
        context = config.context_sanitizer(context, request.message)
        if not _has_entities(context):
            log_event(
                logger, logging.INFO, request_id=request_id, stream=stream,
                event="question_unchanged",
                reason="context_sanitized_empty",
                duration_ms=trace.get("extractor_ms", 0),
            )
            return request.message

    previous_question = last_user or request.message
    rewritten = await _rewrite_with_context(
        llm, context, request.message, previous_question, trace, config,
        request_id=request_id, stream=stream,
    )

    if rewritten is None:
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="question_unchanged",
            reason="rewrite_failed",
            duration_ms=(
                trace.get("extractor_ms", 0) + trace.get("rewriter_ms", 0)
            ),
        )
        return request.message

    total_ms = trace.get("extractor_ms", 0) + trace.get("rewriter_ms", 0)
    log_event(
        logger, logging.INFO, request_id=request_id, stream=stream,
        event="question_rewritten",
        original=request.message,
        rewritten=rewritten,
        context=context,
        duration_ms=total_ms,
    )
    trace["rewritten_question"] = rewritten
    return rewritten
