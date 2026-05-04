"""Reusable agent orchestrator.

Encapsulates the full pipeline: question rewriting, routing loop,
answer generation, post-processing, error handling, and tracing.
Consumers provide a ``PipelineConfig``, an LLM client, and optionally
a list of routable entities — then call ``chat()`` or ``stream()``.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List
from uuid import uuid4

from openai import APIConnectionError, APIError, APITimeoutError

from framework.clients.base import BaseLLMClient
from framework.pipeline_config import PipelineConfig
from framework.constants import LOG_ANSWER_CHARS
from framework.models import ChatRequest, ChatResponse
from framework.pipeline.logging import log_event
from framework.pipeline.messages import (
    build_action_messages,
    build_answer_messages,
)
from framework.pipeline.parsing import parse_action_response
from framework.pipeline.postprocess import (
    ThinkBlockFilter,
    clean_response,
    clean_token,
    fix_answer_links,
    llm_error_message,
)
from framework.pipeline.rewriter import rewrite_question
from framework.pipeline.routing import routing_loop, sse_event
from framework.types import RoutableEntity

logger = logging.getLogger(__name__)


@dataclass
class _RunContext:
    """Mutable state for a single request through the pipeline."""

    req_id: str = field(default_factory=lambda: uuid4().hex[:8])
    request_start: float = field(default_factory=time.time)
    tool_results: List[Dict[str, str]] = field(default_factory=list)
    tool_calls_made: List[Dict[str, str]] = field(default_factory=list)
    seen_calls: set = field(default_factory=set)
    trace: Dict[str, Any] = field(default=None)
    stream: bool = False

    def __post_init__(self):
        if self.trace is None:
            self.trace = {"request_id": self.req_id, "rounds": []}


class AgentRunner:
    """Runs the multi-layer routing pipeline for any agent implementation."""

    def __init__(
        self,
        llm: BaseLLMClient,
        config: PipelineConfig,
        entities: List[RoutableEntity] | None = None,
    ):
        self.llm = llm
        self.config = config
        self.entities = entities or []

    # -- Public API -----------------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming: run the full pipeline and return the response."""
        ctx = _RunContext()
        self._log_request(ctx, request)

        try:
            routing_request = await self._prepare(ctx, request)
            if routing_request is not None:
                await self._route(ctx, routing_request)
            answer = await self._answer(ctx, request)
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            return self._handle_api_error(ctx, exc)
        except Exception as exc:
            return self._handle_unexpected_error(ctx, exc)

        self._log_completion(ctx, answer)
        return ChatResponse(
            message=answer,
            tool_calls=ctx.tool_calls_made if ctx.tool_calls_made else None,
            trace=ctx.trace,
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Streaming: yield SSE events for the full pipeline."""
        ctx = _RunContext(stream=True)
        self._log_request(ctx, request)

        try:
            routing_request = await self._prepare(ctx, request)

            if routing_request is not None:
                async for event in self._route_streaming(ctx, routing_request):
                    yield event

            async for event in self._answer_streaming(ctx, request):
                yield event

        except (APIConnectionError, APITimeoutError, APIError) as exc:
            self._log_error(ctx, exc)
            yield sse_event(
                {"type": "error", "message": llm_error_message(exc)}
            )
        except Exception as exc:
            self._log_error(ctx, exc)
            yield sse_event(
                {
                    "type": "error",
                    "message": (
                        "An unexpected error occurred. Please try again."
                    ),
                }
            )

    # -- Pipeline steps -------------------------------------------------------

    async def _prepare(
        self, ctx: _RunContext, request: ChatRequest,
    ) -> ChatRequest | None:
        """Pre-classify and optionally rewrite the message.

        Returns the request to route, or None if routing should be skipped.
        """
        if not request.history:
            return request

        action = await self._pre_classify(ctx, request)
        if action != "tool":
            return None

        rewritten = await rewrite_question(
            self.llm, request, ctx.trace, self.config,
            request_id=ctx.req_id, stream=ctx.stream,
        )
        return request.model_copy(update={"message": rewritten})

    _PRE_CLASSIFY_HISTORY_CHARS = 1500

    async def _pre_classify(
        self, ctx: _RunContext, request: ChatRequest,
    ) -> str:
        """Run the action classifier on the raw message before rewriting.

        Includes the previous conversation turn (user question + assistant
        response) so the classifier can recognize data follow-ups like
        "what was the first?" by seeing that the previous turn returned
        database results.
        """
        action_cfg = self.config.models.action

        prev_question = None
        prev_answer = None
        for m in reversed(request.history):
            if m.role == "assistant" and prev_answer is None:
                prev_answer = m.content
            elif m.role == "user" and prev_question is None:
                prev_question = m.content
            if prev_question and prev_answer:
                break

        if prev_question or prev_answer:
            parts = []
            if prev_question:
                parts.append(f'Previous question: "{prev_question}"')
            if prev_answer:
                truncated = prev_answer[:self._PRE_CLASSIFY_HISTORY_CHARS]
                if len(prev_answer) > self._PRE_CLASSIFY_HISTORY_CHARS:
                    truncated += "\n... (truncated)"
                parts.append(f"Previous answer: {truncated}")
            parts.append(f"Current question: {request.message}")
            augmented_message = "\n".join(parts)
            classify_request = request.model_copy(
                update={"message": augmented_message},
            )
        else:
            classify_request = request

        messages = build_action_messages(
            self.config.prompts.build_action_prompt(),
            classify_request, tool_results=[],
            max_tool_result_chars=self.config.max_tool_result_chars,
        )

        start = time.time()
        response = await self.llm.chat_completion(
            messages=messages,
            temperature=action_cfg.temperature,
            max_tokens=action_cfg.max_tokens,
            model=action_cfg.model,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content or ""
        action = parse_action_response(content)

        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=ctx.stream,
            event="pre_classified",
            action=action,
            duration_ms=round(elapsed * 1000),
            model=action_cfg.model or str(self.llm.model),
        )
        ctx.trace["pre_classification"] = action
        ctx.trace["pre_classification_ms"] = round(elapsed * 1000)
        return action

    async def _route(
        self, ctx: _RunContext, request: ChatRequest,
    ) -> None:
        async for _ in routing_loop(
            self.llm, request,
            ctx.tool_results, ctx.tool_calls_made, ctx.seen_calls, ctx.trace,
            entities=self.entities, config=self.config,
            request_id=ctx.req_id, stream=ctx.stream,
        ):
            pass

    async def _route_streaming(
        self, ctx: _RunContext, request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        async for event in routing_loop(
            self.llm, request,
            ctx.tool_results, ctx.tool_calls_made, ctx.seen_calls, ctx.trace,
            entities=self.entities, config=self.config,
            request_id=ctx.req_id, stream=True, emit_events=True,
        ):
            yield event

    async def _answer(
        self, ctx: _RunContext, request: ChatRequest,
    ) -> str:
        answer_cfg = self.config.models.answer
        messages = build_answer_messages(
            request, ctx.tool_results,
            answer_prompt=self.config.prompts.build_answer_prompt(),
            system_prompt=self.config.prompts.build_system_prompt(),
        )
        source = "tool_data" if ctx.tool_results else "knowledge"
        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=ctx.stream,
            event="answer_started", source=source,
        )

        start = time.time()
        response = await self.llm.chat_completion(
            messages=messages,
            temperature=answer_cfg.temperature,
            max_tokens=answer_cfg.max_tokens,
            model=answer_cfg.model,
        )
        elapsed = time.time() - start

        answer = clean_response(response.choices[0].message.content or "")
        answer = fix_answer_links(
            answer, ctx.tool_results,
            link_fix_patterns=self.config.link_fix_patterns,
        )

        ctx.trace["answer_ms"] = round(elapsed * 1000)
        ctx.trace["answer_model"] = answer_cfg.model or str(self.llm.model)
        ctx.trace["answer_source"] = source
        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=ctx.stream,
            event="answer_completed", source=source,
            duration_ms=round(elapsed * 1000),
            model=answer_cfg.model or str(self.llm.model),
            answer_chars=len(answer),
        )
        return answer

    async def _answer_streaming(
        self, ctx: _RunContext, request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        answer_cfg = self.config.models.answer
        messages = build_answer_messages(
            request, ctx.tool_results,
            answer_prompt=self.config.prompts.build_answer_prompt(),
            system_prompt=self.config.prompts.build_system_prompt(),
        )
        source = "tool_data" if ctx.tool_results else "knowledge"
        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=True,
            event="answer_started", source=source,
        )

        answer_start = time.time()
        llm_stream = await self.llm.chat_completion_stream(
            messages=messages,
            temperature=answer_cfg.temperature,
            max_tokens=answer_cfg.max_tokens,
            model=answer_cfg.model,
        )

        think_filter = ThinkBlockFilter()
        accumulated = ""
        async for chunk in llm_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                filtered = think_filter.feed(delta.content)
                if filtered:
                    cleaned = clean_token(filtered)
                    if cleaned:
                        accumulated += cleaned
                        yield sse_event({"type": "token", "content": cleaned})

        flushed = think_filter.flush()
        if flushed:
            cleaned = clean_token(flushed)
            if cleaned:
                accumulated += cleaned
                yield sse_event({"type": "token", "content": cleaned})

        final_answer = accumulated
        if ctx.tool_results:
            fixed = fix_answer_links(
                accumulated, ctx.tool_results,
                link_fix_patterns=self.config.link_fix_patterns,
            )
            if fixed != accumulated:
                yield sse_event(
                    {"type": "content_replace", "content": fixed}
                )
                final_answer = fixed

        total = time.time() - ctx.request_start
        answer_ms = round((time.time() - answer_start) * 1000)
        ctx.trace["answer_ms"] = answer_ms
        ctx.trace["answer_model"] = answer_cfg.model or str(self.llm.model)
        ctx.trace["answer_source"] = source
        ctx.trace["total_ms"] = round(total * 1000)

        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=True,
            event="answer_completed", source=source,
            duration_ms=answer_ms,
            model=answer_cfg.model or str(self.llm.model),
            answer_chars=len(final_answer),
        )
        self._log_completion(ctx, final_answer)

        yield sse_event(
            {
                "type": "done",
                "tool_calls": (
                    ctx.tool_calls_made if ctx.tool_calls_made else None
                ),
                "trace": ctx.trace,
            }
        )

    # -- Logging & error handling ---------------------------------------------

    def _log_request(self, ctx: _RunContext, request: ChatRequest) -> None:
        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=ctx.stream,
            event="request_received",
            question=request.message,
            history_len=len(request.history) if request.history else 0,
        )

    def _log_completion(self, ctx: _RunContext, answer: str) -> None:
        total = time.time() - ctx.request_start
        ctx.trace["total_ms"] = round(total * 1000)
        log_limit = int(
            os.environ.get("AGENT_LOG_ANSWER_CHARS", LOG_ANSWER_CHARS)
        )
        preview = ""
        if log_limit and answer:
            preview = answer[:log_limit] + (
                "..." if len(answer) > log_limit else ""
            )
        log_event(
            logger, logging.INFO, request_id=ctx.req_id, stream=ctx.stream,
            event="request_completed",
            total_ms=round(total * 1000),
            tool_count=len(ctx.tool_calls_made),
            **({
                "answer_preview": preview,
            } if preview else {}),
        )

    def _log_error(self, ctx: _RunContext, exc: Exception) -> None:
        log_event(
            logger, logging.ERROR, request_id=ctx.req_id, stream=ctx.stream,
            event="request_failed",
            error=str(exc), error_type=type(exc).__name__,
            total_ms=round((time.time() - ctx.request_start) * 1000),
        )

    def _handle_api_error(
        self, ctx: _RunContext, exc: Exception,
    ) -> ChatResponse:
        self._log_error(ctx, exc)
        return ChatResponse(
            message=llm_error_message(exc),
            tool_calls=ctx.tool_calls_made if ctx.tool_calls_made else None,
        )

    def _handle_unexpected_error(
        self, ctx: _RunContext, exc: Exception,
    ) -> ChatResponse:
        self._log_error(ctx, exc)
        return ChatResponse(
            message=(
                "An unexpected error occurred while processing your "
                "question. Please try again."
            ),
            tool_calls=ctx.tool_calls_made if ctx.tool_calls_made else None,
        )
