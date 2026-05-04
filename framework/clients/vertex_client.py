"""Vertex AI Claude client.

Translates between the OpenAI-style interface used by the pipeline
and the Anthropic Messages API exposed via Vertex AI on Models.corp.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from framework.clients.base import BaseLLMClient
from framework.constants import LLM_MAX_RETRIES, LLM_TIMEOUT
from framework.pipeline.logging import log_event
from framework.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Models.corp Vertex AI endpoint pattern:
# {base_url}/{tier}/models/{model_id}:streamRawPredict
# e.g. /sonnet/models/claude-sonnet-4@20250514:streamRawPredict

_MODEL_TIERS = {
    "claude-sonnet-4@20250514": "sonnet",
    "claude-haiku-4-5@20251001": "haiku",
    "claude-opus-4@20250514": "opus",
    "claude-opus-4-1@20250805": "opus",
}


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Message:
    content: str = ""
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message = field(default_factory=_Message)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _ChatCompletionResponse:
    """Mimics the OpenAI ChatCompletion response shape."""

    choices: List[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)
    model: str = ""
    id: str = ""


def _anthropic_to_openai(data: dict) -> _ChatCompletionResponse:
    """Convert an Anthropic Messages API response to OpenAI shape."""
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")

    usage_data = data.get("usage", {})
    return _ChatCompletionResponse(
        choices=[_Choice(message=_Message(content=text))],
        usage=_Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
        ),
        model=data.get("model", ""),
        id=data.get("id", ""),
    )


@dataclass
class _Delta:
    content: str = ""


@dataclass
class _StreamChoice:
    delta: _Delta = field(default_factory=_Delta)
    index: int = 0


@dataclass
class _StreamChunk:
    choices: List[_StreamChoice] = field(default_factory=list)


class _FakeStream:
    """Wraps a complete response as an async iterator of stream chunks."""

    def __init__(self, text: str, chunk_size: int = 20):
        self._text = text
        self._chunk_size = chunk_size
        self._pos = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._text):
            raise StopAsyncIteration
        end = min(self._pos + self._chunk_size, len(self._text))
        chunk_text = self._text[self._pos:end]
        self._pos = end
        return _StreamChunk(choices=[_StreamChoice(delta=_Delta(content=chunk_text))])


class VertexClient(BaseLLMClient):
    """LLM client for Claude via Vertex AI on Models.corp.

    Uses the Anthropic Messages API format with Bearer auth.

    Env vars:
        LLM_API_KEY: Models.corp application credential
        LLM_BASE_URL: Base URL (default: Models.corp Claude endpoint)
        LLM_MODEL: Model ID (e.g. claude-sonnet-4@20250514)
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise ConfigurationError("LLM_BASE_URL must be set for vertex provider")
        self._base_url = base_url.rstrip("/")
        self._model = os.environ.get("LLM_MODEL", "claude-sonnet-4@20250514")

        if not self._api_key:
            raise ConfigurationError("LLM_API_KEY must be set for vertex provider")

        timeout = float(os.environ.get("LLM_TIMEOUT", LLM_TIMEOUT))
        self._max_retries = int(os.environ.get("LLM_MAX_RETRIES", LLM_MAX_RETRIES))
        self._timeout = timeout

        # Per-layer model overrides (e.g. LLM_MODEL_ANSWER)
        self._layer_models: dict[str, str] = {}
        self._layer_keys: dict[str, str] = {}
        for suffix in ("CLASSIFIER", "TOOL_SELECTOR", "TOOL_ARGUMENTS", "ANSWER"):
            layer_model = os.environ.get(f"LLM_MODEL_{suffix}")
            if layer_model:
                self._layer_models[suffix] = layer_model
                layer_key = os.environ.get(f"LLM_API_KEY_{suffix}")
                if layer_key:
                    self._layer_keys[layer_model] = layer_key

        self._client = httpx.AsyncClient(
            timeout=timeout, verify=True,
        )

        log_event(
            logger, logging.INFO,
            event="llm_client_ready",
            provider="vertex", model=self._model, base_url=self._base_url,
        )

    @property
    def provider(self) -> str:
        return "vertex"

    @property
    def model(self) -> str:
        return self._model

    def _build_url(self, model: str) -> str:
        """Build the Vertex AI endpoint URL for a model."""
        tier = _MODEL_TIERS.get(model)
        if not tier:
            raise ConfigurationError(
                f"Unknown Claude model '{model}'. "
                f"Known models: {list(_MODEL_TIERS)}"
            )
        return f"{self._base_url}/{tier}/models/{model}:streamRawPredict"

    def _get_api_key(self, model: str) -> str:
        """Get the API key for a model (per-layer or default)."""
        return self._layer_keys.get(model, self._api_key)

    def _build_body(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Convert OpenAI-style messages to Anthropic format."""
        system_text = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text = content
                continue

            anthropic_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })

        body: dict = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_text:
            body["system"] = system_text

        return body

    async def _do_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        effective_model = model or self._model
        url = self._build_url(effective_model)
        body = self._build_body(messages, temperature, max_tokens, response_format)
        api_key = self._get_api_key(effective_model)

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(
                    url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return _anthropic_to_openai(data)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                if attempt < self._max_retries:
                    log_event(
                        logger, logging.WARNING,
                        event="vertex_retry",
                        attempt=attempt + 1, error=str(e),
                    )
                    import asyncio
                    await asyncio.sleep(2 ** attempt)

        raise last_error

    async def _do_chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Any:
        # Get the full response, then simulate streaming by chunking the text.
        response = await self._do_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            model=model,
        )
        text = response.choices[0].message.content if response.choices else ""
        return _FakeStream(text)
