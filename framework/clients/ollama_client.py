"""Ollama LLM client."""

import logging
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from framework.clients.base import BaseLLMClient
from framework.constants import LLM_MAX_RETRIES, LLM_TIMEOUT
from framework.pipeline.logging import log_event
from framework.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",  # Ollama ignores the key but the SDK requires one
    "model": None,
}


class OllamaClient(BaseLLMClient):
    """LLM client for local Ollama instances.

    Ollama exposes an OpenAI-compatible API, so this uses the same
    AsyncOpenAI SDK internally.  API key defaults to 'ollama' (ignored
    by the server) and base URL defaults to http://localhost:11434/v1.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("LLM_API_KEY") or _DEFAULTS["api_key"]
        base_url = os.environ.get("LLM_BASE_URL") or _DEFAULTS["base_url"]
        self._model = os.environ.get("LLM_MODEL") or _DEFAULTS["model"]
        if not self._model:
            raise ConfigurationError("LLM_MODEL environment variable must be set")

        timeout = float(os.environ.get("LLM_TIMEOUT", LLM_TIMEOUT))
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", LLM_MAX_RETRIES)),
        )

        answer_model = os.environ.get("LLM_MODEL_ANSWER")
        log_event(
            logger, logging.INFO,
            event="llm_client_ready",
            provider="ollama", model=self._model, base_url=base_url,
            routing_model=self._model,
            answer_model=answer_model or self._model,
        )

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def _do_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        kwargs = {
            "model": model or self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format
        return await self._client.chat.completions.create(**kwargs)

    async def _do_chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Any:
        kwargs = {
            "model": model or self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await self._client.chat.completions.create(**kwargs)
