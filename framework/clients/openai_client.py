"""OpenAI-compatible LLM client."""

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
    "base_url": None,
    "api_key": None,
    "model": None,
}


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI-compatible endpoints.

    Requires LLM_API_KEY and LLM_BASE_URL environment variables.

    Per-layer overrides are supported for providers with per-model
    endpoints (e.g. Models.corp).  Each layer suffix (CLASSIFIER,
    TOOL_SELECTOR, TOOL_ARGUMENTS, ANSWER) can override the API key,
    base URL, and model independently — unset values fall back to the
    global defaults.
    """

    # Layer suffixes that may have their own endpoint configuration.
    _LAYER_SUFFIXES = ("CLASSIFIER", "TOOL_SELECTOR", "TOOL_ARGUMENTS", "ANSWER")

    def __init__(self) -> None:
        api_key = os.environ.get("LLM_API_KEY") or _DEFAULTS["api_key"]
        base_url = os.environ.get("LLM_BASE_URL") or _DEFAULTS["base_url"]
        self._model = os.environ.get("LLM_MODEL") or _DEFAULTS["model"]

        if not api_key:
            raise ConfigurationError("LLM_API_KEY environment variable must be set")
        if not base_url:
            raise ConfigurationError("LLM_BASE_URL environment variable must be set")
        if not self._model:
            raise ConfigurationError("LLM_MODEL environment variable must be set")

        timeout = float(os.environ.get("LLM_TIMEOUT", LLM_TIMEOUT))
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", LLM_MAX_RETRIES))

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Build per-layer clients when a layer overrides api_key or
        # base_url.  Maps model name -> dedicated AsyncOpenAI client.
        self._layer_clients: dict[str, AsyncOpenAI] = {}

        for suffix in self._LAYER_SUFFIXES:
            layer_model = os.environ.get(f"LLM_MODEL_{suffix}")
            layer_url = os.environ.get(f"LLM_BASE_URL_{suffix}")
            layer_key = os.environ.get(f"LLM_API_KEY_{suffix}")

            if layer_model and (layer_url or layer_key):
                effective_url = layer_url or base_url
                self._layer_clients[layer_model] = AsyncOpenAI(
                    api_key=layer_key or api_key,
                    base_url=effective_url,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                log_event(
                    logger, logging.INFO,
                    event="llm_layer_configured",
                    layer=suffix, model=layer_model, base_url=effective_url,
                )

        log_event(
            logger, logging.INFO,
            event="llm_client_ready",
            provider="openai", model=self._model, base_url=base_url,
        )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self, model: str | None) -> AsyncOpenAI:
        """Return the appropriate client for the requested model."""
        if model and model in self._layer_clients:
            return self._layer_clients[model]
        return self._client

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
        return await self._get_client(model).chat.completions.create(**kwargs)

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
        return await self._get_client(model).chat.completions.create(**kwargs)
