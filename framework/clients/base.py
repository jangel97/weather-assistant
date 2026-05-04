"""Abstract base class for LLM provider clients."""

import abc
import logging
import time
from typing import Any, Dict, List, Optional

from framework.pipeline.logging import log_event

logger = logging.getLogger(__name__)


class BaseLLMClient(abc.ABC):
    """Interface that all LLM provider clients must implement.

    Concrete ``chat_completion`` and ``chat_completion_stream`` methods
    wrap the provider-specific ``_do_*`` methods with automatic timing
    and structured logging.  Subclasses only need to implement the raw
    SDK calls.
    """

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Return the provider identifier (e.g., 'openai', 'ollama')."""

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Return the default model name."""

    # -- Public API (timing + logging) ----------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Send a chat completion request with automatic timing."""
        effective_model = model or self.model
        start = time.time()
        response = await self._do_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            model=effective_model,
            response_format=response_format,
        )
        elapsed_ms = round((time.time() - start) * 1000)

        usage = getattr(response, "usage", None)
        log_event(
            logger, logging.INFO,
            event="llm_call",
            provider=self.provider, model=effective_model,
            duration_ms=elapsed_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
        return response

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Send a streaming chat completion request with automatic timing."""
        effective_model = model or self.model
        start = time.time()
        stream = await self._do_chat_completion_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            model=effective_model,
        )
        elapsed_ms = round((time.time() - start) * 1000)

        log_event(
            logger, logging.INFO,
            event="llm_stream_start",
            provider=self.provider, model=effective_model,
            time_to_first_chunk_ms=elapsed_ms,
        )
        return stream

    # -- Provider-specific implementations (subclasses override these) ---------

    @abc.abstractmethod
    async def _do_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Raw chat completion call — implement per provider."""

    @abc.abstractmethod
    async def _do_chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Raw streaming chat completion call — implement per provider."""
