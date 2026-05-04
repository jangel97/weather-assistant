"""Singleton factory for LLM clients."""

import logging
import os
import threading
from typing import Dict, Type

from framework.clients.base import BaseLLMClient
from framework.clients.ollama_client import OllamaClient
from framework.clients.openai_client import OpenAIClient
from framework.clients.vertex_client import VertexClient
from framework.pipeline.logging import log_event
from framework.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_instance: BaseLLMClient | None = None
_lock = threading.Lock()

_PROVIDERS: Dict[str, Type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "ollama": OllamaClient,
    "vertex": VertexClient,
}


def get_llm_client() -> BaseLLMClient:
    """Return the singleton LLM client, creating it on first call.

    The provider is selected via the LLM_PROVIDER env var (default: 'openai').
    Thread-safe via double-checked locking.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                provider = os.environ.get("LLM_PROVIDER", "openai").lower()

                if provider not in _PROVIDERS:
                    raise ConfigurationError(
                        f"LLM_PROVIDER must be one of {list(_PROVIDERS)}, "
                        f"got '{provider}'"
                    )

                log_event(
                    logger, logging.INFO,
                    event="llm_client_init", provider=provider,
                )
                _instance = _PROVIDERS[provider]()
    return _instance


def reset_llm_client() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    with _lock:
        _instance = None
