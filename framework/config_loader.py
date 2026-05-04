"""YAML-based agent configuration loader.

Reads an ``agent.yaml`` file and returns pipeline configuration.  String
values matching ``${ENV_VAR}`` are resolved from the environment at load
time — if the variable is unset the value becomes ``None``, which tells
the framework to fall back to defaults.

Top-level ``llm`` fields (``provider``, ``api_key``, ``base_url``,
``model``, ``timeout``, ``max_retries``) are written into the
corresponding ``LLM_*`` environment variables so the existing client
factory picks them up.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from framework.pipeline_config import (
    LayerConfig,
    ModelConfig,
    PipelineConfig,
    PromptConfig,
)

_ENV_PATTERN = re.compile(r"^\$\{(\w+)\}$")

_LAYER_NAMES = {
    "classifier": "action",
    "tool_selector": "selector",
    "tool_arguments": "argument",
    "answer": "answer",
    "rewriter": "rewriter",
    "extractor": "extractor",
}

_LLM_ENV_MAP = {
    "provider": "LLM_PROVIDER",
    "api_key": "LLM_API_KEY",
    "base_url": "LLM_BASE_URL",
    "model": "LLM_MODEL",
    "timeout": "LLM_TIMEOUT",
    "max_retries": "LLM_MAX_RETRIES",
}

_PIPELINE_FIELDS = {
    "max_tool_rounds": int,
    "max_tool_result_chars": int,
}


@dataclass
class AgentConfig:
    """Full agent configuration loaded from YAML."""

    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    # timeout and max_retries are set as LLM_TIMEOUT / LLM_MAX_RETRIES
    # env vars; the LLM clients read them from the environment at init time.
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    max_tool_rounds: Optional[int] = None
    max_tool_result_chars: Optional[int] = None
    link_fix_patterns: Optional[list[str]] = None
    models: ModelConfig = field(default_factory=ModelConfig)


def _resolve_env(value):
    """Resolve ``${ENV_VAR}`` references in string values."""
    if not isinstance(value, str):
        return value
    m = _ENV_PATTERN.match(value.strip())
    if m:
        # Treat empty env vars the same as unset (fall back to defaults)
        return os.environ.get(m.group(1)) or None
    return value


def _load_models(models_data: dict) -> ModelConfig:
    """Parse the ``models:`` section into a ``ModelConfig``."""
    kwargs = {}
    for yaml_key, field_name in _LAYER_NAMES.items():
        section = models_data.get(yaml_key)
        if section is None:
            continue
        kwargs[field_name] = LayerConfig(
            model=_resolve_env(section.get("model")),
            temperature=float(section.get("temperature", 0.0)),
            max_tokens=int(section.get("max_tokens", 1024)),
        )
    return ModelConfig(**kwargs)


def load_agent_config(path: Path) -> AgentConfig:
    """Load full agent configuration from a YAML file.

    Top-level ``llm`` fields are resolved and set as environment
    variables so ``get_llm_client()`` picks them up automatically.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    llm_data = data.get("llm", {})
    config = AgentConfig()

    for yaml_key, env_var in _LLM_ENV_MAP.items():
        raw = llm_data.get(yaml_key)
        if raw is not None:
            resolved = _resolve_env(raw)
            if resolved is not None:
                setattr(config, yaml_key, resolved)
                os.environ.setdefault(env_var, str(resolved))

    pipeline_data = data.get("pipeline", {})
    for yaml_key, cast in _PIPELINE_FIELDS.items():
        raw = pipeline_data.get(yaml_key)
        if raw is not None:
            resolved = _resolve_env(raw)
            if resolved is not None:
                setattr(config, yaml_key, cast(resolved))

    postprocessing = data.get("output_text_postprocessing", {})
    patterns = postprocessing.get("link_fix_patterns")
    if patterns is not None:
        config.link_fix_patterns = patterns

    config.models = _load_models(data.get("models", {}))
    return config


def load_pipeline_config(
    path: Path,
    prompts: PromptConfig | None = None,
) -> PipelineConfig:
    """Load a complete ``PipelineConfig`` from a YAML file.

    This is the primary entry point for consumers.  It reads all
    declarative settings from the YAML and combines them with the
    provided ``prompts`` (which must be Python callables).
    """
    agent_cfg = load_agent_config(path)

    kwargs: dict = {}
    if agent_cfg.max_tool_rounds is not None:
        kwargs["max_tool_rounds"] = agent_cfg.max_tool_rounds
    if agent_cfg.max_tool_result_chars is not None:
        kwargs["max_tool_result_chars"] = agent_cfg.max_tool_result_chars
    if agent_cfg.link_fix_patterns is not None:
        kwargs["link_fix_patterns"] = agent_cfg.link_fix_patterns

    return PipelineConfig(
        prompts=prompts or PromptConfig(),
        models=agent_cfg.models,
        **kwargs,
    )


def load_model_config(path: Path) -> ModelConfig:
    """Load only the model configuration from a YAML file."""
    return load_agent_config(path).models
