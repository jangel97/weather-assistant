from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

from framework.constants import (
    MAX_TOOL_RESULT_CHARS,
    MAX_TOOL_ROUNDS,
)


@dataclass
class LayerConfig:
    """LLM settings for one pipeline layer."""

    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class PromptConfig:
    """Prompt builders for each pipeline layer."""

    build_action_prompt: Callable[[], str] = lambda: ""
    build_tool_selector_prompt: Callable[[], str] = lambda: ""
    build_argument_prompt: Callable = lambda tool_name, tool_schema, products: ""
    build_answer_prompt: Callable[[], str] = lambda: ""
    build_system_prompt: Callable[[], str] = lambda: ""
    extractor_system_prompt: str = ""
    rewriter_system_prompt: str = ""


@dataclass
class ModelConfig:
    """Per-layer LLM settings."""

    action: LayerConfig = field(default_factory=LayerConfig)
    selector: LayerConfig = field(default_factory=LayerConfig)
    argument: LayerConfig = field(default_factory=LayerConfig)
    # Answer generation benefits from slight temperature for natural language
    answer: LayerConfig = field(
        default_factory=lambda: LayerConfig(temperature=0.1, max_tokens=4096)
    )
    rewriter: LayerConfig = field(default_factory=LayerConfig)
    extractor: LayerConfig = field(
        default_factory=lambda: LayerConfig(max_tokens=256)
    )


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    prompts: PromptConfig = field(default_factory=PromptConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS
    link_fix_patterns: list[str] = field(default_factory=list)
    context_sanitizer: Optional[Callable[[Dict, str], Dict]] = None
    _domains: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._domains = [
            urlparse(p).hostname + "/"
            for p in self.link_fix_patterns
            if urlparse(p).hostname
        ]

    def has_specific_key(self, msg: str) -> bool:
        """Check if the message contains a key matching any link_fix_patterns domain."""
        return any(domain in msg for domain in self._domains)
