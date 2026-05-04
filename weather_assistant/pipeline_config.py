"""Weather assistant configuration — assembles PipelineConfig from prompt modules."""

from pathlib import Path

from weather_assistant.prompts import action, answer, rewriter, tool_arguments, tool_selector
from framework.config_loader import load_pipeline_config
from framework.pipeline_config import PromptConfig

_AGENT_YAML = Path(__file__).resolve().parent / "agent.yaml"


def get_pipeline_config():
    return load_pipeline_config(
        _AGENT_YAML,
        prompts=PromptConfig(
            build_action_prompt=action.build_action_prompt,
            build_tool_selector_prompt=tool_selector.build_tool_selector_prompt,
            build_argument_prompt=tool_arguments.build_argument_prompt,
            build_answer_prompt=answer.build_answer_prompt,
            build_system_prompt=answer.build_system_prompt,
            extractor_system_prompt=rewriter.EXTRACTOR_SYSTEM_PROMPT,
            rewriter_system_prompt=rewriter.REWRITER_SYSTEM_PROMPT,
        ),
    )
