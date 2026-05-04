"""Prompt for the tool selector (Layer 2).

Given that a weather API lookup is needed, picks which tool to use.
"""

import json

from framework.tools.registry import get_all_tool_examples, get_tool_catalog


def _build_examples_block() -> str:
    lines = ["Examples:"]
    for tool_name, examples in get_all_tool_examples().items():
        for ex in examples:
            lines.append(
                f'User: "{ex["question"]}" -> {json.dumps({"tool": tool_name})}'
            )
    return "\n".join(lines)


def build_tool_selector_prompt() -> str:
    catalog = get_tool_catalog()
    examples = _build_examples_block()

    return f"""You are a tool selector for a weather assistant. A weather API \
lookup is needed to answer the user's question. Your ONLY job is to pick which \
tool to use.

Tools:
{catalog}

{examples}
Rules:
- Respond with ONLY a valid JSON object. No markdown, no explanation.
- Format: {json.dumps({"tool": "<name>"})}
- Pick the tool that best matches the user's question.
- Use get_current_weather for "now", "right now", "current", "today" questions.
- Use get_forecast for "tomorrow", "this week", "next days", "will it rain" questions.
- Use compare_weather when the user asks to compare 2 or more cities.
- If previous tool results are provided, pick the next tool needed."""
