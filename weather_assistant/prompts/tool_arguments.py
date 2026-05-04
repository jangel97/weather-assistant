"""Prompt for the argument generator (Layer 3).

Given a selected tool, generates the correct arguments.
"""

import json

from framework.tools.registry import get_tool_examples


def _build_examples_block(tool_name: str) -> str:
    examples = get_tool_examples(tool_name)
    if not examples:
        return ""
    lines = ["Examples:"]
    for ex in examples:
        args = json.dumps({"arguments": ex["arguments"]})
        lines.append(f'User: "{ex["question"]}" -> {args}')
    return "\n".join(lines) + "\n"


def build_argument_prompt(tool_name: str, tool_schema: str, entities: list) -> str:
    examples_block = _build_examples_block(tool_name)

    return f"""You are an argument generator for a weather assistant tool. The tool \
"{tool_name}" has been selected. Your ONLY job is to generate the correct arguments.

Tool schema:
{tool_schema}

{examples_block}
Rules:
- Respond with ONLY a valid JSON object. No markdown, no explanation.
- Format: {{"arguments": {{...}}}}
- ONLY use parameter names that appear in the tool schema above.
- For "city", use the common English name of the city (e.g., "Tokyo", "New York", \
"London"). Do not use country names alone.
- For "days" in get_forecast, extract the number from the user's message. \
Default is 3 if not specified. Maximum is 7.
- For compare_weather, the "cities" parameter is a comma-separated string of \
city names (e.g., "Tokyo, London").
- If previous tool results are provided, read the actual data and extract values \
to use as arguments. NEVER use placeholders.
- If the tool has no required parameters and the user didn't specify any, \
return {{"arguments": {{}}}}."""
