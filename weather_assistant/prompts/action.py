"""Prompt for the action classifier (Layer 1).

Decides whether the user's question needs a weather API lookup (tool)
or can be answered from general knowledge alone.
"""

import json

from framework.tools.registry import get_all_tool_examples

_KNOWLEDGE_EXAMPLES = [
    "What is weather?",
    "How does a barometer work?",
    "What causes hurricanes?",
    "What is the difference between Celsius and Fahrenheit?",
    "Hello",
    "Thanks!",
]


def _build_examples_block() -> str:
    lines = ["Examples:"]

    for tool_name, examples in get_all_tool_examples().items():
        for ex in examples[:2]:
            lines.append(f'User: "{ex["question"]}" -> {{"action": "tool"}}')

    for q in _KNOWLEDGE_EXAMPLES:
        lines.append(f'User: "{q}" -> {{"action": "answer"}}')

    return "\n".join(lines)


def build_action_prompt() -> str:
    examples = _build_examples_block()

    return f"""You are a classifier for a weather assistant. Your only job is to \
decide whether the user's question needs a weather API lookup or can be answered \
from general knowledge.

Questions that need a weather API lookup (action=tool):
- Asking for current weather, temperature, humidity, wind for a specific city
- Asking for weather forecasts
- Comparing weather between cities
- Any question about actual weather conditions right now or in coming days

Questions answerable from knowledge (action=answer):
- Definitions and explanations (what is humidity, how does weather work)
- General science questions about weather phenomena
- Greetings, thanks, compliments, off-topic questions

{examples}
Follow-up examples:
Previous data is current weather for Tokyo.
User: "what about London?" -> {{"action": "tool"}}
(Different city = new lookup needed.)

Previous data is current weather for Paris.
User: "will it rain tomorrow?" -> {{"action": "tool"}}
(Forecast = different data than current conditions.)

Previous data is current weather for Madrid.
User: "thanks!" -> {{"action": "answer"}}

Rules:
- Respond with ONLY a valid JSON object. No markdown, no explanation.
- {json.dumps({"action": "tool"})} if a weather API lookup is needed.
- {json.dumps({"action": "answer"})} if answerable from general knowledge or the data \
already provided.
- If previous tool results are provided and contain enough data to answer the \
question, respond with {json.dumps({"action": "answer"})}.
- If previous tool results are provided but more data is needed, respond with \
{json.dumps({"action": "tool"})}.
- Conversation history is NOT data. Only raw JSON blocks labeled \
"Data collected so far" count as available data."""
