"""JSON extraction from LLM output.

Small language models often wrap their JSON responses in markdown code blocks,
reasoning tags, or extra prose.  The helpers here robustly extract the JSON
object from whatever the model returns, using three strategies in order:
direct parse, markdown block extraction, and brace-counting.
"""

import json
import logging
import re

from framework.pipeline.logging import log_event

logger = logging.getLogger(__name__)


def parse_json(content: str, fallback: dict) -> dict:
    """Extract a JSON object from LLM output.

    Tries direct parse, markdown code block, and brace-counting.
    Returns *fallback* if nothing can be parsed.
    """
    text = content.strip()

    # 0. Strip reasoning model tags (Granite 3.3, Qwen3, DeepSeek, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract <response> content if present (Granite 3.3)
    resp_match = re.search(r"<response>(.*?)</response>", text, flags=re.DOTALL)
    if resp_match:
        text = resp_match.group(1).strip()

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Outermost JSON object via brace counting
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    log_event(
        logger, logging.WARNING,
        event="json_parse_failed", input_preview=text[:200],
    )
    return fallback


def parse_action_response(content: str) -> str:
    """Parse action classifier output and return the action string."""
    parsed = parse_json(content, {"action": "answer"})
    return parsed.get("action", "answer")


_TOOL_NAME_KEYS = ("tool", "next_tool", "nextTool", "tool_name", "toolName")


def parse_tool_name_response(content: str) -> str:
    """Parse tool selector output and return the tool name string.

    Small models sometimes use variant keys (``next_tool``, ``nextTool``,
    ``tool_name``) instead of the expected ``tool``.  All are accepted.
    """
    parsed = parse_json(content, {"tool": ""})
    for key in _TOOL_NAME_KEYS:
        value = parsed.get(key)
        if value:
            return value
    return ""


def parse_argument_response(content: str) -> dict:
    """Parse argument generator output and return the arguments dict."""
    parsed = parse_json(content, {"arguments": {}})
    return parsed.get("arguments", {})
