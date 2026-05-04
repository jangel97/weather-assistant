"""Message list builders for each LLM layer.

Each stage of the routing pipeline (classifier, tool selector, argument
generator, answer LLM) needs its own message list with the right system
prompt and context.  This module builds those lists, keeping routing layers
free of conversation history (which causes small models to answer directly
instead of producing structured JSON).
"""

from typing import Callable, Dict, List, Optional

from framework.constants import MAX_TOOL_RESULT_CHARS
from framework.models import ChatRequest
from framework.tools.registry import get_detail_tool_names, get_detail_tools


def format_tool_results(
    tool_results: List[Dict[str, str]],
    max_chars_per_result: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    """Format accumulated tool results as compact context.

    Each result is truncated to keep the context manageable for small models.
    """
    parts = []
    total = len(tool_results)
    for i, tr in enumerate(tool_results, 1):
        result_text = tr["result"]
        limit = max_chars_per_result * 2 if i == total else max_chars_per_result
        if len(result_text) > limit:
            result_text = result_text[:limit] + "\n... (truncated)"
        parts.append(
            f"Step {i}: Called {tr['tool']}({tr['arguments']})\n"
            f"Result:\n```json\n{result_text}\n```"
        )
    return "\n\n".join(parts)


def format_tool_results_for_answer(
    user_message: str, tool_results: List[Dict[str, str]]
) -> str:
    """Build the user message for the answer LLM with all tool results."""
    parts = [user_message, ""]
    for i, tr in enumerate(tool_results, 1):
        parts.append(f"```json\n{tr['result']}\n```")
    return "\n\n".join(parts)


def build_action_messages(
    action_prompt: str,
    request: ChatRequest,
    tool_results: List[Dict[str, str]],
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> List[Dict[str, str]]:
    """Build message list for the action classifier."""
    return build_routing_messages(
        action_prompt,
        request,
        tool_results,
        suffix="Does the data above answer THIS question, or is a NEW lookup needed? "
        "Respond with JSON."
        if tool_results
        else "",
        max_tool_result_chars=max_tool_result_chars,
    )


def build_routing_messages(
    system_prompt: str,
    request: ChatRequest,
    tool_results: List[Dict[str, str]],
    suffix: str,
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> List[Dict[str, str]]:
    """Build message list for routing layers (classifier, selector, arguments).

    Excludes conversation history — routing layers only need the current
    question and any tool results from the current loop.  History causes
    small models to answer directly instead of producing structured JSON.
    """
    messages = [{"role": "system", "content": system_prompt}]

    if tool_results:
        user_content = (
            f"{request.message}\n\n"
            "Data collected so far:\n\n"
            + format_tool_results(tool_results, max_tool_result_chars)
            + f"\n\n{suffix}"
        )
    else:
        user_content = f"{request.message}\n\n{suffix}" if suffix else request.message

    messages.append({"role": "user", "content": user_content})
    return messages


def build_tool_selector_messages(
    selector_prompt: str,
    request: ChatRequest,
    tool_results: List[Dict[str, str]],
    has_specific_key: Optional[Callable[[str], bool]] = None,
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> List[Dict[str, str]]:
    """Build message list for the tool name selector."""
    if tool_results:
        previous_tools = ", ".join(tr["tool"] for tr in tool_results)
        suffix = (
            f"Tools already called: {previous_tools}. "
            "Pick the tool to call for more data. Respond with JSON."
        )
    elif has_specific_key and has_specific_key(request.message):
        suffix = (
            "The user mentioned a specific artifact key or URL. "
            "Pick the tool that can look it up directly. Respond with JSON."
        )
    else:
        detail_names = ", ".join(get_detail_tool_names())
        suffix = (
            "This is the FIRST round — no data has been collected yet. "
            "Pick a list or search tool to discover items first. "
            f"Do NOT pick {detail_names} without prior data. Respond with JSON."
        )
    return build_routing_messages(
        selector_prompt,
        request,
        tool_results,
        suffix=suffix,
        max_tool_result_chars=max_tool_result_chars,
    )


def build_argument_messages(
    argument_prompt: str,
    request: ChatRequest,
    tool_results: List[Dict[str, str]],
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
) -> List[Dict[str, str]]:
    """Build message list for the argument generator."""
    if tool_results:
        suffix = (
            "Generate the arguments for the tool using data from previous steps. "
            "Respond with JSON."
        )
    else:
        key_params = ", ".join(
            meta["key_param"] for meta in get_detail_tools().values()
        )
        suffix = (
            "Generate the arguments for the tool. "
            "No previous tool data is available, so do NOT invent or guess "
            f"any key values ({key_params}). "
            "Respond with JSON."
        )
    return build_routing_messages(
        argument_prompt,
        request,
        tool_results,
        suffix=suffix,
        max_tool_result_chars=max_tool_result_chars,
    )


def build_answer_messages(
    request: ChatRequest,
    tool_results: List[Dict[str, str]],
    answer_prompt: str,
    system_prompt: str,
) -> List[Dict[str, str]]:
    """Build message list for the answer/knowledge LLM.

    *answer_prompt* is used when tool results are available (data
    interpretation).  *system_prompt* is used when answering from
    knowledge alone (no tools were called).
    """
    if tool_results:
        messages = [{"role": "system", "content": answer_prompt}]

        if request.history:
            for msg in request.history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append(
            {
                "role": "user",
                "content": format_tool_results_for_answer(
                    request.message, tool_results
                ),
            }
        )
    else:
        messages = [{"role": "system", "content": system_prompt}]

        if request.history:
            for msg in request.history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": request.message})

    return messages
