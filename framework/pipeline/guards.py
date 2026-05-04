"""Hallucination detection and correction.

Small models sometimes invent keys (artifact names, drop identifiers) when
asked about "latest" or "newest" items, bypassing the discovery step.  The
guard detects this pattern — a detail tool called with a key but no prior
tool results — and redirects to the appropriate discovery tool.
"""

import logging

from framework.pipeline.logging import log_event
from framework.tools.registry import get_detail_tools

logger = logging.getLogger(__name__)


def check_for_hallucinated_key(
    tool_name: str,
    arguments: dict,
    tool_results: list,
    user_message: str = "",
    request_id: str | None = None,
    stream: bool = False,
) -> tuple | None:
    """Detect when a detail tool is called with a key but no prior results.

    Uses metadata from the tool registry (``get_detail_tools()``) so that
    adding a new detail tool automatically extends the guard.

    Returns ``(corrected_tool, corrected_args)`` if hallucination is detected,
    or ``None`` if the call looks legitimate.
    """
    if tool_results:
        return None

    detail_tools = get_detail_tools()
    if tool_name not in detail_tools:
        return None

    meta = detail_tools[tool_name]
    key_param = meta["key_param"]
    discovery_tool = meta["discovery_tool"]

    if key_param not in arguments:
        return None

    hallucinated_key = arguments[key_param]

    # If the user explicitly typed this key, allow it
    if hallucinated_key in user_message:
        return None

    log_event(
        logger, logging.WARNING, request_id=request_id, stream=stream,
        event="hallucination_corrected",
        original_tool=tool_name, corrected_tool=discovery_tool,
        key_param=key_param, key_value=hallucinated_key,
    )

    build_corrected = meta.get("build_corrected_args")
    if build_corrected:
        corrected_args = build_corrected(hallucinated_key, arguments)
    else:
        corrected_args = {}

    return discovery_tool, corrected_args
