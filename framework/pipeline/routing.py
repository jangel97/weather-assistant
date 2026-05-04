"""Main routing loop and tool execution.

Orchestrates the three-stage routing pipeline that runs each round:
1. **Action classifier** — decides whether to call a tool or answer.
2. **Tool selector** — picks which tool to use.
3. **Argument generator** — generates the tool's arguments.

After each round the tool is executed and results accumulate until the
classifier decides enough data has been collected to answer.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List

from framework.clients import BaseLLMClient
from framework.pipeline_config import PipelineConfig
from framework.models import ChatRequest
from framework.pipeline.guards import check_for_hallucinated_key
from framework.pipeline.logging import log_event
from framework.pipeline.messages import (
    build_action_messages,
    build_argument_messages,
    build_tool_selector_messages,
)
from framework.pipeline.parsing import (
    parse_action_response,
    parse_argument_response,
    parse_tool_name_response,
)
from framework.tools.registry import (
    execute_tool,
    fuzzy_match_tool,
    get_tool_params,
    get_tool_schema,
    is_valid_tool,
)
from framework.types import RoutableEntity

logger = logging.getLogger(__name__)


def sse_event(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


async def execute_tool_call(
    decision: dict,
    seen_calls: set,
    tool_results: List[Dict[str, str]],
    tool_calls_made: List[Dict[str, str]],
    request_id: str | None = None,
    stream: bool = False,
) -> bool:
    """Validate and execute a tool call.

    Returns True if the tool was executed, False if execution was skipped
    (invalid arguments or duplicate call).
    """
    tool_name = decision.get("tool", "")
    tool_args = decision.get("arguments", {})
    if not isinstance(tool_args, dict):
        log_event(
            logger, logging.WARNING, request_id=request_id, stream=stream,
            event="tool_skipped", tool=tool_name, reason="invalid_arguments",
        )
        return False
    args_json = json.dumps(tool_args, sort_keys=True)

    call_signature = (tool_name, args_json)
    if call_signature in seen_calls:
        log_event(
            logger, logging.WARNING, request_id=request_id, stream=stream,
            event="tool_skipped", tool=tool_name,
            arguments=args_json, reason="duplicate",
        )
        return False
    seen_calls.add(call_signature)

    tool_start = time.time()
    tool_result = await execute_tool(tool_name, args_json)
    tool_elapsed = time.time() - tool_start
    log_event(
        logger, logging.INFO, request_id=request_id, stream=stream,
        event="tool_executed", tool=tool_name,
        arguments=args_json,
        duration_ms=round(tool_elapsed * 1000),
        result_chars=len(tool_result),
    )

    tool_results.append(
        {"tool": tool_name, "arguments": args_json, "result": tool_result}
    )
    tool_calls_made.append({"tool": tool_name, "arguments": args_json})
    return True


async def routing_loop(
    llm: BaseLLMClient,
    request: ChatRequest,
    tool_results: List[Dict[str, str]],
    tool_calls_made: List[Dict[str, str]],
    seen_calls: set,
    trace: Dict,
    entities: List[RoutableEntity],
    config: PipelineConfig,
    request_id: str | None = None,
    stream: bool = False,
    emit_events: bool = False,
) -> AsyncGenerator[str, None]:
    """Run the three-stage routing loop.

    Stage 1 (action classifier): Decides whether to call a tool or answer.
    Stage 2 (tool selector): Picks which tool to use.
    Stage 3 (argument generator): Generates arguments for the selected tool.

    When *emit_events* is True, yields SSE event strings for tool progress
    (streaming mode).  The caller must consume the generator even in
    non-streaming mode so that the loop executes.

    The *trace* dict is populated with structured timing and decision data
    for evaluation and debugging.
    """
    action_prompt = config.prompts.build_action_prompt()
    selector_prompt = config.prompts.build_tool_selector_prompt()

    action_cfg = config.models.action
    selector_cfg = config.models.selector
    argument_cfg = config.models.argument

    for round_num in range(config.max_tool_rounds):
        round_trace: Dict[str, Any] = {"round": round_num + 1}

        # -- Stage 1: Action classification ------------------------------------
        action_messages = build_action_messages(
            action_prompt, request, tool_results, config.max_tool_result_chars,
        )
        start = time.time()
        action_response = await llm.chat_completion(
            messages=action_messages,
            temperature=action_cfg.temperature,
            max_tokens=action_cfg.max_tokens,
            model=action_cfg.model,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - start
        action_content = action_response.choices[0].message.content or ""
        action = parse_action_response(action_content)
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="action_classified",
            round=round_num + 1, action=action,
            duration_ms=round(elapsed * 1000),
            model=action_cfg.model or str(llm.model),
        )

        round_trace["action"] = action
        round_trace["classifier_ms"] = round(elapsed * 1000)
        round_trace["classifier_model"] = action_cfg.model or str(llm.model)

        if action != "tool":
            trace["rounds"].append(round_trace)
            return

        # -- Stage 2: Tool name selection --------------------------------------
        selector_messages = build_tool_selector_messages(
            selector_prompt, request, tool_results,
            has_specific_key=config.has_specific_key,
            max_tool_result_chars=config.max_tool_result_chars,
        )
        start = time.time()
        selector_response = await llm.chat_completion(
            messages=selector_messages,
            temperature=selector_cfg.temperature,
            max_tokens=selector_cfg.max_tokens,
            model=selector_cfg.model,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - start
        selector_content = selector_response.choices[0].message.content or ""
        tool_name = parse_tool_name_response(selector_content)
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="tool_selected",
            round=round_num + 1, tool=tool_name,
            duration_ms=round(elapsed * 1000),
            model=selector_cfg.model or str(llm.model),
        )

        round_trace["tool"] = tool_name
        round_trace["selector_ms"] = round(elapsed * 1000)
        round_trace["selector_model"] = selector_cfg.model or str(llm.model)

        if not is_valid_tool(tool_name):
            corrected = fuzzy_match_tool(tool_name) if tool_name else None
            if corrected:
                log_event(
                    logger, logging.INFO, request_id=request_id, stream=stream,
                    event="tool_fuzzy_matched",
                    from_tool=tool_name, to_tool=corrected,
                )
                round_trace["fuzzy_corrected"] = {
                    "from": tool_name, "to": corrected
                }
                tool_name = corrected
                round_trace["tool"] = tool_name
            else:
                log_event(
                    logger, logging.WARNING, request_id=request_id, stream=stream,
                    event="tool_invalid",
                    round=round_num + 1, tool=tool_name,
                )
                round_trace["error"] = f"invalid tool: {tool_name}"
                trace["rounds"].append(round_trace)
                continue

        # -- Stage 3: Argument generation --------------------------------------
        tool_schema = get_tool_schema(tool_name)
        arg_prompt = config.prompts.build_argument_prompt(
            tool_name, tool_schema, entities
        )
        arg_messages = build_argument_messages(
            arg_prompt, request, tool_results, config.max_tool_result_chars,
        )
        start = time.time()
        arg_response = await llm.chat_completion(
            messages=arg_messages,
            temperature=argument_cfg.temperature,
            max_tokens=argument_cfg.max_tokens,
            model=argument_cfg.model,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - start
        arg_content = arg_response.choices[0].message.content or ""
        arguments = parse_argument_response(arg_content)
        log_event(
            logger, logging.INFO, request_id=request_id, stream=stream,
            event="arguments_generated",
            round=round_num + 1, tool=tool_name,
            arguments=arguments,
            duration_ms=round(elapsed * 1000),
            model=argument_cfg.model or str(llm.model),
        )

        # -- Strip unknown parameters ------------------------------------------
        valid_params = get_tool_params(tool_name)
        unknown = set(arguments.keys()) - valid_params
        if unknown:
            log_event(
                logger, logging.WARNING, request_id=request_id, stream=stream,
                event="params_stripped", tool=tool_name, params=list(unknown),
            )
            arguments = {k: v for k, v in arguments.items() if k in valid_params}

        round_trace["arguments"] = arguments
        round_trace["argument_ms"] = round(elapsed * 1000)
        round_trace["argument_model"] = argument_cfg.model or str(llm.model)

        # -- Entity key injection ----------------------------------------------
        msg_lower = request.message.lower()
        for entity in entities:
            pname = entity.param_name
            if pname not in arguments and pname in valid_params:
                if entity.key in msg_lower or entity.short_name.lower() in msg_lower:
                    arguments[pname] = entity.key
                    log_event(
                        logger, logging.INFO, request_id=request_id, stream=stream,
                        event="entity_key_injected", param=pname, key=entity.key,
                    )
                    break

        # -- Hallucination guard -----------------------------------------------
        correction = check_for_hallucinated_key(
            tool_name,
            arguments,
            tool_results,
            user_message=request.message,
            request_id=request_id,
            stream=stream,
        )
        if correction:
            round_trace["hallucination_corrected"] = True
            round_trace["original_tool"] = tool_name
            tool_name, arguments = correction
            round_trace["tool"] = tool_name
            round_trace["arguments"] = arguments

        if emit_events:
            args_json = json.dumps(arguments, sort_keys=True)
            yield sse_event(
                {"type": "tool_call", "tool": tool_name, "arguments": args_json}
            )

        decision = {"tool": tool_name, "arguments": arguments}
        tool_start = time.time()
        if not await execute_tool_call(
            decision,
            seen_calls,
            tool_results,
            tool_calls_made,
            request_id=request_id,
            stream=stream,
        ):
            round_trace["skipped"] = "duplicate or invalid call"
            trace["rounds"].append(round_trace)
            continue

        round_trace["tool_exec_ms"] = round((time.time() - tool_start) * 1000)
        trace["rounds"].append(round_trace)

        if emit_events:
            yield sse_event({"type": "tool_result", "tool": tool_name})
