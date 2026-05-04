import inspect
import json
import logging
from typing import Any, Callable, Dict, List

from framework.pipeline.logging import log_event

logger = logging.getLogger(__name__)

# Registry: tool_name -> handler_function
_TOOL_REGISTRY: Dict[str, Callable] = {}

# Examples: tool_name -> list of {"question": str, "arguments": dict}
_TOOL_EXAMPLES: Dict[str, List[dict]] = {}

# Detail-tool metadata: tool_name -> {"key_param": str, "discovery_tool": str}
_TOOL_DETAIL_META: Dict[str, dict] = {}

# Multi-step routing examples: list of {"question": str, "steps": list, "notes": list}
_MULTI_STEP_EXAMPLES: List[dict] = []


def register_tool(func: Callable) -> Callable:
    """Decorator that registers an async function as a callable tool.

    Uses the function's ``__name__`` as the tool identifier and its
    ``__doc__`` / ``inspect.signature`` for catalog generation.
    """
    _TOOL_REGISTRY[func.__name__] = func
    return func


def register_examples(tool_name: str, examples: List[dict]) -> None:
    """Register usage examples for a tool.

    Each example is a dict with ``question`` (str) and ``arguments`` (dict).
    These are used by prompt builders to dynamically generate few-shot
    examples for the action classifier, tool selector, and argument generator.
    """
    _TOOL_EXAMPLES[tool_name] = examples


def register_multi_step_examples(examples: List[dict]) -> None:
    """Register multi-step routing examples.

    Each example is a dict with:
    - ``question`` (str): The user question that triggers this pattern.
    - ``steps`` (list): Each step has ``round`` (int), ``context`` (str),
      and ``tool`` (str).
    - ``notes`` (list[str], optional): Extra hints for the model.
    """
    _MULTI_STEP_EXAMPLES.extend(examples)


def get_multi_step_examples() -> List[dict]:
    """Return all registered multi-step routing examples."""
    return list(_MULTI_STEP_EXAMPLES)


def register_detail_tool(
    tool_name: str,
    key_param: str,
    discovery_tool: str,
    build_corrected_args: Callable[[str, dict], dict] | None = None,
) -> None:
    """Declare a tool as a detail tool that requires a key from a prior discovery call.

    *build_corrected_args*, if provided, receives ``(hallucinated_key, original_args)``
    and returns the arguments dict to pass to *discovery_tool* when hallucination
    is detected.  If omitted, the framework passes an empty dict.
    """
    _TOOL_DETAIL_META[tool_name] = {
        "key_param": key_param,
        "discovery_tool": discovery_tool,
        "build_corrected_args": build_corrected_args,
    }


def get_detail_tools() -> Dict[str, dict]:
    """Return all registered detail tool metadata."""
    return dict(_TOOL_DETAIL_META)


def get_detail_tool_names() -> List[str]:
    """Return names of all detail tools."""
    return list(_TOOL_DETAIL_META.keys())


def get_discovery_tool_names() -> List[str]:
    """Return names of all non-detail (discovery/search) tools."""
    detail_names = set(_TOOL_DETAIL_META.keys())
    return [name for name in _TOOL_REGISTRY if name not in detail_names]


def get_tool_examples(tool_name: str) -> List[dict]:
    """Get registered examples for a specific tool."""
    return _TOOL_EXAMPLES.get(tool_name, [])


def get_all_tool_examples() -> Dict[str, List[dict]]:
    """Get all registered tool examples, keyed by tool name."""
    return dict(_TOOL_EXAMPLES)


def get_tool_catalog() -> str:
    """Generate a compact tool catalog from function metadata.

    Each line shows the tool name, parameters (required vs optional),
    and the first line of the docstring.
    """
    lines = []
    for name, func in _TOOL_REGISTRY.items():
        doc = (func.__doc__ or "").strip().split("\n")[0]
        sig = inspect.signature(func)

        param_parts = []
        for pname, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                param_parts.append(f"{pname}?")
            else:
                param_parts.append(pname)

        param_str = ", ".join(param_parts)
        lines.append(f"- {name}({param_str}): {doc}")

    return "\n".join(lines)


def is_valid_tool(name: str) -> bool:
    """Return True if *name* is a registered tool."""
    return name in _TOOL_REGISTRY


def fuzzy_match_tool(invalid_name: str) -> str | None:
    """Try to match an invalid tool name to a registered tool.

    Uses stem matching (first 4 chars of each underscore-separated word)
    to find the closest registered tool.  Requires at least 2 stem
    matches to avoid false positives.

    Returns the best matching tool name, or None if no good match.
    """
    _MIN_STEM = 4
    _MIN_MATCHES = 2

    invalid_stems = {
        w[:_MIN_STEM]
        for w in invalid_name.lower().split("_")
        if len(w) >= _MIN_STEM
    }
    if len(invalid_stems) < _MIN_MATCHES:
        return None

    best_match = None
    best_score = 0

    for name in _TOOL_REGISTRY:
        tool_stems = {
            w[:_MIN_STEM]
            for w in name.lower().split("_")
            if len(w) >= _MIN_STEM
        }
        score = len(invalid_stems & tool_stems)
        if score > best_score:
            best_score = score
            best_match = name

    return best_match if best_score >= _MIN_MATCHES else None


def get_tool_params(name: str) -> set:
    """Return the set of valid parameter names for a tool."""
    if name not in _TOOL_REGISTRY:
        return set()
    return set(inspect.signature(_TOOL_REGISTRY[name]).parameters.keys())


def get_tool_schema(name: str) -> str:
    """Generate a detailed schema for a single tool.

    Includes the full docstring and parameter list with types, defaults,
    and required/optional indicators.  Returns an empty string if the
    tool is not registered.
    """
    if name not in _TOOL_REGISTRY:
        return ""

    func = _TOOL_REGISTRY[name]
    sig = inspect.signature(func)

    # Build signature line
    param_parts = []
    for pname, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            param_parts.append(f"{pname}?")
        else:
            param_parts.append(pname)
    sig_line = f"{name}({', '.join(param_parts)})"

    # Full docstring
    docstring = (func.__doc__ or "").strip()

    # Parameter details from type hints + defaults
    type_hints = getattr(func, "__annotations__", {})
    param_lines = []
    for pname, param in sig.parameters.items():
        hint = type_hints.get(pname)
        type_str = _format_type_hint(hint) if hint else "any"
        required = param.default is inspect.Parameter.empty
        if required:
            param_lines.append(f"  {pname} ({type_str}, required)")
        elif param.default is None:
            param_lines.append(f"  {pname} ({type_str}, optional)")
        else:
            param_lines.append(
                f"  {pname} ({type_str}, optional, default={param.default!r})"
            )

    parts = [sig_line, "", docstring]
    if param_lines:
        parts.append("")
        parts.append("Parameters:")
        parts.extend(param_lines)

    return "\n".join(parts)


def _unwrap_optional(hint):
    """Unwrap Optional[X] to X. Returns the hint unchanged if not Optional."""
    args = getattr(hint, "__args__", None)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0]
    return hint


def _format_type_hint(hint) -> str:
    """Convert a type hint to a compact, readable string."""
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", None)

    # typing.Optional[X] is Union[X, None]
    if origin is type(None):
        return "None"

    if args and type(None) in args:
        # Optional[X] → unwrap to X
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _format_type_hint(inner[0])

    if origin is list or (hasattr(origin, "__name__") and origin.__name__ == "list"):
        if args:
            return f"list[{_format_type_hint(args[0])}]"
        return "list"

    if hasattr(hint, "__name__"):
        return hint.__name__

    return str(hint).replace("typing.", "")


async def execute_tool(name: str, arguments: str) -> str:
    """Execute a registered tool by name with JSON arguments.

    Returns serialized JSON result string for the LLM.
    """
    if name not in _TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})

    handler = _TOOL_REGISTRY[name]
    try:
        args: Dict[str, Any] = json.loads(arguments) if arguments else {}
        # Strip unknown parameters the LLM may have hallucinated
        sig = inspect.signature(handler)
        valid_params = set(sig.parameters.keys())
        unknown = set(args.keys()) - valid_params
        if unknown:
            log_event(
                logger, logging.WARNING,
                event="params_stripped", tool=name, params=list(unknown),
            )
            args = {k: v for k, v in args.items() if k in valid_params}

        # Coerce arrays to scalars when the schema expects a string
        type_hints = getattr(handler, "__annotations__", {})
        for pname, value in list(args.items()):
            if isinstance(value, list) and pname in type_hints:
                hint = type_hints[pname]
                # Unwrap Optional[str] → str
                inner = _unwrap_optional(hint)
                if inner is str and value:
                    log_event(
                        logger, logging.WARNING,
                        event="param_coerced", tool=name, param=pname,
                        from_value=value, to_value=value[0],
                    )
                    args[pname] = value[0]

        result = await handler(**args)
        return json.dumps(result, default=str)
    except Exception as e:
        log_event(
            logger, logging.ERROR,
            event="tool_exec_failed", tool=name, error=str(e),
        )
        return json.dumps({"error": str(e)})
