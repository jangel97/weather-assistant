"""Response cleaning and link fixing.

LLM output often contains artifacts that need stripping before presenting
to the user: leaked chat-template tokens (``<|im_start|>``, etc.),
``<think>`` reasoning blocks, and broken markdown links.  This module
handles all post-processing of model output.
"""

import json
import re
from typing import Dict, List

from openai import APIConnectionError, APIError, APITimeoutError

# Regex to strip leaked chat template tokens from model output
CHAT_TEMPLATE_TOKENS = re.compile(
    r"<\|im_start\|>.*?</\|im_sep\|>\s*|<\|im_end\|>|<\|endoftext\|>"
)
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def clean_response(text: str) -> str:
    """Strip leaked chat template tokens and thinking blocks from model output."""
    cleaned = CHAT_TEMPLATE_TOKENS.sub("", text)
    cleaned = THINK_BLOCK.sub("", cleaned)
    return cleaned.strip()


def fix_answer_links(
    answer: str,
    tool_results: List[Dict[str, str]],
    link_fix_patterns: List[str] | None = None,
) -> str:
    """Fix links in LLM answers using URL data from tool results.

    Handles two problems small models create:
    1. Writing ``[text](url)`` literally instead of copying the real URL.
    2. Inventing external links from artifact keys instead of using the
       ``url`` field from tool results.

    *link_fix_patterns* is a list of URL prefixes that indicate an invented
    link (e.g. ``["https://quay.io", "https://registry.redhat.io"]``).
    """
    # Build a lookup: name/key -> dashboard url from all tool results
    name_to_url: Dict[str, str] = {}
    fallback_urls: list = []
    for tr in tool_results:
        try:
            data = json.loads(tr["result"])
        except (json.JSONDecodeError, KeyError):
            continue
        items = (
            data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        )
        for item in items:
            if not isinstance(item, dict) or "url" not in item:
                continue
            url = item["url"]
            fallback_urls.append(url)
            for field in ("name", "key"):
                if field in item:
                    name_to_url[item[field]] = url

    if not name_to_url and not fallback_urls:
        return answer

    _patterns = link_fix_patterns or []

    # Use an index into fallback_urls instead of pop() to avoid
    # mutating the list (safe if this function is called more than once).
    _fallback_idx = [0]

    # Match markdown links: [text](href)
    _LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def _replace_link(m: re.Match) -> str:
        link_text = m.group(1)
        href = m.group(2)

        needs_fix = href == "url" or any(
            href.startswith(p) for p in _patterns
        )
        if not needs_fix:
            return m.group(0)

        # Try to match link text to a known item
        if link_text in name_to_url:
            return f"[{link_text}]({name_to_url[link_text]})"
        # Fallback: use next available URL in order
        if _fallback_idx[0] < len(fallback_urls):
            url = fallback_urls[_fallback_idx[0]]
            _fallback_idx[0] += 1
            return f"[{link_text}]({url})"
        # Can't fix — remove the broken link, keep plain text
        return link_text

    answer = _LINK_RE.sub(_replace_link, answer)

    # Second pass: wrap plain-text keys in markdown links.
    # Split by existing markdown links so we only touch non-link text.
    link_parts = _LINK_RE.split(answer)
    # _LINK_RE has 2 capture groups, so split produces [text, g1, g2, text, ...]
    rebuilt = []
    for i, part in enumerate(link_parts):
        if i % 3 == 0:
            # Non-link text segment — linkify known keys
            replaced_keys: list[str] = []
            for key, url in sorted(
                name_to_url.items(), key=lambda kv: len(kv[0]), reverse=True
            ):
                if len(key) < 10 or key not in part:
                    continue
                # Skip if this key is a substring of an already-replaced
                # longer key (it would match inside the created link text)
                if any(key in rk for rk in replaced_keys):
                    continue
                if f"`{key}`" in part:
                    part = part.replace(f"`{key}`", f"[{key}]({url})")
                else:
                    part = part.replace(key, f"[{key}]({url})")
                replaced_keys.append(key)
            rebuilt.append(part)
        elif i % 3 == 1:
            # Link text group — reconstruct the original markdown link
            link_text = part
            link_href = link_parts[i + 1]
            rebuilt.append(f"[{link_text}]({link_href})")
        # i % 3 == 2 is the href group, already consumed above

    return "".join(rebuilt)


class ThinkBlockFilter:
    """Suppress ``<think>...</think>`` blocks from streaming token output.

    Unlike ``clean_response`` (which operates on a complete string), this
    handles tags that may be split across chunk boundaries by buffering
    partial tag prefixes.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._suppressing = False
        self._buf = ""

    def feed(self, token: str) -> str:
        """Process a streaming token, returning text to emit (may be empty)."""
        self._buf += token
        out: list[str] = []

        while self._buf:
            if self._suppressing:
                idx = self._buf.find(self._CLOSE)
                if idx != -1:
                    self._suppressing = False
                    self._buf = self._buf[idx + len(self._CLOSE) :]
                else:
                    # Keep a tail that could be a partial closing tag
                    safe = len(self._buf) - len(self._CLOSE) + 1
                    if safe > 0:
                        self._buf = self._buf[safe:]
                    break
            else:
                idx = self._buf.find(self._OPEN)
                if idx != -1:
                    out.append(self._buf[:idx])
                    self._suppressing = True
                    self._buf = self._buf[idx + len(self._OPEN) :]
                else:
                    # Keep a tail that could be a partial opening tag
                    safe = len(self._buf) - len(self._OPEN) + 1
                    if safe > 0:
                        out.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break

        return "".join(out)

    def flush(self) -> str:
        """Flush any remaining buffered text (call at end of stream)."""
        if self._suppressing:
            self._buf = ""
            return ""
        remaining = self._buf
        self._buf = ""
        return remaining


def clean_token(text: str) -> str:
    """Strip leaked chat template tokens from a single streaming token.

    Unlike clean_response, this preserves whitespace so spaces between
    words are not lost during streaming.
    """
    return CHAT_TEMPLATE_TOKENS.sub("", text)


def llm_error_message(exc: Exception) -> str:
    """Return a user-facing error message for LLM failures."""
    if isinstance(exc, APITimeoutError):
        return (
            "The request to the language model timed out. "
            "Please try again or ask a simpler question."
        )
    if isinstance(exc, APIConnectionError):
        return (
            "Could not connect to the language model. "
            "The service may be temporarily unavailable."
        )
    if isinstance(exc, APIError):
        return f"The language model returned an error: {exc.message}"
    return f"An unexpected error occurred: {exc}"
