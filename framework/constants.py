"""Centralised default constants for the agent.

All tuneable defaults live here so they are easy to find.  Each value
can be overridden at runtime via the corresponding environment variable
(noted in the comment).  Modules import the constant they need and pass
it as the fallback to ``os.environ.get()``.
"""

# ---------------------------------------------------------------------------
# Request limits  (env: AGENT_MAX_MESSAGE_LENGTH, etc.)
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_MESSAGE_LENGTH = 4000

# ---------------------------------------------------------------------------
# LLM client  (env: LLM_TIMEOUT, etc.)
# ---------------------------------------------------------------------------
LLM_TIMEOUT = 300
LLM_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Routing / tool execution  (env: AGENT_MAX_TOOL_ROUNDS, etc.)
# ---------------------------------------------------------------------------
MAX_TOOL_ROUNDS = 3
MAX_TOOL_RESULT_CHARS = 2000

# ---------------------------------------------------------------------------
# Logging  (env: AGENT_LOG_ANSWER_CHARS)
# ---------------------------------------------------------------------------
# Max characters of the final answer to include in log output.
# Set to 0 to disable answer logging entirely.
LOG_ANSWER_CHARS = 200
