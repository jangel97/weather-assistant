"""Prompts for the two-layer question rewriter (Layer 0).

Layer 1 (context extraction): Extracts key entities from the last
assistant message into a small JSON.

Layer 2 (rewrite with context): Rewrites the follow-up question using
the extracted entities, producing a standalone question for routing.

Only runs when conversation history is present.
"""

EXTRACTOR_SYSTEM_PROMPT = """\
Extract key entities from the assistant's last message. Return a JSON object \
with these fields:
- "city": city name mentioned (e.g., "Tokyo", "New York") or null
- "metric": what was discussed (e.g., "temperature", "humidity", "forecast", \
"wind") or null
- "timeframe": time period mentioned (e.g., "today", "tomorrow", "this week", \
"5-day") or null

Rules:
- Copy city names EXACTLY as they appear — do not modify or abbreviate.
- If a field is not present in the message, set it to null.
- Respond with ONLY a valid JSON object. No markdown, no explanation."""

REWRITER_SYSTEM_PROMPT = """\
Rewrite the user's latest message into a standalone question using the \
provided context.

The context JSON contains entities from the previous assistant response. \
Use these to resolve references like "it", "there", "that city", etc.

Rules:
- If the follow-up is NOT related to weather or the context \
(e.g., a compliment, greeting, or off-topic statement), \
output exactly: PASS_THROUGH
- Your output MUST be a single question ending with "?" or a request \
starting with an imperative verb (show, get, tell, compare) \
— or PASS_THROUGH if unrelated.
- Do NOT answer the question. Do NOT include any data or facts.
- Keep it short — one sentence maximum.
- Replace pronouns and references with the actual entity from the context.

Examples:
Context: {"city": "Tokyo", "metric": "temperature"}
User asked before: "What's the weather in Tokyo?"
Follow-up: "what about London?"
Output: What's the weather in London?

Context: {"city": "Paris", "metric": "forecast"}
User asked before: "5-day forecast for Paris"
Follow-up: "and for Berlin?"
Output: What is the 5-day forecast for Berlin?

Context: {"city": "Madrid", "metric": "temperature"}
User asked before: "Temperature in Madrid"
Follow-up: "will it rain tomorrow?"
Output: Will it rain tomorrow in Madrid?

Context: {"city": "New York", "metric": "temperature"}
User asked before: "What's the weather in New York?"
Follow-up: "compare it with London"
Output: Compare weather in New York and London

Context: {"city": "Tokyo"}
User asked before: "What's the weather in Tokyo?"
Follow-up: "thanks!"
Output: PASS_THROUGH"""
