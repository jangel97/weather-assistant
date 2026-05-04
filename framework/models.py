import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from framework.constants import (
    MAX_HISTORY_MESSAGE_LENGTH,
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_LENGTH,
)

_max_message_length = int(
    os.environ.get("AGENT_MAX_MESSAGE_LENGTH", MAX_MESSAGE_LENGTH)
)
_max_history_messages = int(
    os.environ.get("AGENT_MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES)
)
_max_history_message_length = int(
    os.environ.get("AGENT_MAX_HISTORY_MESSAGE_LENGTH", MAX_HISTORY_MESSAGE_LENGTH)
)


class Message(BaseModel):
    """A single message in the conversation history."""

    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(
        ..., description="Message content", max_length=_max_history_message_length
    )


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    message: str = Field(
        ...,
        description="The user's question",
        min_length=1,
        max_length=_max_message_length,
    )
    history: Optional[List[Message]] = Field(
        default=None,
        description="Previous conversation messages for multi-turn context",
        max_length=_max_history_messages,
    )


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    message: str = Field(..., description="The assistant's response")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tools that were called to answer the question",
    )
    trace: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured trace with per-layer timing and decisions",
    )
