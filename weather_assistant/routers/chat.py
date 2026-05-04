"""Chat endpoints for the weather assistant."""

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from weather_assistant.entities import CITIES
from weather_assistant.pipeline_config import get_pipeline_config
from framework.clients import get_llm_client
from framework.models import ChatRequest, ChatResponse
from framework.runner import AgentRunner

router = APIRouter(tags=["weather"])


@router.post("/weather/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a chat message."""
    runner = AgentRunner(get_llm_client(), get_pipeline_config(), entities=CITIES)
    return await runner.chat(request)


@router.post("/weather/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming variant of the chat endpoint."""
    runner = AgentRunner(get_llm_client(), get_pipeline_config(), entities=CITIES)
    return StreamingResponse(
        runner.stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
