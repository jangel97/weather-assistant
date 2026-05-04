import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from framework.clients import get_llm_client  # noqa: E402
from framework.pipeline.logging import log_event, setup_logging  # noqa: E402
from weather_assistant.routers import chat  # noqa: E402

# Import tool modules so they register with the registry
from weather_assistant.tools import weather  # noqa: E402, F401

AGENT_PORT = 8002

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_event(logger, logging.INFO, event="startup")

    try:
        llm = get_llm_client()
        log_event(
            logger, logging.INFO,
            event="llm_initialized", provider=llm.provider, model=str(llm.model),
        )
    except Exception as exc:
        log_event(
            logger, logging.ERROR, event="llm_init_failed", error=str(exc),
        )
        raise

    yield

    log_event(logger, logging.INFO, event="shutdown")


app = FastAPI(
    title="Weather Assistant",
    description="Weather Q&A agent powered by small LLMs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "weather_assistant.main:app", host="0.0.0.0", port=AGENT_PORT, reload=True,
    )
