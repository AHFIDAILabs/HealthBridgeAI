"""
HealthBridgeAI — Processor service.

Receives inbound WhatsApp messages via Pub/Sub push, runs the full
RAG + LLM pipeline, and sends the response back via WhatChimp.

Entry point:  uvicorn healthbridgeai.processor.main:app
Cloud Run:    internal-only service, no public traffic
"""
from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response

from healthbridgeai.config import get_disease_registry, settings
from healthbridgeai.core.models.message import InboundMessage, MessageType
from healthbridgeai.core.services import (
    LanguageService,
    MessagePipeline,
    RAGService,
    ResponseGenerator,
    RouterService,
)
from healthbridgeai.infrastructure.audio.synthesizer import AudioSynthesizer
from healthbridgeai.infrastructure.audio.transcriber import AudioTranscriber
from healthbridgeai.infrastructure.cache import SemanticCache
from healthbridgeai.infrastructure.llm import OpenRouterClient
from healthbridgeai.infrastructure.messaging import WhatChampAdapter
from healthbridgeai.infrastructure.search import TavilyAdapter
from healthbridgeai.infrastructure.storage import (
    FirestoreConversationStore,
    FirestoreUserStore,
    GCSStorage,
)
from healthbridgeai.infrastructure.vector_store import PineconeAdapter

log = structlog.get_logger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    log.info("healthbridge.startup", service="processor")

    # Infrastructure adapters
    llm = OpenRouterClient()
    vector_store = PineconeAdapter()
    web_search = TavilyAdapter()
    user_store = FirestoreUserStore()
    conv_store = FirestoreConversationStore()
    cache = SemanticCache()
    messaging = WhatChampAdapter()
    gcs = GCSStorage()
    registry = get_disease_registry()

    # Core services
    lang_svc = LanguageService(llm)
    router_svc = RouterService(llm, registry)
    rag_svc = RAGService(llm, vector_store, web_search, registry)
    generator = ResponseGenerator(llm)
    pipeline = MessagePipeline(
        language_svc=lang_svc,
        router_svc=router_svc,
        rag_svc=rag_svc,
        generator=generator,
        llm=llm,
        user_store=user_store,
        conv_store=conv_store,
        cache=cache,
    )

    # Audio (optional — only used when user.audio_mode is True)
    synthesizer = AudioSynthesizer()
    transcriber = AudioTranscriber()

    app.state.pipeline = pipeline
    app.state.messaging = messaging
    app.state.gcs = gcs
    app.state.synthesizer = synthesizer
    app.state.transcriber = transcriber
    app.state.user_store = user_store

    log.info("healthbridge.ready", service="processor")
    yield
    log.info("healthbridge.shutdown", service="processor")


app = FastAPI(
    title="HealthBridgeAI Processor",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/process", status_code=204)
async def process(request: Request) -> Response:
    """
    Pub/Sub push endpoint.

    Pub/Sub delivers messages as:
        {"message": {"data": "<base64>", "messageId": "...", "attributes": {...}},
         "subscription": "..."}

    Returns 204 to acknowledge (any 2xx works). Non-2xx causes Pub/Sub to retry.
    We always return 204 — errors are logged but never trigger retries, which
    would produce duplicate WhatsApp messages.
    """
    pipeline: MessagePipeline = request.app.state.pipeline
    messaging: WhatChampAdapter = request.app.state.messaging

    try:
        envelope = await request.json()
        encoded_data = envelope["message"]["data"]
        raw = json.loads(base64.b64decode(encoded_data))
    except Exception as exc:
        log.error("processor.parse_envelope_failed", error=str(exc))
        return Response(status_code=204)  # ack; malformed messages won't self-heal on retry

    # ── Deserialize InboundMessage ────────────────────────────────────────────
    try:
        message = InboundMessage(**raw)
    except Exception as exc:
        log.error("processor.deserialize_failed", error=str(exc))
        return Response(status_code=204)

    log.info(
        "processor.received",
        message_id=message.message_id,
        message_type=message.type.value,
    )

    # ── Audio transcription (if voice message) ────────────────────────────────
    if message.type == MessageType.AUDIO and message.media:
        try:
            audio_bytes = await messaging.download_media(message.media.media_id)
            lang_hint = await _get_user_lang(request, message.from_number)
            text, detected_lang = await request.app.state.transcriber.transcribe(
                audio_bytes, lang_hint
            )
            # Inject transcription as text into a new message copy
            message = message.model_copy(update={"type": MessageType.TEXT, "text": text})
            log.info("processor.transcribed", lang=detected_lang, length=len(text))
        except Exception as exc:
            log.error("processor.transcription_failed", error=str(exc))
            await _safe_send(messaging, message.from_number,
                             "Sorry, I couldn't understand the audio. Please send a text message.")
            return Response(status_code=204)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        bot = await pipeline.process(message)
    except Exception as exc:
        log.error("processor.pipeline_failed", error=str(exc))
        await _safe_send(messaging, message.from_number,
                         "⚠️ Something went wrong. Please try again in a moment.")
        return Response(status_code=204)

    # ── Send response ─────────────────────────────────────────────────────────
    try:
        await messaging.send_text(message.from_number, bot.text)
        log.info(
            "processor.sent",
            message_id=message.message_id,
            confidence=bot.confidence,
            cache_hit=bot.cache_hit,
        )
    except Exception as exc:
        log.error("processor.send_failed", error=str(exc))

    return Response(status_code=204)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_user_lang(request: Request, phone_number: str) -> str:
    """Return the user's stored language preference, or 'auto'."""
    try:
        user = await request.app.state.user_store.get_user(phone_number)
        return user.language_code if user else "auto"
    except Exception:
        return "auto"


async def _safe_send(messaging: WhatChampAdapter, to: str, text: str) -> None:
    try:
        await messaging.send_text(to, text)
    except Exception as exc:
        log.error("processor.safe_send_failed", error=str(exc))
