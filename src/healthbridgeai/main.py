"""
HealthBridgeAI — Webhook service app factory.

This service receives inbound WhatChimp webhooks, validates signatures,
and publishes messages to Pub/Sub. Heavy processing (RAG, LLM, WhatsApp reply)
happens in the separate processor service.

Entry point:  uvicorn healthbridgeai.main:app
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from google.cloud import pubsub_v1

from healthbridgeai.api.health import router as health_router
from healthbridgeai.api.webhook import router as webhook_router
from healthbridgeai.config import settings
from healthbridgeai.infrastructure.messaging import WhatChampAdapter
from healthbridgeai.infrastructure.vector_store import PineconeAdapter


def _configure_logging() -> None:
    """JSON structured logging — compatible with GCP Cloud Logging."""
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )
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
    log = structlog.get_logger(__name__)
    log.info("healthbridge.startup", service="webhook")

    # Messaging adapter (WhatChimp)
    app.state.messaging = WhatChampAdapter()

    # Pinecone — used only for /ready health check in this service
    app.state.pinecone = PineconeAdapter()

    # Pub/Sub publisher — the processor service subscribes on the other end
    publisher = pubsub_v1.PublisherClient()
    app.state.publisher = publisher
    app.state.topic_path = publisher.topic_path(
        settings.GCP_PROJECT_ID,
        settings.PUBSUB_TOPIC_INBOUND,
    )

    log.info("healthbridge.ready", topic=settings.PUBSUB_TOPIC_INBOUND)
    yield
    log.info("healthbridge.shutdown", service="webhook")


def create_app() -> FastAPI:
    return FastAPI(
        title="HealthBridgeAI",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )


app = create_app()
app.include_router(webhook_router)
app.include_router(health_router)
