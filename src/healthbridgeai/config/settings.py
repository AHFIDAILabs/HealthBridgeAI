"""
Single source of truth for all configuration.
Loaded once at startup; missing required values raise ValidationError immediately.
Replaces: config.py + modules/config_manager.py (both deleted).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── WhatChimp (WhatsApp BSP) ──────────────────────────────────────────────
    WHATCHAMP_API_KEY: str
    WHATCHAMP_API_URL: str = "https://app.whatchimp.com/api/v1"
    WHATCHAMP_PHONE_NUMBER: str          # E.164 sender number, e.g. +2348012345678
    WHATCHAMP_PHONE_NUMBER_ID: str       # Meta phone_number_id (required in all API calls)
    WHATCHAMP_WEBHOOK_SECRET: str = ""   # Optional — Meta X-Hub-Signature-256 app secret

    # ── LLM (OpenRouter) ─────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str
    LLM_PRIMARY_MODEL: str = "anthropic/claude-haiku-4-5"
    LLM_HEAVY_MODEL: str = "anthropic/claude-sonnet-4-6"
    LLM_ROUTER_MODEL: str = "anthropic/claude-haiku-4-5"
    LLM_TIMEOUT_SECONDS: int = 30
    MAX_USER_INPUT_CHARS: int = 2000

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "healthbridge"
    PINECONE_REGION: str = "us-east-1"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    PINECONE_INDEX_DIMENSION: int = 1024

    # ── Web Search (Tavily) ───────────────────────────────────────────────────
    TAVILY_API_KEY: str

    # ── Audio (all optional — fallback chain activates when empty) ────────────
    YARNGPT_API_KEY: str = ""
    HUGGINGFACE_TOKEN: str = ""
    NATLAS_API_KEY: str = ""

    # ── GCP ───────────────────────────────────────────────────────────────────
    GCP_PROJECT_ID: str
    GCS_BUCKET_NAME: str = "healthbridge-assets"
    FIRESTORE_DATABASE: str = "(default)"

    # ── Pub/Sub ───────────────────────────────────────────────────────────────
    PUBSUB_TOPIC_INBOUND: str = "healthbridge-inbound"
    PUBSUB_SUBSCRIPTION_INBOUND: str = "healthbridge-inbound-sub"

    # ── Behaviour ─────────────────────────────────────────────────────────────
    SUPPORTED_LANGUAGES: list[str] = ["en", "yo", "ig", "ha", "pidgin"]
    RATE_LIMIT_MESSAGES_PER_MINUTE: int = 20
    CONVERSATION_HISTORY_TURNS: int = 5

    # ── Semantic Cache ────────────────────────────────────────────────────────
    # Cosine similarity threshold — must be >= this to serve a cached response
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    CACHE_TTL_DAYS: int = 7

    # ── RAG Retrieval ─────────────────────────────────────────────────────────
    # Min Pinecone score after re-ranking before triggering web search fallback
    MIN_RETRIEVAL_SCORE_DEFAULT: float = 0.6
    # If best score < this, attempt HyDE before falling back to web search
    HYDE_FALLBACK_THRESHOLD: float = 0.5


# Module-level singleton — import this everywhere:
#   from healthbridgeai.config.settings import settings
settings = Settings()  # type: ignore[call-arg]
