"""
Single source of truth for all configuration.
Loaded once at startup; missing required values raise ValidationError immediately.
Replaces: config.py + modules/config_manager.py (both deleted).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Twilio ────────────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_WHATSAPP_FROM: str
    TWILIO_WEBHOOK_URL: str

    # ── LLM ───────────────────────────────────────────────────────────────────
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

    # ── Web Search ────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str

    # ── Audio (optional — empty string activates fallback chain) ──────────────
    YARNGPT_API_KEY: str = ""
    HUGGINGFACE_TOKEN: str = ""
    NATLAS_API_KEY: str = ""

    # ── GCP ───────────────────────────────────────────────────────────────────
    GCP_PROJECT_ID: str
    GCS_BUCKET_NAME: str = "healthbridge-assets"
    FIRESTORE_DATABASE: str = "(default)"

    # ── Behaviour ─────────────────────────────────────────────────────────────
    SUPPORTED_LANGUAGES: list[str] = ["en", "yo", "ig", "ha", "pidgin"]
    RATE_LIMIT_MESSAGES_PER_MINUTE: int = 20
    CONVERSATION_HISTORY_TURNS: int = 5

    # ── Semantic Cache ────────────────────────────────────────────────────────
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    CACHE_TTL_DAYS: int = 7

    # ── RAG ───────────────────────────────────────────────────────────────────
    MIN_RETRIEVAL_SCORE_DEFAULT: float = 0.6
    HYDE_FALLBACK_THRESHOLD: float = 0.5


# Module-level singleton — import this everywhere:
#   from healthbridgeai.config.settings import settings
settings = Settings()  # type: ignore[call-arg]
