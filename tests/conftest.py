"""
Shared pytest fixtures — AsyncMock implementations of all 6 port interfaces.

All adapters return sensible defaults so tests only need to override the
specific call(s) relevant to what they're testing.
"""
from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

# Inject dummy env vars so Settings() doesn't raise ValidationError in unit tests.
# setdefault keeps real values intact if a .env file already provides them.
_TEST_ENV_DEFAULTS = {
    "WHATCHAMP_API_KEY": "test-api-key",
    "WHATCHAMP_PHONE_NUMBER": "+2340000000000",
    "WHATCHAMP_PHONE_NUMBER_ID": "000000000000",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    "PINECONE_API_KEY": "test-pinecone-key",
    "TAVILY_API_KEY": "test-tavily-key",
    "GCP_PROJECT_ID": "test-project-id",
}
for _k, _v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

from healthbridgeai.core.models.disease import (
    DiseaseConfig,
    DiseaseRegistry,
    QueryIntent,
    RouteResult,
)
from healthbridgeai.core.models.message import InboundMessage, MessageType
from healthbridgeai.core.models.response import BotResponse, CachedResponse, LLMResponse
from healthbridgeai.core.models.retrieval import Chunk, RetrievalResult, Source, WebResult
from healthbridgeai.core.models.user import RateLimit, User


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(text: str = "TB is caused by Mycobacterium tuberculosis.", score: float = 0.8) -> Chunk:
    return Chunk(
        text=text,
        score=score,
        disease="tb",
        doc_id="abc123",
        source=Source(
            index=1,
            title="WHO TB Guidelines 2023",
            url="https://who.int/tb",
            domain="who.int",
            source_type="guideline",
        ),
        chunk_type="treatment",
    )


# ── Fixtures: ports ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm() -> AsyncMock:
    """ILLMClient mock with sensible defaults."""
    llm = AsyncMock()
    llm.embed.return_value = [[0.1] * 1024]
    llm.embed_sparse.return_value = [{"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}]
    llm.complete.return_value = "This is a test response."
    llm.structured.return_value = MagicMock()
    return llm


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """IVectorStore mock returning one chunk by default."""
    vs = AsyncMock()
    vs.hybrid_search.return_value = [_make_chunk()]
    vs.upsert_chunks.return_value = 1
    vs.delete_namespace.return_value = None
    vs.describe_index.return_value = {"total_vector_count": 100, "namespaces": {}}
    return vs


@pytest.fixture
def mock_web_search() -> AsyncMock:
    """IWebSearch mock returning empty results by default."""
    ws = AsyncMock()
    ws.search.return_value = []
    return ws


@pytest.fixture
def mock_user_store() -> AsyncMock:
    """IUserStore mock with a default user and a passing rate limit."""
    now = int(time.time())
    store = AsyncMock()
    store.get_user.return_value = User(
        phone_number="+2348012345678",
        language_code="en",
        audio_mode=False,
        created_at=now,
        last_seen_at=now,
    )
    store.upsert_user.return_value = None
    store.check_rate_limit.return_value = RateLimit(
        phone_hash="abc123testxx",
        window_start=now,
        message_count=1,
        is_blocked=False,
    )
    return store


@pytest.fixture
def mock_conv_store() -> AsyncMock:
    """IConversationStore mock returning empty history."""
    store = AsyncMock()
    store.get_turns.return_value = []
    store.save_turn.return_value = None
    return store


@pytest.fixture
def mock_cache() -> AsyncMock:
    """ISemanticCache mock — cache miss by default."""
    cache = AsyncMock()
    cache.lookup.return_value = None
    cache.store.return_value = None
    cache.invalidate.return_value = 0
    return cache


@pytest.fixture
def mock_messaging() -> AsyncMock:
    msg = AsyncMock()
    msg.send_text.return_value = "msg-id-001"
    msg.send_audio.return_value = "msg-id-002"
    msg.download_media.return_value = b"fake-audio-bytes"
    msg.validate_webhook.return_value = True
    msg.parse_webhook.return_value = None
    return msg


# ── Fixtures: domain objects ──────────────────────────────────────────────────

@pytest.fixture
def tb_disease() -> DiseaseConfig:
    return DiseaseConfig(
        name="Tuberculosis",
        short_name="TB",
        enabled=True,
        pinecone_namespace="tb",
        kb_gcs_path="knowledge-bases/tb/TB_knowledge_base.zip",
        search_domains=["who.int", "cdc.gov"],
        aliases=["tuberculosis", "consumption"],
        emergency_keywords=["coughing blood", "haemoptysis"],
        system_prompt_extra="Focus on TB guidelines for Nigeria.",
    )


@pytest.fixture
def disease_registry(tb_disease: DiseaseConfig) -> DiseaseRegistry:
    return DiseaseRegistry(configs={"tb": tb_disease})


@pytest.fixture
def inbound_message() -> InboundMessage:
    return InboundMessage(
        message_id="wamid.test001",
        from_number="+2348012345678",
        to_number="+2348000000000",
        type=MessageType.TEXT,
        text="What are the symptoms of TB?",
        timestamp=1700000000,
    )


@pytest.fixture
def route_result() -> RouteResult:
    return RouteResult(
        disease_ids=["tb"],
        disease_confidence=0.9,
        query_intent=QueryIntent.SYMPTOMS,
        intent_confidence=0.85,
        is_emergency=False,
        is_personal=False,
    )


@pytest.fixture
def retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunks=[_make_chunk()],
        best_score=0.8,
        used_hyde=False,
        used_web_fallback=False,
    )


@pytest.fixture
def llm_response() -> LLMResponse:
    return LLMResponse(
        answer="TB symptoms include persistent cough, fever, and night sweats.",
        confidence="high",
        needs_professional=False,
        caveat=None,
        sources_used=[1],
    )
