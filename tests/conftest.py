"""
Shared pytest fixtures — AsyncMock implementations of all 6 port interfaces.

All adapters return sensible defaults so tests only need to override the
specific call(s) relevant to what they're testing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from healthbridgeai.core.models.disease import DiseaseConfig, DiseaseRegistry
from healthbridgeai.core.models.message import InboundMessage, MessageType
from healthbridgeai.core.models.response import BotResponse, LLMResponse
from healthbridgeai.core.models.retrieval import Chunk, RetrievalResult, RouteResult, Source
from healthbridgeai.core.models.user import RateLimit, User


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dense_vec(val: float = 0.1, dim: int = 1024) -> list[float]:
    return [val] * dim


def _sparse_vec(n: int = 3) -> dict:
    return {"indices": list(range(n)), "values": [0.5] * n}


def _make_chunk(text: str = "TB is caused by Mycobacterium tuberculosis.", score: float = 0.8) -> Chunk:
    return Chunk(
        chunk_id="abc123",
        text=text,
        source=Source(
            title="WHO TB Guidelines 2023",
            url="https://who.int/tb",
            publication_date="2023-01-01",
            publisher="WHO",
            is_web_result=False,
        ),
        disease_id="tb",
        chunk_type="knowledge",
        embedding=_dense_vec(0.5),
        sparse_embedding=_sparse_vec(),
        score=score,
    )


# ── Fixtures: ports ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm() -> AsyncMock:
    """ILLMClient mock with sensible defaults."""
    llm = AsyncMock()
    llm.embed.return_value = [_dense_vec()]
    llm.embed_sparse.return_value = [_sparse_vec()]
    llm.complete.return_value = "This is a test response."
    # structured() returns are set per-test via side_effect or return_value
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
    """IUserStore mock with a default user and pass-through rate limit."""
    store = AsyncMock()
    store.get_user.return_value = User(
        phone_number="+2348012345678",
        language_code="en",
        audio_mode=False,
    )
    store.upsert_user.return_value = None
    store.check_rate_limit.return_value = RateLimit(
        allowed=True, current_count=1, limit=20, window_seconds=60
    )
    return store


@pytest.fixture
def mock_conv_store() -> AsyncMock:
    """IConversationStore mock returning empty history."""
    store = AsyncMock()
    store.get_turns.return_value = []
    store.save_turns.return_value = None
    return store


@pytest.fixture
def mock_cache() -> AsyncMock:
    """ISemanticCache mock — miss by default."""
    cache = AsyncMock()
    cache.lookup.return_value = None   # cache miss
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
        id="tb",
        name="Tuberculosis",
        enabled=True,
        pinecone_namespace="tb",
        tavily_domains=["who.int", "cdc.gov"],
        llm_system_addendum="Focus on TB guidelines for Nigeria.",
        aliases=["tuberculosis", "consumption"],
        emergency_keywords=["coughing blood", "haemoptysis"],
    )


@pytest.fixture
def disease_registry(tb_disease: DiseaseConfig) -> DiseaseRegistry:
    return DiseaseRegistry(diseases={"tb": tb_disease})


@pytest.fixture
def inbound_message() -> InboundMessage:
    return InboundMessage(
        message_id="wamid.test001",
        from_number="+2348012345678",
        type=MessageType.TEXT,
        text="What are the symptoms of TB?",
        timestamp=1700000000,
    )


@pytest.fixture
def route_result(tb_disease: DiseaseConfig) -> RouteResult:
    from healthbridgeai.core.models.retrieval import QueryIntent

    return RouteResult(
        disease_ids=["tb"],
        diseases=[tb_disease],
        intent=QueryIntent.SYMPTOM_QUERY,
        is_emergency=False,
        is_personal=False,
        confidence=0.9,
        raw_text="What are the symptoms of TB?",
    )


@pytest.fixture
def retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunks=[_make_chunk()],
        best_score=0.8,
        used_hyde=False,
        used_web_search=False,
        query_text="What are the symptoms of TB?",
    )


@pytest.fixture
def llm_response() -> LLMResponse:
    return LLMResponse(
        answer="TB symptoms include persistent cough, fever, and night sweats.",
        confidence="high",
        needs_professional=False,
        caveat="",
        sources_used=[1],
    )
