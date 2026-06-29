"""
Unit tests for SemanticCache.

Pure-function tests (_make_id, _cached_to_meta, _meta_to_cached) run without
any network connection. Adapter-level tests mock the Pinecone client.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from healthbridgeai.core.models.response import CachedResponse
from healthbridgeai.infrastructure.cache.semantic import (
    SemanticCache,
    _cached_to_meta,
    _make_id,
    _meta_to_cached,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

_FUTURE = 9_999_999_999   # far future unix timestamp
_PAST   = 1_000_000_000   # year 2001 — guaranteed expired


def _make_cached(
    confidence: str = "high",
    english_response: str = "TB symptoms include cough and fever.",
    disease_ids: str = "tb",
) -> CachedResponse:
    return CachedResponse(
        english_query="What are the symptoms of TB?",
        disease_ids=disease_ids,
        query_intent="symptoms",
        english_response=english_response,
        sources_json="[]",
        confidence=confidence,
        created_at=1_700_000_000,
        expires_at=_FUTURE,
        hit_count=0,
    )


def _make_match(
    score: float = 0.95,
    expires_at: int = _FUTURE,
    **meta_overrides,
) -> MagicMock:
    m = MagicMock()
    m.id = "cache-deadbeef"
    m.score = score
    m.metadata = {
        "english_query": "What are the symptoms of TB?",
        "disease_ids": "tb",
        "query_intent": "symptoms",
        "english_response": "TB symptoms include cough and fever.",
        "sources_json": "[]",
        "confidence": "high",
        "created_at": 1_700_000_000,
        "expires_at": expires_at,
        "hit_count": 3,
        **meta_overrides,
    }
    return m


def _make_cache() -> tuple[SemanticCache, MagicMock]:
    with patch("healthbridgeai.infrastructure.cache.semantic.Pinecone") as mock_pc:
        mock_index = MagicMock()
        mock_pc.return_value.Index.return_value = mock_index
        cache = SemanticCache()
    cache._index = mock_index
    return cache, mock_index


# ── _make_id ───────────────────────────────────────────────────────────────────

def test_make_id_is_deterministic():
    a = _make_id("What is TB?", "tb")
    b = _make_id("What is TB?", "tb")
    assert a == b


def test_make_id_different_queries_differ():
    a = _make_id("What is TB?", "tb")
    b = _make_id("What is HIV?", "tb")
    assert a != b


def test_make_id_different_diseases_differ():
    a = _make_id("What is TB?", "tb")
    b = _make_id("What is TB?", "hiv")
    assert a != b


def test_make_id_length():
    assert len(_make_id("query", "tb")) == 32


# ── _cached_to_meta ────────────────────────────────────────────────────────────

def test_cached_to_meta_includes_all_fields():
    meta = _cached_to_meta(_make_cached())
    for key in ("english_query", "disease_ids", "query_intent", "english_response",
                "sources_json", "confidence", "created_at", "expires_at", "hit_count"):
        assert key in meta


def test_cached_to_meta_truncates_long_response():
    long_resp = "x" * 10_000
    meta = _cached_to_meta(_make_cached(english_response=long_resp))
    assert len(meta["english_response"]) <= 8_000


def test_cached_to_meta_confidence_preserved():
    meta = _cached_to_meta(_make_cached(confidence="medium"))
    assert meta["confidence"] == "medium"


# ── _meta_to_cached ────────────────────────────────────────────────────────────

def test_meta_to_cached_round_trips():
    original = _make_cached()
    meta = _cached_to_meta(original)
    recovered = _meta_to_cached(meta)
    assert recovered.english_query == original.english_query
    assert recovered.disease_ids == original.disease_ids
    assert recovered.confidence == original.confidence


def test_meta_to_cached_defaults_hit_count_to_zero():
    meta = _cached_to_meta(_make_cached())
    del meta["hit_count"]
    recovered = _meta_to_cached(meta)
    assert recovered.hit_count == 0


def test_meta_to_cached_preserves_hit_count():
    m = _make_match(hit_count=7)
    recovered = _meta_to_cached(m.metadata)
    assert recovered.hit_count == 7


# ── lookup ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lookup_returns_cached_on_hit():
    cache, mock_index = _make_cache()
    result_mock = MagicMock()
    result_mock.matches = [_make_match(score=0.95)]
    mock_index.query.return_value = result_mock

    # Properly close the fire-and-forget coroutine so it doesn't leak
    def _close_coro(coro):
        coro.close()

    with patch("asyncio.ensure_future", side_effect=_close_coro):
        result = await cache.lookup(
            "What are TB symptoms?",
            [0.1] * 1024,
            ["tb"],
        )
    assert result is not None
    assert result.disease_ids == "tb"
    assert result.confidence == "high"


@pytest.mark.asyncio
async def test_lookup_returns_none_score_below_threshold():
    cache, mock_index = _make_cache()
    result_mock = MagicMock()
    result_mock.matches = [_make_match(score=0.85)]   # below default 0.92
    mock_index.query.return_value = result_mock

    result = await cache.lookup("query", [0.1] * 1024, ["tb"])
    assert result is None


@pytest.mark.asyncio
async def test_lookup_returns_none_expired():
    cache, mock_index = _make_cache()
    result_mock = MagicMock()
    result_mock.matches = [_make_match(score=0.97, expires_at=_PAST)]
    mock_index.query.return_value = result_mock

    result = await cache.lookup("query", [0.1] * 1024, ["tb"])
    assert result is None


@pytest.mark.asyncio
async def test_lookup_returns_none_on_empty_matches():
    cache, mock_index = _make_cache()
    result_mock = MagicMock()
    result_mock.matches = []
    mock_index.query.return_value = result_mock

    result = await cache.lookup("query", [0.1] * 1024, ["tb"])
    assert result is None


@pytest.mark.asyncio
async def test_lookup_returns_none_on_pinecone_error():
    cache, mock_index = _make_cache()
    mock_index.query.side_effect = RuntimeError("Pinecone timeout")

    result = await cache.lookup("query", [0.1] * 1024, ["tb"])
    assert result is None   # must never raise


@pytest.mark.asyncio
async def test_lookup_filters_by_disease_ids():
    """Filter is sent to Pinecone with sorted, joined disease_ids."""
    cache, mock_index = _make_cache()
    result_mock = MagicMock()
    result_mock.matches = []
    mock_index.query.return_value = result_mock

    await cache.lookup("query", [0.1] * 1024, ["hiv", "tb"])

    call_kwargs = mock_index.query.call_args.kwargs
    assert call_kwargs["filter"] == {"disease_ids": {"$eq": "hiv,tb"}}


# ── store ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_upserts_to_pinecone():
    cache, mock_index = _make_cache()
    mock_index.upsert.return_value = None

    await cache.store("What is TB?", [0.1] * 1024, _make_cached())
    assert mock_index.upsert.call_count == 1


@pytest.mark.asyncio
async def test_store_skips_low_confidence():
    """Safety-net: low-confidence responses must never be cached."""
    cache, mock_index = _make_cache()

    await cache.store("What is TB?", [0.1] * 1024, _make_cached(confidence="low"))
    mock_index.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_store_stores_to_cache_namespace():
    cache, mock_index = _make_cache()
    mock_index.upsert.return_value = None

    await cache.store("What is TB?", [0.1] * 1024, _make_cached())
    call_kwargs = mock_index.upsert.call_args.kwargs
    assert call_kwargs.get("namespace") == "response-cache"


@pytest.mark.asyncio
async def test_store_swallows_pinecone_error():
    """Store failures must never propagate — cache is best-effort."""
    cache, mock_index = _make_cache()
    mock_index.upsert.side_effect = RuntimeError("write failed")

    await cache.store("What is TB?", [0.1] * 1024, _make_cached())  # must not raise


# ── invalidate ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalidate_deletes_matching_ids():
    cache, mock_index = _make_cache()

    query_result = MagicMock()
    m1, m2 = MagicMock(), MagicMock()
    m1.id, m2.id = "id-1", "id-2"
    query_result.matches = [m1, m2]
    mock_index.query.return_value = query_result
    mock_index.delete.return_value = None

    count = await cache.invalidate("tb")

    assert count == 2
    mock_index.delete.assert_called_once()
    deleted_ids = mock_index.delete.call_args.kwargs.get("ids") or mock_index.delete.call_args.args[0]
    assert set(deleted_ids) == {"id-1", "id-2"}


@pytest.mark.asyncio
async def test_invalidate_skips_delete_when_no_matches():
    cache, mock_index = _make_cache()
    query_result = MagicMock()
    query_result.matches = []
    mock_index.query.return_value = query_result

    count = await cache.invalidate("tb")

    assert count == 0
    mock_index.delete.assert_not_called()


@pytest.mark.asyncio
async def test_invalidate_returns_zero_on_error():
    cache, mock_index = _make_cache()
    mock_index.query.side_effect = RuntimeError("Pinecone error")

    count = await cache.invalidate("tb")
    assert count == 0   # must never raise
