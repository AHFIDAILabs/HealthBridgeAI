"""Unit tests for RAGService."""
from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from healthbridgeai.core.models.disease import QueryIntent, RouteResult
from healthbridgeai.core.models.retrieval import RetrievalResult, WebResult
from healthbridgeai.core.services.rag import RAGService
from tests.conftest import _make_chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_route(intent: QueryIntent = QueryIntent.SYMPTOMS) -> RouteResult:
    return RouteResult(
        disease_ids=["tb"],
        disease_confidence=0.9,
        query_intent=intent,
        intent_confidence=0.85,
    )


def _make_svc(mock_llm, mock_vector_store, mock_web_search, disease_registry):
    svc = RAGService(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    # Bypass the FlashRank model download — unit tests verify logic, not re-ranking
    svc._rerank = lambda query, chunks, top_k: chunks[:top_k]
    return svc


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_returns_result(mock_llm, mock_vector_store, mock_web_search, disease_registry):
    svc = _make_svc(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    result = await svc.retrieve("What are TB symptoms?", _make_route(), phone_hash="abc123")

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) > 0
    assert result.best_score > 0


# ── Drug interaction alpha ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drug_interaction_uses_lexical_alpha(
    mock_llm, mock_vector_store, mock_web_search, disease_registry
):
    """DRUG_INTERACTION intent should call hybrid_search with alpha=0.3."""
    svc = _make_svc(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    await svc.retrieve(
        "rifampicin + isoniazid interaction",
        _make_route(QueryIntent.DRUG_INTERACTION),
        phone_hash="abc123",
    )

    call_kwargs = mock_vector_store.hybrid_search.call_args
    alpha = call_kwargs.kwargs.get("alpha")
    assert alpha == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_symptom_query_uses_semantic_alpha(
    mock_llm, mock_vector_store, mock_web_search, disease_registry
):
    """SYMPTOMS intent should call hybrid_search with alpha=0.7 (semantic-heavy)."""
    svc = _make_svc(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    await svc.retrieve(
        "What are the symptoms of TB?",
        _make_route(QueryIntent.SYMPTOMS),
        phone_hash="abc123",
    )

    call_kwargs = mock_vector_store.hybrid_search.call_args
    alpha = call_kwargs.kwargs.get("alpha")
    assert alpha == pytest.approx(0.7)


# ── HyDE fallback ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hyde_triggered_on_low_score(
    mock_llm, mock_vector_store, mock_web_search, disease_registry
):
    """When best chunk score < HYDE_FALLBACK_THRESHOLD, HyDE re-embeds a hypothetical doc."""
    mock_vector_store.hybrid_search.return_value = [_make_chunk(score=0.2)]
    mock_llm.complete.return_value = "A hypothetical document about TB symptoms."
    mock_llm.embed.return_value = [[0.1] * 1024]
    mock_llm.embed_sparse.return_value = [{"indices": [0], "values": [0.5]}]

    svc = _make_svc(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    result = await svc.retrieve("obscure query", _make_route(), phone_hash="abc123")

    assert result.used_hyde is True
    # embed() called at least twice: once for original query, once for HyDE hypothesis
    assert mock_llm.embed.call_count >= 2


# ── Tavily web fallback ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tavily_fallback_on_very_low_score(
    mock_llm, mock_vector_store, mock_web_search, disease_registry
):
    """After HyDE still returns low score, Tavily web search must be called."""
    mock_vector_store.hybrid_search.return_value = [_make_chunk(score=0.2)]
    mock_llm.complete.return_value = "Hypothetical doc."
    mock_llm.embed.return_value = [[0.1] * 1024]
    mock_llm.embed_sparse.return_value = [{"indices": [0], "values": [0.5]}]
    mock_web_search.search.return_value = [
        WebResult(
            title="WHO TB Facts",
            url="https://who.int/tb-facts",
            content="TB is a bacterial disease caused by Mycobacterium tuberculosis.",
            score=0.7,
            domain="who.int",
        )
    ]

    svc = _make_svc(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    result = await svc.retrieve("very obscure query", _make_route(), phone_hash="abc123")

    assert result.used_web_fallback is True
    mock_web_search.search.assert_called_once()
