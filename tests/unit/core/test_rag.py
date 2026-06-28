"""Unit tests for RAGService."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from healthbridgeai.core.models.retrieval import QueryIntent, RetrievalResult
from healthbridgeai.core.services.rag import RAGService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(mock_llm, mock_vector_store, mock_web_search, disease_registry):
    return RAGService(mock_llm, mock_vector_store, mock_web_search, disease_registry)


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_returns_result(
    mock_llm, mock_vector_store, mock_web_search, disease_registry, route_result
):
    svc = _make_service(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    result = await svc.retrieve("What are TB symptoms?", route_result, phone_hash="abc123")

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) > 0
    assert result.best_score > 0


# ── Drug interaction alpha ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drug_interaction_uses_lexical_alpha(
    mock_llm, mock_vector_store, mock_web_search, disease_registry, route_result
):
    """DRUG_INTERACTION intent should call hybrid_search with alpha=0.3 (lexical-heavy)."""
    route_result.intent = QueryIntent.DRUG_INTERACTION
    svc = _make_service(mock_llm, mock_vector_store, mock_web_search, disease_registry)

    await svc.retrieve("rifampicin + isoniazid interaction", route_result, phone_hash="abc123")

    call_kwargs = mock_vector_store.hybrid_search.call_args
    alpha = call_kwargs.kwargs.get("alpha") or call_kwargs.args[5] if call_kwargs.args else None
    assert alpha == pytest.approx(0.3)


# ── HyDE fallback ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hyde_triggered_on_low_score(
    mock_llm, mock_vector_store, mock_web_search, disease_registry, route_result
):
    """When best chunk score < HYDE_FALLBACK_THRESHOLD, HyDE should re-embed."""
    from tests.conftest import _make_chunk

    mock_vector_store.hybrid_search.return_value = [_make_chunk(score=0.2)]
    mock_llm.complete.return_value = "A hypothetical document about TB symptoms."
    mock_llm.embed.return_value = [[0.1] * 1024]
    mock_llm.embed_sparse.return_value = [{"indices": [0], "values": [0.5]}]

    svc = _make_service(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    result = await svc.retrieve("obscure query", route_result, phone_hash="abc123")

    assert result.used_hyde is True
    # embed should have been called at least twice (original + HyDE)
    assert mock_llm.embed.call_count >= 2


# ── Tavily web fallback ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tavily_fallback_on_very_low_score(
    mock_llm, mock_vector_store, mock_web_search, disease_registry, route_result
):
    """After HyDE still returns low score, Tavily web search should be called."""
    from healthbridgeai.core.models.retrieval import Source, WebResult
    from tests.conftest import _make_chunk

    mock_vector_store.hybrid_search.return_value = [_make_chunk(score=0.2)]
    mock_llm.complete.return_value = "Hypothetical doc."
    mock_llm.embed.return_value = [[0.1] * 1024]
    mock_llm.embed_sparse.return_value = [{"indices": [0], "values": [0.5]}]
    mock_web_search.search.return_value = [
        WebResult(
            title="WHO TB Facts",
            url="https://who.int/tb-facts",
            snippet="TB is a bacterial disease.",
            score=0.7,
            source=Source(
                title="WHO TB Facts",
                url="https://who.int/tb-facts",
                publication_date=None,
                publisher="WHO",
                is_web_result=True,
            ),
        )
    ]

    svc = _make_service(mock_llm, mock_vector_store, mock_web_search, disease_registry)
    result = await svc.retrieve("very obscure query", route_result, phone_hash="abc123")

    assert result.used_web_search is True
    mock_web_search.search.assert_called_once()
