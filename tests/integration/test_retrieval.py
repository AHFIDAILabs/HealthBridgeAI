"""
Integration tests — real Pinecone + real BGE-M3 embeddings.

These tests hit the live Pinecone index in the  tb  namespace and require:
  - PINECONE_API_KEY, PINECONE_INDEX_NAME set in .env
  - The TB knowledge base already indexed (run scripts/populate_kb.py first)

Run selectively — they are slow and consume Pinecone quota:
    pytest tests/integration/ -v -m integration

Skip automatically in CI unless INTEGRATION=1 is set:
    INTEGRATION=1 pytest tests/integration/ -v
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio

# Skip entire module if not in integration mode
if not os.getenv("INTEGRATION"):
    pytest.skip("Set INTEGRATION=1 to run integration tests", allow_module_level=True)


pytestmark = pytest.mark.integration


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def vector_store():
    from healthbridgeai.infrastructure.vector_store import PineconeAdapter
    return PineconeAdapter()


@pytest_asyncio.fixture(scope="module")
async def llm():
    from healthbridgeai.infrastructure.llm import OpenRouterClient
    return OpenRouterClient()


# ── Tests: PineconeAdapter ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pinecone_index_reachable(vector_store):
    stats = await vector_store.describe_index()
    assert "total_vector_count" in stats or "namespaces" in stats


@pytest.mark.asyncio
async def test_hybrid_search_returns_tb_chunks(vector_store, llm):
    query = "What are the symptoms of tuberculosis?"
    dense = await llm.embed([query])
    sparse = await llm.embed_sparse([query])

    chunks = await vector_store.hybrid_search(
        query_text=query,
        query_embedding=dense[0],
        sparse_vector=sparse[0],
        namespace="tb",
        top_k=5,
        alpha=0.7,
    )

    assert len(chunks) > 0, "Should return at least one chunk"
    assert all(c.disease_id == "tb" for c in chunks)
    assert all(0.0 <= c.score <= 1.0 for c in chunks)
    assert all(len(c.text) > 50 for c in chunks)


@pytest.mark.asyncio
async def test_hybrid_search_drug_alpha(vector_store, llm):
    """Drug queries use alpha=0.3 (lexical-heavy) — should still return results."""
    query = "rifampicin isoniazid interaction"
    dense = await llm.embed([query])
    sparse = await llm.embed_sparse([query])

    chunks = await vector_store.hybrid_search(
        query_text=query,
        query_embedding=dense[0],
        sparse_vector=sparse[0],
        namespace="tb",
        top_k=5,
        alpha=0.3,
    )
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_embedding_dimension(llm):
    vecs = await llm.embed(["test sentence"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 1024


@pytest.mark.asyncio
async def test_sparse_embedding_structure(llm):
    sparse = await llm.embed_sparse(["rifampicin"])
    assert len(sparse) == 1
    sv = sparse[0]
    assert "indices" in sv and "values" in sv
    assert len(sv["indices"]) == len(sv["values"])
    assert len(sv["indices"]) > 0


# ── Tests: full RAG retrieval ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_service_end_to_end(vector_store, llm):
    from unittest.mock import AsyncMock

    from healthbridgeai.config import get_disease_registry
    from healthbridgeai.core.models.retrieval import QueryIntent, RouteResult
    from healthbridgeai.core.services.rag import RAGService
    from healthbridgeai.infrastructure.search import TavilyAdapter

    registry = get_disease_registry()
    tb_disease = registry.get_disease("tb")

    route = RouteResult(
        disease_ids=["tb"],
        diseases=[tb_disease],
        intent=QueryIntent.SYMPTOM_QUERY,
        is_emergency=False,
        is_personal=False,
        confidence=0.9,
        raw_text="What are the symptoms of TB?",
    )

    try:
        web_search = TavilyAdapter()
    except Exception:
        web_search = AsyncMock()
        web_search.search.return_value = []

    svc = RAGService(llm, vector_store, web_search, registry)
    result = await svc.retrieve("What are the symptoms of TB?", route, phone_hash="inttest01")

    assert len(result.chunks) > 0
    assert result.best_score > 0.0
    assert result.best_score > 0.3, f"Score too low ({result.best_score}) — is the TB namespace populated?"
