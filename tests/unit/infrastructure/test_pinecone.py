"""
Unit tests for PineconeAdapter.

Pure-function tests (_hybrid_convex_scale, _match_to_chunk) run without any
network connection. Adapter-level tests mock the Pinecone client to avoid
requiring a live index.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbridgeai.core.exceptions import RetrievalError
from healthbridgeai.core.models.retrieval import Chunk
from healthbridgeai.infrastructure.vector_store.pinecone import (
    PineconeAdapter,
    _hybrid_convex_scale,
    _match_to_chunk,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_match(
    id: str = "chunk-001",
    score: float = 0.85,
    metadata: dict | None = None,
) -> MagicMock:
    m = MagicMock()
    m.id = id
    m.score = score
    m.metadata = metadata if metadata is not None else {
        "text": "TB is caused by Mycobacterium tuberculosis.",
        "disease": "tb",
        "doc_id": "who-tb-2022",
        "source_title": "WHO TB Guidelines 2022",
        "source_url": "https://who.int/tb",
        "source_domain": "who.int",
        "source_type": "guideline",
        "source_index": 1,
        "source_section": "Chapter 4",
        "source_page_number": 45,
        "chunk_type": "treatment",
        "chunk_index": 12,
        "language": "en",
    }
    return m


def _make_adapter() -> tuple[PineconeAdapter, MagicMock]:
    """Return a PineconeAdapter with a mocked Pinecone index."""
    with patch("healthbridgeai.infrastructure.vector_store.pinecone.Pinecone") as mock_pc:
        mock_index = MagicMock()
        mock_pc.return_value.Index.return_value = mock_index
        adapter = PineconeAdapter()
    adapter._index = mock_index   # keep reference for assertion
    return adapter, mock_index


# ── _hybrid_convex_scale ──────────────────────────────────────────────────────

def test_hybrid_convex_scale_alpha_07():
    dense = [1.0, 2.0, 3.0]
    sparse = {"indices": [0, 1], "values": [0.5, 0.4]}
    sd, ss = _hybrid_convex_scale(dense, sparse, alpha=0.7)

    assert sd == pytest.approx([0.7, 1.4, 2.1])
    assert ss["values"] == pytest.approx([0.15, 0.12])  # 0.5×0.3, 0.4×0.3
    assert ss["indices"] == [0, 1]   # indices unchanged


def test_hybrid_convex_scale_alpha_03():
    dense = [1.0, 0.0]
    sparse = {"indices": [0], "values": [1.0]}
    sd, ss = _hybrid_convex_scale(dense, sparse, alpha=0.3)

    assert sd == pytest.approx([0.3, 0.0])
    assert ss["values"] == pytest.approx([0.7])


def test_hybrid_convex_scale_alpha_1_all_dense():
    dense = [1.0, 1.0]
    sparse = {"indices": [0], "values": [0.8]}
    sd, ss = _hybrid_convex_scale(dense, sparse, alpha=1.0)

    assert sd == pytest.approx([1.0, 1.0])
    assert ss["values"] == pytest.approx([0.0])


def test_hybrid_convex_scale_alpha_0_all_sparse():
    dense = [1.0, 1.0]
    sparse = {"indices": [0, 1], "values": [0.6, 0.4]}
    sd, ss = _hybrid_convex_scale(dense, sparse, alpha=0.0)

    assert sd == pytest.approx([0.0, 0.0])
    assert ss["values"] == pytest.approx([0.6, 0.4])


def test_hybrid_convex_scale_preserves_indices():
    sparse = {"indices": [5, 42, 100], "values": [0.1, 0.2, 0.3]}
    _, ss = _hybrid_convex_scale([0.0], sparse, alpha=0.5)
    assert ss["indices"] == [5, 42, 100]


def test_hybrid_convex_scale_empty_sparse():
    sparse = {"indices": [], "values": []}
    sd, ss = _hybrid_convex_scale([1.0], sparse, alpha=0.7)
    assert ss["values"] == []
    assert ss["indices"] == []


# ── _match_to_chunk ───────────────────────────────────────────────────────────

def test_match_to_chunk_full_metadata():
    match = _make_match()
    chunk = _match_to_chunk(match)

    assert isinstance(chunk, Chunk)
    assert chunk.text == "TB is caused by Mycobacterium tuberculosis."
    assert chunk.score == pytest.approx(0.85)
    assert chunk.disease == "tb"
    assert chunk.doc_id == "who-tb-2022"
    assert chunk.chunk_type == "treatment"
    assert chunk.chunk_index == 12
    assert chunk.language == "en"


def test_match_to_chunk_source_fields():
    match = _make_match()
    chunk = _match_to_chunk(match)

    assert chunk.source.title == "WHO TB Guidelines 2022"
    assert chunk.source.url == "https://who.int/tb"
    assert chunk.source.domain == "who.int"
    assert chunk.source.source_type == "guideline"
    assert chunk.source.index == 1
    assert chunk.source.section == "Chapter 4"
    assert chunk.source.page_number == 45


def test_match_to_chunk_optional_fields_default_to_none():
    """section and page_number should be None when absent from metadata."""
    match = _make_match(metadata={
        "text": "HIV text",
        "disease": "hiv",
        "doc_id": "unaids-report",
        "source_title": "UNAIDS Report",
        "source_url": "https://unaids.org",
        "source_domain": "unaids.org",
        "source_type": "report",
        "source_index": 2,
        # No source_section, no source_page_number
        "chunk_type": "general",
    })
    chunk = _match_to_chunk(match)
    assert chunk.source.section is None
    assert chunk.source.page_number is None


def test_match_to_chunk_uses_match_id_as_doc_id_fallback():
    """When doc_id is absent from metadata, fall back to the Pinecone vector ID."""
    match = _make_match(id="fallback-chunk-id", metadata={
        "text": "Some text",
        "disease": "tb",
        # No doc_id key
        "source_title": "Source",
        "source_url": "https://who.int",
        "source_domain": "who.int",
        "source_type": "guideline",
        "source_index": 1,
        "chunk_type": "general",
    })
    chunk = _match_to_chunk(match)
    assert chunk.doc_id == "fallback-chunk-id"


def test_match_to_chunk_score_zero_on_none():
    match = _make_match(score=None)   # type: ignore[arg-type]
    chunk = _match_to_chunk(match)
    assert chunk.score == pytest.approx(0.0)


# ── hybrid_search ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_search_returns_chunks():
    adapter, mock_index = _make_adapter()
    result_mock = MagicMock()
    result_mock.matches = [_make_match()]
    mock_index.query.return_value = result_mock

    chunks = await adapter.hybrid_search(
        query_text="TB symptoms",
        query_embedding=[0.1] * 1024,
        sparse_vector={"indices": [0, 1], "values": [0.5, 0.3]},
        namespace="tb",
        top_k=5,
        alpha=0.7,
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].disease == "tb"


@pytest.mark.asyncio
async def test_hybrid_search_applies_chunk_type_filter():
    adapter, mock_index = _make_adapter()
    result_mock = MagicMock()
    result_mock.matches = []
    mock_index.query.return_value = result_mock

    await adapter.hybrid_search(
        query_text="drug dosage",
        query_embedding=[0.1] * 1024,
        sparse_vector={"indices": [0], "values": [0.5]},
        namespace="tb",
        top_k=20,
        alpha=0.3,
        chunk_type_filter="drug_interaction",
    )

    call_kwargs = mock_index.query.call_args.kwargs
    assert "filter" in call_kwargs
    assert call_kwargs["filter"] == {"chunk_type": {"$eq": "drug_interaction"}}


@pytest.mark.asyncio
async def test_hybrid_search_no_filter_when_none():
    adapter, mock_index = _make_adapter()
    result_mock = MagicMock()
    result_mock.matches = []
    mock_index.query.return_value = result_mock

    await adapter.hybrid_search(
        query_text="general question",
        query_embedding=[0.1] * 1024,
        sparse_vector={"indices": [], "values": []},
        namespace="tb",
        top_k=20,
        alpha=0.7,
        chunk_type_filter=None,
    )

    call_kwargs = mock_index.query.call_args.kwargs
    assert "filter" not in call_kwargs


@pytest.mark.asyncio
async def test_hybrid_search_scales_vectors():
    """Dense vector values must be scaled by alpha before being sent to Pinecone."""
    adapter, mock_index = _make_adapter()
    result_mock = MagicMock()
    result_mock.matches = []
    mock_index.query.return_value = result_mock

    dense = [1.0] * 1024
    await adapter.hybrid_search(
        query_text="q",
        query_embedding=dense,
        sparse_vector={"indices": [0], "values": [1.0]},
        namespace="tb",
        top_k=5,
        alpha=0.7,
    )

    sent_dense = mock_index.query.call_args.kwargs["vector"]
    assert sent_dense[0] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_hybrid_search_skips_matches_without_metadata():
    """Pinecone matches with empty metadata must be filtered out."""
    adapter, mock_index = _make_adapter()
    good_match = _make_match()
    bad_match = MagicMock()
    bad_match.metadata = {}   # empty → filtered
    bad_match.score = 0.5

    result_mock = MagicMock()
    result_mock.matches = [good_match, bad_match]
    mock_index.query.return_value = result_mock

    chunks = await adapter.hybrid_search(
        query_text="TB",
        query_embedding=[0.1] * 1024,
        sparse_vector={"indices": [], "values": []},
        namespace="tb",
        top_k=20,
        alpha=0.7,
    )

    assert len(chunks) == 1   # bad_match filtered because metadata is falsy


@pytest.mark.asyncio
async def test_hybrid_search_raises_retrieval_error_on_pinecone_failure():
    adapter, mock_index = _make_adapter()
    mock_index.query.side_effect = RuntimeError("Pinecone timeout")

    with pytest.raises(RetrievalError, match="Pinecone query failed"):
        await adapter.hybrid_search(
            query_text="TB",
            query_embedding=[0.1] * 1024,
            sparse_vector={"indices": [], "values": []},
            namespace="tb",
        )


# ── upsert_chunks ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_chunks_returns_count(mock_pinecone=None):
    adapter, mock_index = _make_adapter()
    mock_index.upsert.return_value = None
    chunks = [
        {"id": f"c{i}", "embedding": [0.1] * 1024,
         "sparse_embedding": {"indices": [0], "values": [0.5]},
         "metadata": {"text": "text", "disease": "tb"}}
        for i in range(10)
    ]
    total = await adapter.upsert_chunks(chunks, namespace="tb")
    assert total == 10


@pytest.mark.asyncio
async def test_upsert_chunks_batches_in_100s():
    """150 chunks → 2 upsert() calls (100 + 50)."""
    adapter, mock_index = _make_adapter()
    mock_index.upsert.return_value = None
    chunks = [
        {"id": f"c{i}", "embedding": [0.0] * 1024,
         "sparse_embedding": {"indices": [], "values": []},
         "metadata": {}}
        for i in range(150)
    ]
    await adapter.upsert_chunks(chunks, namespace="tb")
    assert mock_index.upsert.call_count == 2


@pytest.mark.asyncio
async def test_upsert_chunks_raises_on_failure():
    adapter, mock_index = _make_adapter()
    mock_index.upsert.side_effect = RuntimeError("write failed")

    with pytest.raises(RetrievalError, match="Pinecone upsert failed"):
        await adapter.upsert_chunks(
            [{"id": "c1", "embedding": [0.1], "sparse_embedding": {"indices": [], "values": []}, "metadata": {}}],
            namespace="tb",
        )
