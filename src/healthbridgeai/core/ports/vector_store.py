"""IVectorStore — contract for Pinecone hybrid-search operations."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models.disease import QueryIntent
from ..models.retrieval import Chunk


@runtime_checkable
class IVectorStore(Protocol):
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        sparse_vector: dict[str, list],
        namespace: str,
        top_k: int = 20,
        alpha: float = 0.7,
        chunk_type_filter: str | None = None,
    ) -> list[Chunk]:
        """
        Pinecone hybrid search combining dense (BGE-M3) + BM25 sparse vectors.
        alpha=0.7 → 70% dense, 30% sparse. For drug-name queries flip to 0.3/0.7.
        chunk_type_filter maps to the QueryIntent-based Pinecone metadata filter.
        """
        ...

    async def upsert_chunks(
        self,
        chunks: list[dict],
        namespace: str,
    ) -> int:
        """Upsert chunk vectors during KB ingestion. Returns count upserted."""
        ...

    async def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace (used when re-indexing a disease KB)."""
        ...

    async def describe_index(self) -> dict:
        """Return index stats (dimension, vector_count per namespace, etc.)."""
        ...
