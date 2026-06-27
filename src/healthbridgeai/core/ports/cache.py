"""IResponseCache — contract for semantic response caching in Pinecone."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..models.response import CachedResponse


@runtime_checkable
class IResponseCache(Protocol):
    async def lookup(
        self,
        english_query: str,
        query_embedding: list[float],
        disease_ids: list[str],
        threshold: float = 0.92,
    ) -> Optional[CachedResponse]:
        """
        Search Pinecone response-cache namespace by cosine similarity.
        Returns a hit only if score >= threshold AND disease_ids match.
        Returns None on cache miss or any internal error (never raises).
        """
        ...

    async def store(
        self,
        english_query: str,
        query_embedding: list[float],
        response: CachedResponse,
    ) -> None:
        """
        Upsert a response into the cache.
        Implementations must enforce TTL via expires_at metadata.
        Must NOT store: emergency responses, low-confidence responses,
        personal health queries, or needs_professional=True responses.
        """
        ...

    async def invalidate(self, disease_id: str) -> int:
        """
        Delete all cache entries for a disease (called after KB re-indexing).
        Returns the number of entries deleted.
        """
        ...
