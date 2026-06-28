"""SemanticCache — IResponseCache implementation using Pinecone response-cache namespace."""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Optional

import structlog
from pinecone import Pinecone

from healthbridgeai.config.settings import settings
from healthbridgeai.core.models.response import CachedResponse

log = structlog.get_logger(__name__)

_CACHE_NAMESPACE = "response-cache"


class SemanticCache:
    """
    Stores response embeddings in a dedicated Pinecone namespace.
    Metadata holds all CachedResponse fields (flat key-value, ≤ 40 KB limit).

    Never-cache rules are enforced by the pipeline before calling store().
    This adapter enforces them again as a safety net.
    """

    def __init__(self) -> None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = pc.Index(settings.PINECONE_INDEX_NAME)

    async def lookup(
        self,
        english_query: str,
        query_embedding: list[float],
        disease_ids: list[str],
        threshold: float = 0.92,
    ) -> Optional[CachedResponse]:
        """Return a CachedResponse if score >= threshold, disease_ids match, and not expired."""
        now = int(time.time())
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._index.query(
                    namespace=_CACHE_NAMESPACE,
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    filter={"disease_ids": {"$eq": ",".join(sorted(disease_ids))}},
                ),
            )
        except Exception as exc:
            log.warning("cache.lookup.failed", error=str(exc))
            return None

        for match in results.matches:
            if match.score < threshold:
                break  # sorted descending; no point checking further
            meta = match.metadata or {}
            if int(meta.get("expires_at", 0)) < now:
                continue  # expired entry
            try:
                cached = _meta_to_cached(meta)
            except Exception:
                continue
            log.info("cache.hit", score=round(match.score, 3), disease_ids=disease_ids)
            # Increment hit_count asynchronously (best-effort, fire-and-forget)
            asyncio.ensure_future(self._bump_hit_count(match.id, meta))
            return cached

        return None

    async def store(
        self,
        english_query: str,
        query_embedding: list[float],
        response: CachedResponse,
    ) -> None:
        """Upsert the response embedding + metadata into the cache namespace."""
        # Safety-net: enforce never-cache rules even if caller skipped them
        if response.confidence == "low":
            return

        vector_id = _make_id(english_query, response.disease_ids)
        meta = _cached_to_meta(response)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._index.upsert(
                    vectors=[{"id": vector_id, "values": query_embedding, "metadata": meta}],
                    namespace=_CACHE_NAMESPACE,
                ),
            )
            log.debug("cache.stored", vector_id=vector_id[:12])
        except Exception as exc:
            log.warning("cache.store.failed", error=str(exc))

    async def invalidate(self, disease_id: str) -> int:
        """Delete all cache entries that contain this disease_id. Returns count deleted."""
        try:
            loop = asyncio.get_event_loop()
            # Fetch matching IDs first (metadata filter on delete requires paid plan)
            results = await loop.run_in_executor(
                None,
                lambda: self._index.query(
                    namespace=_CACHE_NAMESPACE,
                    vector=[0.0] * settings.PINECONE_INDEX_DIMENSION,
                    top_k=10_000,
                    include_metadata=False,
                    filter={"disease_ids": {"$contains": disease_id}},
                ),
            )
            ids = [m.id for m in results.matches]
            if ids:
                await loop.run_in_executor(
                    None,
                    lambda: self._index.delete(ids=ids, namespace=_CACHE_NAMESPACE),
                )
            log.info("cache.invalidated", disease_id=disease_id, count=len(ids))
            return len(ids)
        except Exception as exc:
            log.warning("cache.invalidate.failed", error=str(exc))
            return 0

    async def _bump_hit_count(self, vector_id: str, meta: dict) -> None:
        try:
            new_meta = dict(meta)
            new_meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._index.update(
                    id=vector_id,
                    set_metadata={"hit_count": new_meta["hit_count"]},
                    namespace=_CACHE_NAMESPACE,
                ),
            )
        except Exception:
            pass  # best-effort


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(english_query: str, disease_ids_str: str) -> str:
    key = f"{disease_ids_str}:{english_query[:200]}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _cached_to_meta(r: CachedResponse) -> dict:
    return {
        "english_query": r.english_query[:500],
        "disease_ids": r.disease_ids,
        "query_intent": r.query_intent,
        "english_response": r.english_response[:8000],  # Pinecone metadata limit ~40 KB
        "sources_json": r.sources_json[:4000],
        "confidence": r.confidence,
        "created_at": r.created_at,
        "expires_at": r.expires_at,
        "hit_count": r.hit_count,
    }


def _meta_to_cached(meta: dict) -> CachedResponse:
    return CachedResponse(
        english_query=meta["english_query"],
        disease_ids=meta["disease_ids"],
        query_intent=meta["query_intent"],
        english_response=meta["english_response"],
        sources_json=meta.get("sources_json", "[]"),
        confidence=meta["confidence"],
        created_at=int(meta["created_at"]),
        expires_at=int(meta["expires_at"]),
        hit_count=int(meta.get("hit_count", 0)),
    )
