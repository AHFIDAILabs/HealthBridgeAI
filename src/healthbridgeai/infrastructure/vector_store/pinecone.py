"""PineconeAdapter — IVectorStore implementation using Pinecone hybrid search."""
from __future__ import annotations

import structlog
from pinecone import Pinecone

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import RetrievalError
from healthbridgeai.core.models.retrieval import Chunk, Source

log = structlog.get_logger(__name__)

_BATCH_SIZE = 100  # max vectors per upsert call


class PineconeAdapter:
    """
    Pinecone hybrid-search adapter.
    Uses pre-computed BGE-M3 dense + lexical sparse vectors.
    Index must be created with dotproduct metric and 1024 dimensions.
    """

    def __init__(self) -> None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = pc.Index(settings.PINECONE_INDEX_NAME)
        log.info("pinecone.connected", index=settings.PINECONE_INDEX_NAME)

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
        Convex combination of dense + sparse via manual vector scaling.
        alpha=0.7 → 70% dense, 30% sparse.
        """
        scaled_dense, scaled_sparse = _hybrid_convex_scale(
            query_embedding, sparse_vector, alpha
        )

        query_kwargs: dict = {
            "namespace": namespace,
            "vector": scaled_dense,
            "sparse_vector": scaled_sparse,
            "top_k": top_k,
            "include_metadata": True,
        }
        if chunk_type_filter:
            query_kwargs["filter"] = {"chunk_type": {"$eq": chunk_type_filter}}

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: self._index.query(**query_kwargs)
            )
        except Exception as exc:
            raise RetrievalError(f"Pinecone query failed in {namespace}: {exc}") from exc

        return [_match_to_chunk(m) for m in results.matches if m.metadata]

    async def upsert_chunks(self, chunks: list[dict], namespace: str) -> int:
        """Upsert pre-embedded chunk vectors. Returns count upserted."""
        total = 0
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            for i in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[i: i + _BATCH_SIZE]
                vectors = [
                    {
                        "id": c["id"],
                        "values": c["embedding"],
                        "sparse_values": c.get("sparse_embedding", {"indices": [], "values": []}),
                        "metadata": c["metadata"],
                    }
                    for c in batch
                ]
                await loop.run_in_executor(
                    None,
                    lambda v=vectors: self._index.upsert(vectors=v, namespace=namespace),
                )
                total += len(batch)
                log.debug("pinecone.upsert.batch", count=total, namespace=namespace)
        except Exception as exc:
            raise RetrievalError(f"Pinecone upsert failed: {exc}") from exc
        return total

    async def delete_namespace(self, namespace: str) -> None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._index.delete(delete_all=True, namespace=namespace),
            )
            log.info("pinecone.namespace_deleted", namespace=namespace)
        except Exception as exc:
            raise RetrievalError(f"Pinecone delete_namespace failed: {exc}") from exc

    async def describe_index(self) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self._index.describe_index_stats)
        return {
            "dimension": stats.dimension,
            "total_vector_count": stats.total_vector_count,
            "namespaces": {
                ns: ns_stats.vector_count
                for ns, ns_stats in (stats.namespaces or {}).items()
            },
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hybrid_convex_scale(
    dense: list[float],
    sparse: dict[str, list],
    alpha: float,
) -> tuple[list[float], dict]:
    """
    Scale dense by alpha and sparse by (1 - alpha).
    Equivalent to pinecone_text.hybrid.hybrid_convex_scale but dependency-free.
    """
    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse.get("indices", []),
        "values": [v * (1 - alpha) for v in sparse.get("values", [])],
    }
    return scaled_dense, scaled_sparse


def _match_to_chunk(match) -> Chunk:
    meta = match.metadata or {}
    source = Source(
        index=int(meta.get("source_index", 1)),
        title=str(meta.get("source_title", "Unknown Source")),
        url=str(meta.get("source_url", "")),
        domain=str(meta.get("source_domain", "")),
        source_type=str(meta.get("source_type", "guideline")),
        section=meta.get("source_section") or None,
        page_number=int(meta["source_page_number"]) if meta.get("source_page_number") else None,
    )
    return Chunk(
        text=str(meta.get("text", "")),
        score=float(match.score or 0.0),
        disease=str(meta.get("disease", "")),
        doc_id=str(meta.get("doc_id", match.id)),
        source=source,
        chunk_type=str(meta.get("chunk_type", "general")),
        chunk_index=int(meta.get("chunk_index", 0)),
        language=str(meta.get("language", "en")),
    )
