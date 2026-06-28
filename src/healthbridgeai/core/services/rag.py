"""RAGService — hybrid retrieval, FlashRank re-ranking, HyDE fallback, Tavily web fallback."""
from __future__ import annotations

import structlog

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import RetrievalError
from healthbridgeai.core.models.disease import DiseaseRegistry, QueryIntent, RouteResult
from healthbridgeai.core.models.retrieval import Chunk, RetrievalResult, WebResult
from healthbridgeai.core.ports.llm import ILLMClient
from healthbridgeai.core.ports.search import IWebSearch
from healthbridgeai.core.ports.vector_store import IVectorStore

log = structlog.get_logger(__name__)

_TOP_K_RETRIEVE = 20
_TOP_K_RERANK = 5

# Flip to sparse-heavy for queries where exact drug/test names matter more than semantics
_SPARSE_HEAVY_INTENTS = {QueryIntent.DRUG_INTERACTION, QueryIntent.TREATMENT}

_HYDE_SYSTEM = (
    "You are a medical knowledge base writer. Given a health question, write a concise factual "
    "paragraph (2-4 sentences) that would appear in an authoritative medical guideline answering "
    "that question. Write only the paragraph — no preamble, no heading."
)


class RAGService:
    """
    Retrieval-Augmented Generation pipeline.

    retrieve() is the single public entry point:
        1. Dense + sparse embedding of the query
        2. Hybrid search across relevant Pinecone namespaces
        3. FlashRank cross-encoder re-rank (top-20 → top-5)
        4. HyDE fallback when best_score < HYDE_FALLBACK_THRESHOLD
        5. Tavily web fallback when score still below MIN_RETRIEVAL_SCORE_DEFAULT
    """

    def __init__(
        self,
        llm: ILLMClient,
        vector_store: IVectorStore,
        web_search: IWebSearch,
        registry: DiseaseRegistry,
    ) -> None:
        self._llm = llm
        self._vs = vector_store
        self._web = web_search
        self._registry = registry
        # Lazy load: 80 MB model; loaded once at first retrieve() call
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker
            self._ranker = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir=".flashrank_cache",
            )
        return self._ranker

    async def retrieve(
        self,
        english_query: str,
        route: RouteResult,
        phone_hash: str,
    ) -> RetrievalResult:
        alpha = 0.3 if route.query_intent in _SPARSE_HEAVY_INTENTS else 0.7

        # Embed once; reuse across all namespace searches and HyDE
        try:
            dense_batch = await self._llm.embed([english_query])
            sparse_batch = await self._llm.embed_sparse([english_query])
            query_embedding: list[float] = dense_batch[0]
            sparse_vector: dict = sparse_batch[0]
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        namespaces = self._resolve_namespaces(route)
        raw_chunks = await self._search_namespaces(
            english_query, query_embedding, sparse_vector, namespaces, alpha, route.query_intent
        )

        chunks = self._rerank(english_query, raw_chunks, _TOP_K_RERANK)
        best_score = chunks[0].score if chunks else 0.0
        used_hyde = False

        # HyDE fallback
        if best_score < settings.HYDE_FALLBACK_THRESHOLD:
            log.info("rag.hyde_fallback", best_score=round(best_score, 3), phone_hash=phone_hash)
            chunks, best_score = await self._hyde_retrieve(
                english_query, namespaces, alpha, route.query_intent
            )
            used_hyde = True

        # Tavily web fallback
        used_web = False
        web_results: list[WebResult] = []
        if best_score < settings.MIN_RETRIEVAL_SCORE_DEFAULT:
            log.info("rag.web_fallback", best_score=round(best_score, 3), phone_hash=phone_hash)
            web_results = await self._web_fallback(english_query, route)
            used_web = bool(web_results)

        log.info(
            "rag.retrieved",
            chunks=len(chunks),
            best_score=round(best_score, 3),
            used_hyde=used_hyde,
            used_web=used_web,
            phone_hash=phone_hash,
        )
        return RetrievalResult(
            chunks=chunks,
            best_score=best_score,
            used_hyde=used_hyde,
            used_web_fallback=used_web,
            web_results=web_results,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _resolve_namespaces(self, route: RouteResult) -> list[str]:
        ids = route.disease_ids if route.disease_ids else self._registry.enabled_ids()
        namespaces = []
        for did in ids:
            cfg = self._registry.get(did)
            if cfg and cfg.enabled:
                namespaces.append(cfg.pinecone_namespace)
        return namespaces or [
            cfg.pinecone_namespace
            for did in self._registry.enabled_ids()
            if (cfg := self._registry.get(did)) and cfg.enabled
        ]

    async def _search_namespaces(
        self,
        query_text: str,
        query_embedding: list[float],
        sparse_vector: dict,
        namespaces: list[str],
        alpha: float,
        intent: QueryIntent,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for ns in namespaces:
            try:
                hits = await self._vs.hybrid_search(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    sparse_vector=sparse_vector,
                    namespace=ns,
                    top_k=_TOP_K_RETRIEVE,
                    alpha=alpha,
                    chunk_type_filter=intent.value,
                )
                chunks.extend(hits)
            except Exception as exc:
                log.warning("rag.namespace_search_failed", namespace=ns, error=str(exc))
        return chunks

    def _rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        from flashrank import RerankRequest
        ranker = self._get_ranker()
        passages = [{"id": i, "text": c.text} for i, c in enumerate(chunks)]
        req = RerankRequest(query=query, passages=passages)
        ranked = ranker.rerank(req)
        idx_map = {i: c for i, c in enumerate(chunks)}
        result = []
        for r in ranked[:top_k]:
            chunk = idx_map.get(r["id"])
            if chunk:
                chunk.score = float(r.get("score", chunk.score))
                result.append(chunk)
        return result

    async def _hyde_retrieve(
        self,
        query: str,
        namespaces: list[str],
        alpha: float,
        intent: QueryIntent,
    ) -> tuple[list[Chunk], float]:
        try:
            hypothetical = await self._llm.complete(
                system=_HYDE_SYSTEM,
                user=query,
                temperature=0.3,
                max_tokens=256,
            )
            hyde_dense = (await self._llm.embed([hypothetical]))[0]
            hyde_sparse = (await self._llm.embed_sparse([hypothetical]))[0]
        except Exception as exc:
            log.warning("rag.hyde_embed_failed", error=str(exc))
            return [], 0.0

        raw = await self._search_namespaces(hypothetical, hyde_dense, hyde_sparse, namespaces, alpha, intent)
        chunks = self._rerank(query, raw, _TOP_K_RERANK)
        best = chunks[0].score if chunks else 0.0
        return chunks, best

    async def _web_fallback(self, query: str, route: RouteResult) -> list[WebResult]:
        ids = route.disease_ids or self._registry.enabled_ids()
        domains: list[str] = []
        for did in ids:
            cfg = self._registry.get(did)
            if cfg:
                domains.extend(cfg.search_domains)
        if not domains:
            return []
        try:
            return await self._web.search(
                query=query,
                allowed_domains=list(set(domains)),
                max_results=5,
            )
        except Exception as exc:
            log.warning("rag.web_search_failed", error=str(exc))
            return []
