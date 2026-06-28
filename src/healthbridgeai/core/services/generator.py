"""ResponseGenerator — LLM prompt construction, structured output, citation assembly."""
from __future__ import annotations

import structlog

from healthbridgeai.core.models.disease import RouteResult
from healthbridgeai.core.models.response import (
    CITATION_DIVIDER,
    MEDICAL_DISCLAIMER,
    BotResponse,
    CachedResponse,
    LLMResponse,
)
from healthbridgeai.core.models.retrieval import RetrievalResult, Source
from healthbridgeai.core.ports.llm import ILLMClient

log = structlog.get_logger(__name__)

_GENERATOR_SYSTEM = """\
You are HealthBridgeAI, a trusted medical information assistant for West Africa,
specialising in TB, HIV/AIDS, and Malaria. You are responding via WhatsApp.

Rules:
- Answer using ONLY the provided context documents; do not add outside knowledge.
- Cite every factual claim with [Doc N] — N matches the context document number.
- If context is insufficient, say so honestly and advise the user to consult a healthcare worker.
- Never diagnose. Never prescribe. Use clear, simple language.
- Keep responses concise — WhatsApp has no infinite scroll.
- For drug interactions or dosage questions, always set needs_professional=true.
- For emergency symptoms, acknowledge urgency and direct to health services.
- caveat: add a short critical qualification only when truly necessary (e.g. drug interaction warning); otherwise leave null.

Context documents:
{context_block}"""


class ResponseGenerator:
    """Generates a structured LLM answer and formats the final WhatsApp message."""

    def __init__(self, llm: ILLMClient) -> None:
        self._llm = llm

    async def generate(
        self,
        english_query: str,
        retrieval: RetrievalResult,
        route: RouteResult,
        phone_hash: str,
    ) -> tuple[LLMResponse, BotResponse]:
        """Call the LLM and assemble a formatted BotResponse. Both are in English."""
        context_block = self._build_context(retrieval)

        llm_resp: LLMResponse = await self._llm.structured(
            system=_GENERATOR_SYSTEM.format(context_block=context_block),
            user=english_query,
            response_model=LLMResponse,
            temperature=0.1,
        )

        all_sources = retrieval.all_sources
        cited: list[Source] = []
        for idx in llm_resp.sources_used:
            if 1 <= idx <= len(all_sources):
                cited.append(all_sources[idx - 1])

        text = self._format_message(llm_resp, cited)
        bot = BotResponse(
            text=text,
            language_code="en",
            sources=cited,
            confidence=llm_resp.confidence,
            needs_professional=llm_resp.needs_professional,
            is_emergency=False,
            cache_hit=False,
        )
        log.info(
            "generator.generated",
            confidence=llm_resp.confidence,
            sources_cited=len(cited),
            needs_professional=llm_resp.needs_professional,
            phone_hash=phone_hash,
        )
        return llm_resp, bot

    def from_cache(self, cached: CachedResponse, language_code: str) -> BotResponse:
        """Re-hydrate a BotResponse from a semantic cache hit (English text; caller translates)."""
        sources = cached.sources()
        text = self._format_cached_message(cached, sources)
        return BotResponse(
            text=text,
            language_code=language_code,
            sources=sources,
            confidence=cached.confidence,
            needs_professional=False,
            is_emergency=False,
            cache_hit=True,
        )

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _build_context(self, retrieval: RetrievalResult) -> str:
        all_sources = retrieval.all_sources
        texts: list[str] = (
            [c.text for c in retrieval.chunks]
            + [w.content for w in retrieval.web_results]
        )
        if not texts:
            return "(No context documents retrieved)"
        parts = []
        for src, text in zip(all_sources, texts):
            parts.append(f"[Doc {src.index}] {src.short_citation}\n{text}")
        return "\n\n".join(parts)

    def _format_message(self, llm: LLMResponse, sources: list[Source]) -> str:
        lines = [llm.answer]
        if llm.caveat:
            lines.append(f"\n⚠️ *{llm.caveat}*")
        if llm.needs_professional:
            lines.append("\n🏥 _Please consult a qualified healthcare provider for personalised advice._")
        if sources:
            lines.append(f"\n{CITATION_DIVIDER}")
            lines.append("*Sources:*")
            for s in sources:
                lines.append(f"[{s.index}] {s.short_citation}")
        lines.append(f"\n{MEDICAL_DISCLAIMER}")
        return "\n".join(lines)

    def _format_cached_message(self, cached: CachedResponse, sources: list[Source]) -> str:
        lines = [cached.english_response]
        if sources:
            lines.append(f"\n{CITATION_DIVIDER}")
            lines.append("*Sources:*")
            for s in sources:
                lines.append(f"[{s.index}] {s.short_citation}")
        lines.append(f"\n{MEDICAL_DISCLAIMER}")
        return "\n".join(lines)
