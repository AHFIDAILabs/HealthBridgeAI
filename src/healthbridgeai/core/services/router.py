"""RouterService — two-dimensional routing: disease classification + intent in one LLM call."""
from __future__ import annotations

import structlog
from pydantic import BaseModel

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import EmergencyDetectedError
from healthbridgeai.core.models.disease import DiseaseRegistry, QueryIntent, RouteResult
from healthbridgeai.core.ports.llm import ILLMClient

log = structlog.get_logger(__name__)

_INTENT_LIST = ", ".join(qi.value for qi in QueryIntent)

_ROUTER_SYSTEM = """\
You are a medical query router for a West African health chatbot covering TB, HIV/AIDS, and Malaria.

Given an English health question, return a JSON object with exactly these fields:
- disease_ids: list of relevant disease IDs from ["tb", "hiv", "malaria"] (empty list if none specific)
- disease_confidence: float 0.0-1.0
- query_intent: one of [{intents}]
- intent_confidence: float 0.0-1.0
- is_general_health: true if the question is not specific to any of the three diseases
- is_emergency: true ONLY for immediate life-threatening emergencies (coughing blood, loss of consciousness, anaphylaxis, suicidal crisis)
- is_personal: true if the user is describing their own or a family member's current symptoms ("I have...", "my child has...")

Be conservative with is_emergency — err on the side of false."""


class _RouterLLMOutput(BaseModel):
    disease_ids: list[str] = []
    disease_confidence: float = 0.0
    query_intent: str = "general"
    intent_confidence: float = 0.0
    is_general_health: bool = False
    is_emergency: bool = False
    is_personal: bool = False


class RouterService:
    """Routes a user query to disease namespace(s) and classifies the query intent."""

    def __init__(self, llm: ILLMClient, registry: DiseaseRegistry) -> None:
        self._llm = llm
        self._registry = registry

    async def route(self, english_text: str, phone_hash: str) -> RouteResult:
        """
        1. Fast alias + emergency-keyword scan — raises EmergencyDetectedError immediately.
        2. LLM structured call for full RouteResult.
        3. Filter disease_ids to enabled only.
        4. Raise EmergencyDetectedError if LLM flagged is_emergency.
        """
        # Fast pre-scan: alias match + emergency keyword check
        alias_hits = self._registry.resolve_alias(english_text)
        for disease_id in alias_hits:
            cfg = self._registry.get(disease_id)
            if cfg:
                matched_kw = [kw for kw in cfg.emergency_keywords if kw in english_text.lower()]
                if matched_kw:
                    log.critical(
                        "router.emergency_detected",
                        disease_id=disease_id,
                        keywords=matched_kw,
                        phone_hash=phone_hash,
                    )
                    raise EmergencyDetectedError([disease_id], matched_kw)

        # LLM routing call
        try:
            raw = await self._llm.structured(
                system=_ROUTER_SYSTEM.format(intents=_INTENT_LIST),
                user=english_text[: settings.MAX_USER_INPUT_CHARS],
                response_model=_RouterLLMOutput,
                model=settings.LLM_ROUTER_MODEL,
                temperature=0.0,
            )
        except Exception as exc:
            log.error("router.llm_failed", error=str(exc), phone_hash=phone_hash)
            # Graceful fallback: search all enabled namespaces as a general query
            return RouteResult(
                disease_ids=self._registry.enabled_ids(),
                disease_confidence=0.5,
                is_general_health=True,
            )

        # LLM-flagged emergency
        if raw.is_emergency:
            log.critical("router.llm_emergency", disease_ids=raw.disease_ids, phone_hash=phone_hash)
            raise EmergencyDetectedError(raw.disease_ids, ["llm_detected"])

        # Filter to enabled diseases only
        enabled = set(self._registry.enabled_ids())
        disease_ids = [d for d in raw.disease_ids if d in enabled]
        if not disease_ids and not raw.is_general_health:
            # LLM named diseases that are disabled — search all enabled instead
            disease_ids = list(enabled)

        try:
            intent = QueryIntent(raw.query_intent)
        except ValueError:
            intent = QueryIntent.GENERAL

        result = RouteResult(
            disease_ids=disease_ids,
            disease_confidence=raw.disease_confidence,
            query_intent=intent,
            intent_confidence=raw.intent_confidence,
            is_general_health=raw.is_general_health,
            is_emergency=False,
            is_personal=raw.is_personal,
        )
        log.info(
            "router.routed",
            disease_ids=disease_ids,
            intent=intent.value,
            is_personal=raw.is_personal,
            phone_hash=phone_hash,
        )
        return result
