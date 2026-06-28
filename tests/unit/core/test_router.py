"""Unit tests for RouterService."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from healthbridgeai.core.exceptions import EmergencyDetectedError
from healthbridgeai.core.models.disease import QueryIntent, RouteResult
from healthbridgeai.core.services.router import RouterService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_llm(
    disease_ids: list[str],
    query_intent: str,
    disease_confidence: float = 0.9,
    intent_confidence: float = 0.85,
    is_emergency: bool = False,
    is_personal: bool = False,
    is_general_health: bool = False,
) -> AsyncMock:
    """Return an LLM mock whose structured() yields the given router output."""
    result = MagicMock()
    result.disease_ids = disease_ids
    result.query_intent = query_intent          # raw string (RouterService converts to enum)
    result.disease_confidence = disease_confidence
    result.intent_confidence = intent_confidence
    result.is_emergency = is_emergency
    result.is_personal = is_personal
    result.is_general_health = is_general_health
    llm = AsyncMock()
    llm.structured.return_value = result
    return llm


# ── Emergency detection: alias + keyword pre-scan ─────────────────────────────

@pytest.mark.asyncio
async def test_emergency_keyword_short_circuits(disease_registry):
    """
    Alias scan finds 'tuberculosis' → checks emergency_keywords → raises.
    LLM structured() must NOT be called.
    """
    llm = AsyncMock()
    svc = RouterService(llm, disease_registry)

    with pytest.raises(EmergencyDetectedError):
        await svc.route("I have tuberculosis and I am coughing blood heavily", phone_hash="abc123")

    llm.structured.assert_not_called()


@pytest.mark.asyncio
async def test_emergency_keyword_haemoptysis(disease_registry):
    """'haemoptysis' in emergency_keywords triggers EmergencyDetectedError."""
    llm = AsyncMock()
    svc = RouterService(llm, disease_registry)

    with pytest.raises(EmergencyDetectedError):
        await svc.route("tuberculosis patient with haemoptysis", phone_hash="abc123")

    llm.structured.assert_not_called()


@pytest.mark.asyncio
async def test_emergency_not_triggered_without_alias(disease_registry):
    """'coughing blood' alone (no TB alias) must NOT trigger the pre-scan."""
    llm = _mock_llm(["tb"], "symptoms")
    svc = RouterService(llm, disease_registry)

    # Should not raise; LLM is called because alias scan found no disease
    result = await svc.route("I am coughing blood", phone_hash="abc123")
    assert isinstance(result, RouteResult)
    llm.structured.assert_called_once()


# ── LLM emergency path ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_flagged_emergency_raises(disease_registry):
    """When LLM sets is_emergency=True, RouterService raises EmergencyDetectedError."""
    llm = _mock_llm(["tb"], "symptoms", is_emergency=True)
    svc = RouterService(llm, disease_registry)

    with pytest.raises(EmergencyDetectedError):
        await svc.route("I cannot breathe and have a high fever", phone_hash="abc123")


# ── LLM routing ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_routing_returns_route_result(disease_registry):
    llm = _mock_llm(["tb"], "treatment", disease_confidence=0.88, intent_confidence=0.9)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("What is the DOTS treatment for TB?", phone_hash="abc123")

    assert isinstance(result, RouteResult)
    assert "tb" in result.disease_ids
    assert result.query_intent == QueryIntent.TREATMENT
    assert result.disease_confidence == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_llm_routing_filters_disabled_diseases(disease_registry):
    """LLM suggests HIV but it is not in the registry → fall back to all enabled diseases."""
    llm = _mock_llm(["hiv", "tb"], "general")
    svc = RouterService(llm, disease_registry)

    result = await svc.route("How does HIV affect TB treatment?", phone_hash="abc123")

    # HIV is not registered → removed; only tb remains
    assert result.disease_ids == ["tb"]


# ── Graceful fallback ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_all_enabled(disease_registry):
    """If structured() raises, RouteResult should cover all enabled diseases."""
    llm = AsyncMock()
    llm.structured.side_effect = RuntimeError("LLM timeout")
    svc = RouterService(llm, disease_registry)

    result = await svc.route("I feel sick", phone_hash="abc123")

    assert result.disease_ids == ["tb"]        # only enabled disease
    assert result.is_general_health is True    # fallback sets this


# ── Intent and flag propagation ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_personal_health_flag_propagated(disease_registry):
    llm = _mock_llm(["tb"], "symptoms", is_personal=True)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("I think I have TB, what should I do?", phone_hash="abc123")

    assert result.is_personal is True


@pytest.mark.asyncio
async def test_drug_interaction_intent_propagated(disease_registry):
    llm = _mock_llm(["tb"], "drug_interaction", intent_confidence=0.85)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("Can I take rifampicin with paracetamol?", phone_hash="abc123")

    assert result.query_intent == QueryIntent.DRUG_INTERACTION


@pytest.mark.asyncio
async def test_unknown_intent_falls_back_to_general(disease_registry):
    """If LLM returns an unrecognised intent string, it maps to GENERAL."""
    llm = _mock_llm(["tb"], "completely_made_up_intent")
    svc = RouterService(llm, disease_registry)

    result = await svc.route("Tell me about health", phone_hash="abc123")

    assert result.query_intent == QueryIntent.GENERAL
