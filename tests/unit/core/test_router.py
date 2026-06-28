"""Unit tests for RouterService."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from healthbridgeai.core.exceptions import EmergencyDetectedError
from healthbridgeai.core.models.retrieval import QueryIntent, RouteResult
from healthbridgeai.core.services.router import RouterService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_llm_structured(disease_ids: list[str], intent: str, confidence: float):
    """Return a mock LLM that yields a structured route response."""
    result = MagicMock()
    result.disease_ids = disease_ids
    result.intent = intent
    result.confidence = confidence
    result.is_emergency = False
    result.is_personal = False
    llm = AsyncMock()
    llm.structured.return_value = result
    return llm


# ── Emergency detection ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_emergency_keyword_short_circuits(disease_registry):
    """Keyword pre-scan raises EmergencyDetectedError before any LLM call."""
    llm = AsyncMock()
    svc = RouterService(llm, disease_registry)

    with pytest.raises(EmergencyDetectedError):
        await svc.route("I am coughing blood and cannot breathe", phone_hash="abc123")

    llm.structured.assert_not_called()


@pytest.mark.asyncio
async def test_emergency_keyword_haemoptysis(disease_registry):
    llm = AsyncMock()
    svc = RouterService(llm, disease_registry)

    with pytest.raises(EmergencyDetectedError):
        await svc.route("haemoptysis started this morning", phone_hash="abc123")


# ── Alias scan ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alias_scan_routes_to_tb(disease_registry):
    """'tuberculosis' alias maps to TB without LLM when intent is clear."""
    llm = _mock_llm_structured(["tb"], "SYMPTOM_QUERY", 0.95)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("What is tuberculosis treatment?", phone_hash="abc123")

    assert "tb" in result.disease_ids


# ── LLM routing ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_routing_returns_route_result(disease_registry):
    llm = _mock_llm_structured(["tb"], "TREATMENT_PROTOCOL", 0.88)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("What is the DOTs treatment for TB?", phone_hash="abc123")

    assert isinstance(result, RouteResult)
    assert "tb" in result.disease_ids
    assert result.intent == QueryIntent.TREATMENT_PROTOCOL
    assert result.confidence == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_llm_routing_filters_disabled_diseases(disease_registry):
    """LLM may suggest HIV but it's disabled — should fall back to enabled diseases."""
    llm = _mock_llm_structured(["hiv", "tb"], "GENERAL_HEALTH", 0.7)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("How does HIV affect TB?", phone_hash="abc123")

    # HIV not in registry.enabled → only tb in result
    assert result.disease_ids == ["tb"]


# ── Graceful fallback ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_all_enabled(disease_registry):
    """If structured() raises, fall back to all enabled diseases."""
    llm = AsyncMock()
    llm.structured.side_effect = RuntimeError("LLM timeout")
    svc = RouterService(llm, disease_registry)

    result = await svc.route("I feel sick", phone_hash="abc123")

    assert result.disease_ids == ["tb"]  # only enabled disease
    assert result.confidence < 0.5


# ── Intent mapping ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_personal_health_flag_propagated(disease_registry):
    llm_result = MagicMock()
    llm_result.disease_ids = ["tb"]
    llm_result.intent = "PERSONAL_HEALTH"
    llm_result.confidence = 0.9
    llm_result.is_emergency = False
    llm_result.is_personal = True
    llm = AsyncMock()
    llm.structured.return_value = llm_result
    svc = RouterService(llm, disease_registry)

    result = await svc.route("I think I have TB, what should I do?", phone_hash="abc123")

    assert result.is_personal is True


@pytest.mark.asyncio
async def test_drug_interaction_intent_propagated(disease_registry):
    llm = _mock_llm_structured(["tb"], "DRUG_INTERACTION", 0.85)
    svc = RouterService(llm, disease_registry)

    result = await svc.route("Can I take rifampicin with paracetamol?", phone_hash="abc123")

    assert result.intent == QueryIntent.DRUG_INTERACTION
