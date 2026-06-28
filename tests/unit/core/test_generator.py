"""Unit tests for ResponseGenerator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from healthbridgeai.core.models.response import BotResponse, LLMResponse
from healthbridgeai.core.services.generator import ResponseGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(llm_response: LLMResponse) -> AsyncMock:
    llm = AsyncMock()
    llm.structured.return_value = llm_response
    return llm


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_returns_bot_response(
    mock_llm, retrieval_result, route_result, llm_response
):
    mock_llm.structured.return_value = llm_response
    gen = ResponseGenerator(mock_llm)

    llm_resp, bot = await gen.generate(
        "What are TB symptoms?", retrieval_result, route_result, phone_hash="abc123"
    )

    assert isinstance(llm_resp, LLMResponse)
    assert isinstance(bot, BotResponse)
    assert bot.text  # non-empty


# ── Source citation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sources_included_in_bot_response(
    mock_llm, retrieval_result, route_result, llm_response
):
    """sources_used=[1] should append a citation footer to the message."""
    mock_llm.structured.return_value = llm_response  # sources_used=[1]
    gen = ResponseGenerator(mock_llm)

    _, bot = await gen.generate(
        "TB symptoms?", retrieval_result, route_result, phone_hash="abc123"
    )

    # Should contain some citation reference
    assert "[1]" in bot.text or "Sources" in bot.text or "WHO" in bot.text


# ── Needs professional ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_needs_professional_included_in_message(
    mock_llm, retrieval_result, route_result
):
    resp = LLMResponse(
        answer="You may have TB.",
        confidence="medium",
        needs_professional=True,
        caveat="",
        sources_used=[1],
    )
    mock_llm.structured.return_value = resp
    gen = ResponseGenerator(mock_llm)

    _, bot = await gen.generate("I have a cough", retrieval_result, route_result, phone_hash="abc123")

    # Message should include a prompt to see a professional
    text_lower = bot.text.lower()
    assert any(kw in text_lower for kw in ["health", "professional", "doctor", "clinic"])


# ── Low confidence ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_low_confidence_response(mock_llm, retrieval_result, route_result):
    resp = LLMResponse(
        answer="I am not sure about this.",
        confidence="low",
        needs_professional=False,
        caveat="Limited information available.",
        sources_used=[],
    )
    mock_llm.structured.return_value = resp
    gen = ResponseGenerator(mock_llm)

    llm_resp, bot = await gen.generate(
        "edge case query", retrieval_result, route_result, phone_hash="abc123"
    )

    assert llm_resp.confidence == "low"
    assert bot.confidence == "low"


# ── Cache-served response ─────────────────────────────────────────────────────

def test_from_cache_returns_translated_response(mock_llm):
    from healthbridgeai.core.models.response import CachedResponse

    cached = CachedResponse(
        query_text="TB symptoms",
        response_text="TB symptoms include cough and fever.",
        language_code="en",
        disease_ids=["tb"],
        confidence="high",
        sources=[],
    )
    gen = ResponseGenerator(mock_llm)
    bot = gen.from_cache(cached, language_code="en")

    assert isinstance(bot, BotResponse)
    assert bot.cache_hit is True
    assert "cough" in bot.text
