"""Unit tests for ResponseGenerator."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from healthbridgeai.core.models.response import BotResponse, CachedResponse, LLMResponse
from healthbridgeai.core.services.generator import ResponseGenerator


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
    assert bot.text


# ── Source citation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sources_included_in_bot_response(
    mock_llm, retrieval_result, route_result, llm_response
):
    """sources_used=[1] should append a [1] citation footer to the message."""
    mock_llm.structured.return_value = llm_response  # sources_used=[1]
    gen = ResponseGenerator(mock_llm)

    _, bot = await gen.generate(
        "TB symptoms?", retrieval_result, route_result, phone_hash="abc123"
    )

    # Source index 1 → "[1] WHO TB Guidelines 2023 — who.int"
    assert "[1]" in bot.text


# ── Needs professional ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_needs_professional_included_in_message(
    mock_llm, retrieval_result, route_result
):
    resp = LLMResponse(
        answer="You may have TB.",
        confidence="medium",
        needs_professional=True,
        caveat=None,
        sources_used=[1],
    )
    mock_llm.structured.return_value = resp
    gen = ResponseGenerator(mock_llm)

    _, bot = await gen.generate("I have a cough", retrieval_result, route_result, phone_hash="abc123")

    # _format_message appends the "consult a qualified healthcare provider" line
    assert "healthcare" in bot.text.lower() or "provider" in bot.text.lower()
    assert bot.needs_professional is True


# ── Caveat ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_caveat_appears_in_message(mock_llm, retrieval_result, route_result):
    resp = LLMResponse(
        answer="Rifampicin treats TB.",
        confidence="high",
        needs_professional=True,
        caveat="Rifampicin may interact with antiretrovirals.",
        sources_used=[1],
    )
    mock_llm.structured.return_value = resp
    gen = ResponseGenerator(mock_llm)

    _, bot = await gen.generate("rifampicin", retrieval_result, route_result, phone_hash="abc123")

    assert "Rifampicin may interact" in bot.text


# ── Low confidence ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_low_confidence_propagated(mock_llm, retrieval_result, route_result):
    resp = LLMResponse(
        answer="I am not sure about this.",
        confidence="low",
        needs_professional=False,
        caveat=None,
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
    now = int(time.time())
    cached = CachedResponse(
        english_query="TB symptoms",
        disease_ids="tb",           # comma-joined string
        query_intent="symptoms",
        english_response="TB symptoms include cough and fever.",
        sources_json="[]",
        confidence="high",
        created_at=now,
        expires_at=now + 604_800,   # 7 days
    )
    gen = ResponseGenerator(mock_llm)
    bot = gen.from_cache(cached, language_code="en")

    assert isinstance(bot, BotResponse)
    assert bot.cache_hit is True
    assert "cough" in bot.text
