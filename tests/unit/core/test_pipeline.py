"""Unit tests for MessagePipeline — the end-to-end orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from healthbridgeai.core.exceptions import EmergencyDetectedError
from healthbridgeai.core.models.disease import QueryIntent, RouteResult
from healthbridgeai.core.models.message import InboundMessage, MessageType, ParsedMessage
from healthbridgeai.core.models.response import BotResponse, CachedResponse, LLMResponse
from healthbridgeai.core.models.user import RateLimit, User
from healthbridgeai.core.services.pipeline import (
    MessagePipeline,
    _EMERGENCY_TEXT,
    _HELP_TEXT,
    _ABOUT_TEXT,
)
from tests.conftest import _make_chunk


# ── Test context ──────────────────────────────────────────────────────────────

@dataclass
class Ctx:
    """Holds the pipeline and all its mocked dependencies for easy assertion."""
    pipe: MessagePipeline
    lang: AsyncMock       # LanguageService (duck-typed)
    router: AsyncMock     # RouterService
    rag: AsyncMock        # RAGService
    gen: MagicMock        # ResponseGenerator (sync from_cache + async generate)
    llm: AsyncMock        # ILLMClient
    users: AsyncMock      # IUserStore
    convs: AsyncMock      # IConversationStore
    cache: AsyncMock      # IResponseCache


@pytest.fixture
def now() -> int:
    return int(time.time())


@pytest.fixture
def existing_user(now: int) -> User:
    return User(
        phone_number="+2348012345678",
        language_code="en",
        audio_mode=False,
        created_at=now - 3600,
        last_seen_at=now - 60,
        message_count=5,
    )


def _make_parsed(
    message: InboundMessage,
    *,
    english_text: str = "What are the symptoms of TB?",
    lang: str = "en",
    is_command: bool = False,
    command: Optional[str] = None,
    command_args: Optional[list[str]] = None,
) -> ParsedMessage:
    return ParsedMessage(
        original=message,
        language_code=lang,
        english_text=english_text,
        is_command=is_command,
        command=command,
        command_args=command_args or [],
    )


def _make_ctx(
    inbound_message: InboundMessage,
    route_result: RouteResult,
    retrieval_result,
    llm_response: LLMResponse,
    existing_user: User,
    now: int,
) -> Ctx:
    """Wire up a full pipeline with all dependencies mocked."""
    # Language service mock
    lang = AsyncMock()
    lang.parse = AsyncMock(
        return_value=_make_parsed(inbound_message, english_text="What are the symptoms of TB?")
    )
    lang.translate_from_english = AsyncMock(side_effect=lambda text, lang_code: text)

    # Router mock
    router = AsyncMock()
    router.route = AsyncMock(return_value=route_result)

    # RAG mock
    rag = AsyncMock()
    rag.retrieve = AsyncMock(return_value=retrieval_result)

    # Generator mock — generate() is async, from_cache() is sync
    gen = MagicMock()
    bot_response = BotResponse(
        text="TB symptoms include persistent cough, fever, and night sweats. [1]",
        language_code="en",
        sources=[_make_chunk().source],
        confidence="high",
        needs_professional=False,
        is_emergency=False,
        cache_hit=False,
    )
    gen.generate = AsyncMock(return_value=(llm_response, bot_response))
    gen.from_cache = MagicMock(
        return_value=BotResponse(
            text="Cached TB answer.",
            language_code="en",
            sources=[],
            confidence="high",
            needs_professional=False,
            is_emergency=False,
            cache_hit=True,
        )
    )

    # LLM mock (just embedding)
    llm = AsyncMock()
    llm.embed = AsyncMock(return_value=[[0.1] * 1024])

    # User store mock
    users = AsyncMock()
    users.get_user = AsyncMock(return_value=existing_user)
    users.upsert_user = AsyncMock(return_value=None)
    users.check_rate_limit = AsyncMock(
        return_value=RateLimit(
            phone_hash="abc123testxx",
            window_start=now,
            message_count=1,
            is_blocked=False,
        )
    )

    # Conversation store mock
    convs = AsyncMock()
    convs.save_turn = AsyncMock(return_value=None)

    # Cache mock — miss by default
    cache = AsyncMock()
    cache.lookup = AsyncMock(return_value=None)
    cache.store = AsyncMock(return_value=None)

    pipe = MessagePipeline(
        language_svc=lang,
        router_svc=router,
        rag_svc=rag,
        generator=gen,
        llm=llm,
        user_store=users,
        conv_store=convs,
        cache=cache,
    )
    return Ctx(pipe=pipe, lang=lang, router=router, rag=rag, gen=gen,
               llm=llm, users=users, convs=convs, cache=cache)


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_returns_bot_response(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    result = await ctx.pipe.process(inbound_message)

    assert isinstance(result, BotResponse)
    assert result.text
    ctx.router.route.assert_called_once()
    ctx.rag.retrieve.assert_called_once()
    ctx.gen.generate.assert_called_once()


@pytest.mark.asyncio
async def test_happy_path_saves_two_conversation_turns(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    await ctx.pipe.process(inbound_message)

    # save_turn called twice: user turn + assistant turn
    assert ctx.convs.save_turn.call_count == 2
    first_turn = ctx.convs.save_turn.call_args_list[0].args[1]
    second_turn = ctx.convs.save_turn.call_args_list[1].args[1]
    assert first_turn.role == "user"
    assert second_turn.role == "assistant"


@pytest.mark.asyncio
async def test_response_stored_in_cache_on_happy_path(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    await ctx.pipe.process(inbound_message)
    # High-confidence, non-personal query → cache.store should be called
    ctx.cache.store.assert_called_once()


# ── User lifecycle ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_new_user_is_created_when_not_found(
    inbound_message, route_result, retrieval_result, llm_response, now
):
    existing_user = User(
        phone_number="+2348012345678", created_at=now, last_seen_at=now
    )
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.users.get_user = AsyncMock(return_value=None)  # user does not exist yet

    await ctx.pipe.process(inbound_message)

    # upsert_user called with a newly constructed User
    ctx.users.upsert_user.assert_called_once()
    created_user: User = ctx.users.upsert_user.call_args.args[0]
    assert created_user.phone_number == inbound_message.from_number
    assert created_user.message_count == 0


@pytest.mark.asyncio
async def test_returning_user_gets_message_count_incremented(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    await ctx.pipe.process(inbound_message)

    upserted: User = ctx.users.upsert_user.call_args.args[0]
    assert upserted.message_count == existing_user.message_count + 1


# ── Rate limiting ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limited_returns_error_response(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    # Set message_count == limit so is_exceeded() returns True
    ctx.users.check_rate_limit = AsyncMock(
        return_value=RateLimit(
            phone_hash="abc123testxx",
            window_start=now,
            message_count=20,   # equals RATE_LIMIT_MESSAGES_PER_MINUTE (20)
            is_blocked=False,
        )
    )

    result = await ctx.pipe.process(inbound_message)

    assert "quickly" in result.text or "wait" in result.text.lower()
    ctx.router.route.assert_not_called()
    ctx.rag.retrieve.assert_not_called()


# ── Emergency ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_emergency_returns_emergency_response(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.router.route = AsyncMock(side_effect=EmergencyDetectedError(["tb"], ["coughing blood"]))

    result = await ctx.pipe.process(inbound_message)

    assert result.is_emergency is True
    assert "emergency" in result.text.lower() or "NEMA" in result.text
    ctx.rag.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_emergency_saves_conversation_turns(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.router.route = AsyncMock(side_effect=EmergencyDetectedError(["tb"], ["coughing blood"]))

    await ctx.pipe.process(inbound_message)

    assert ctx.convs.save_turn.call_count == 2


# ── Cache ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_skips_retrieval_and_generation(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    cached = CachedResponse(
        english_query="TB symptoms",
        disease_ids="tb",
        query_intent="symptoms",
        english_response="TB symptoms: cough, fever, sweats.",
        sources_json="[]",
        confidence="high",
        created_at=now,
        expires_at=now + 604_800,
    )
    ctx.cache.lookup = AsyncMock(return_value=cached)

    result = await ctx.pipe.process(inbound_message)

    assert result.cache_hit is True
    ctx.rag.retrieve.assert_not_called()
    ctx.gen.generate.assert_not_called()
    ctx.gen.from_cache.assert_called_once_with(cached, "en")


@pytest.mark.asyncio
async def test_personal_query_not_stored_in_cache(
    inbound_message, retrieval_result, llm_response, existing_user, now
):
    personal_route = RouteResult(
        disease_ids=["tb"],
        disease_confidence=0.9,
        query_intent=QueryIntent.SYMPTOMS,
        intent_confidence=0.85,
        is_personal=True,    # personal → must not be cached
    )
    ctx = _make_ctx(inbound_message, personal_route, retrieval_result, llm_response, existing_user, now)

    await ctx.pipe.process(inbound_message)

    ctx.cache.store.assert_not_called()


@pytest.mark.asyncio
async def test_low_confidence_response_not_cached(
    inbound_message, route_result, retrieval_result, existing_user, now
):
    low_conf_llm = LLMResponse(
        answer="I'm not sure.",
        confidence="low",    # low confidence → do not cache
        needs_professional=False,
        caveat=None,
        sources_used=[],
    )
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, low_conf_llm, existing_user, now)
    bot_resp = BotResponse(
        text="I'm not sure.", language_code="en", sources=[],
        confidence="low", needs_professional=False, is_emergency=False,
    )
    ctx.gen.generate = AsyncMock(return_value=(low_conf_llm, bot_resp))

    await ctx.pipe.process(inbound_message)

    ctx.cache.store.assert_not_called()


@pytest.mark.asyncio
async def test_needs_professional_response_not_cached(
    inbound_message, route_result, retrieval_result, existing_user, now
):
    prof_llm = LLMResponse(
        answer="Consult a doctor.",
        confidence="high",
        needs_professional=True,    # professional advice → do not cache
        caveat=None,
        sources_used=[],
    )
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, prof_llm, existing_user, now)
    bot_resp = BotResponse(
        text="Consult a doctor.", language_code="en", sources=[],
        confidence="high", needs_professional=True, is_emergency=False,
    )
    ctx.gen.generate = AsyncMock(return_value=(prof_llm, bot_resp))

    await ctx.pipe.process(inbound_message)

    ctx.cache.store.assert_not_called()


# ── Commands ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_help_command_returns_help_text(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.lang.parse = AsyncMock(
        return_value=_make_parsed(
            inbound_message, is_command=True, command="help", command_args=[]
        )
    )

    result = await ctx.pipe.process(inbound_message)

    assert _HELP_TEXT in result.text or "Welcome" in result.text
    ctx.router.route.assert_not_called()
    ctx.rag.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_about_command_returns_about_text(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.lang.parse = AsyncMock(
        return_value=_make_parsed(
            inbound_message, is_command=True, command="about", command_args=[]
        )
    )

    result = await ctx.pipe.process(inbound_message)

    assert "HealthBridgeAI" in result.text
    ctx.router.route.assert_not_called()


@pytest.mark.asyncio
async def test_language_command_updates_user(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.lang.parse = AsyncMock(
        return_value=_make_parsed(
            inbound_message, is_command=True, command="language", command_args=["yo"]
        )
    )

    result = await ctx.pipe.process(inbound_message)

    assert "yo" in result.text
    # upsert_user called twice: once in _get_or_create_user, once to save language change
    assert ctx.users.upsert_user.call_count == 2
    language_update_call: User = ctx.users.upsert_user.call_args_list[1].args[0]
    assert language_update_call.language_code == "yo"


@pytest.mark.asyncio
async def test_language_command_rejects_unsupported_language(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.lang.parse = AsyncMock(
        return_value=_make_parsed(
            inbound_message, is_command=True, command="language", command_args=["zulu"]
        )
    )

    result = await ctx.pipe.process(inbound_message)

    # Should contain an error/unsupported message, not silently succeed
    assert "❌" in result.text or "Unsupported" in result.text or "unsupported" in result.text.lower()


@pytest.mark.asyncio
async def test_audio_command_toggles_mode(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    """Starting with audio_mode=False → toggle sets it to True."""
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    # existing_user.audio_mode = False by default
    ctx.lang.parse = AsyncMock(
        return_value=_make_parsed(
            inbound_message, is_command=True, command="audio", command_args=[]
        )
    )

    result = await ctx.pipe.process(inbound_message)

    assert "🔊" in result.text
    toggled_user: User = ctx.users.upsert_user.call_args_list[1].args[0]
    assert toggled_user.audio_mode is True   # was False, now True


# ── Graceful error handling ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_failure_returns_error_response(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.router.route = AsyncMock(side_effect=RuntimeError("LLM is down"))

    result = await ctx.pipe.process(inbound_message)

    assert "trouble" in result.text.lower() or "try again" in result.text.lower()
    ctx.rag.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_failure_returns_error_response(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.rag.retrieve = AsyncMock(side_effect=RuntimeError("Pinecone is down"))

    result = await ctx.pipe.process(inbound_message)

    assert "trouble" in result.text.lower() or "try again" in result.text.lower()
    ctx.gen.generate.assert_not_called()


# ── Translation ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_yoruba_response_is_translated(
    inbound_message, route_result, retrieval_result, llm_response, existing_user, now
):
    """When the user's language is Yoruba, the final response is translated."""
    ctx = _make_ctx(inbound_message, route_result, retrieval_result, llm_response, existing_user, now)
    ctx.lang.parse = AsyncMock(
        return_value=_make_parsed(
            inbound_message,
            english_text="What are the symptoms of TB?",
            lang="yo",   # ← non-English
        )
    )
    ctx.lang.translate_from_english = AsyncMock(return_value="Àwọn àmì ìrora TB ni...")

    result = await ctx.pipe.process(inbound_message)

    ctx.lang.translate_from_english.assert_called_once()
    assert result.language_code == "yo"
