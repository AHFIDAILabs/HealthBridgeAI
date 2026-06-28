"""MessagePipeline — end-to-end orchestrator for inbound WhatsApp messages."""
from __future__ import annotations

import time
from typing import Optional

import structlog

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import EmergencyDetectedError, RateLimitError
from healthbridgeai.core.models.message import InboundMessage, ParsedMessage, UserCommand
from healthbridgeai.core.models.response import BotResponse, CachedResponse
from healthbridgeai.core.models.user import ConversationTurn, User
from healthbridgeai.core.ports.cache import IResponseCache
from healthbridgeai.core.ports.llm import ILLMClient
from healthbridgeai.core.ports.storage import IConversationStore, IUserStore
from healthbridgeai.core.services.generator import ResponseGenerator
from healthbridgeai.core.services.language import LanguageService
from healthbridgeai.core.services.rag import RAGService
from healthbridgeai.core.services.router import RouterService

log = structlog.get_logger(__name__)

_EMERGENCY_TEXT = (
    "🚨 *This sounds like a medical emergency.*\n\n"
    "Please go to your nearest health facility immediately or call for help.\n\n"
    "*Nigeria Emergency Lines:*\n"
    "• NEMA: 0800-2255-6362\n"
    "• Lagos Emergency: 767 or 112\n\n"
    "_Stay with the person and keep them calm until help arrives._"
)

_HELP_TEXT = (
    "👋 *Welcome to HealthBridgeAI!*\n\n"
    "I can answer questions about TB, HIV/AIDS, and Malaria.\n\n"
    "*Commands:*\n"
    "• /help — Show this message\n"
    "• /language [en|yo|ig|ha|pidgin] — Change language\n"
    "• /audio — Toggle audio responses on/off\n"
    "• /about — About this service\n"
    "• /feedback — Send feedback\n\n"
    "_Ask me any health question in your preferred language._"
)

_ABOUT_TEXT = (
    "🏥 *HealthBridgeAI*\n\n"
    "An AI-powered health information companion for West Africa.\n"
    "Covering TB, HIV/AIDS, and Malaria with sources from WHO, CDC, USAID, and national guidelines.\n\n"
    "_Built by AHFID AI Labs — Not a substitute for professional medical advice_"
)


class MessagePipeline:
    """
    Processes one inbound WhatsApp message end-to-end.

    All dependencies are injected; the pipeline imports no infrastructure directly.
    Shared between the API webhook handler (pub/sub publish path) and the
    Pub/Sub processor service (full async processing path).
    """

    def __init__(
        self,
        language_svc: LanguageService,
        router_svc: RouterService,
        rag_svc: RAGService,
        generator: ResponseGenerator,
        llm: ILLMClient,
        user_store: IUserStore,
        conv_store: IConversationStore,
        cache: IResponseCache,
    ) -> None:
        self._lang = language_svc
        self._router = router_svc
        self._rag = rag_svc
        self._gen = generator
        self._llm = llm
        self._users = user_store
        self._convs = conv_store
        self._cache = cache

    async def process(self, message: InboundMessage) -> BotResponse:
        """
        Full message processing pipeline. Returns a BotResponse ready to send.

        Never raises — all domain errors produce a graceful fallback BotResponse.
        """
        # ── 0. Get/create user + rate limit ───────────────────────────────────
        user = await self._get_or_create_user(message.from_number, message.timestamp)
        phone_hash = user.phone_hash

        try:
            rate = await self._users.check_rate_limit(
                phone_hash=phone_hash,
                limit=settings.RATE_LIMIT_MESSAGES_PER_MINUTE,
                window_seconds=60,
            )
        except Exception as exc:
            log.error("pipeline.rate_limit_check_failed", error=str(exc), phone_hash=phone_hash)
            rate = None

        if rate and rate.is_exceeded(settings.RATE_LIMIT_MESSAGES_PER_MINUTE):
            log.warning("pipeline.rate_limited", phone_hash=phone_hash)
            return _rate_limit_response()

        # ── 1. Parse message (language detection + command extraction) ────────
        try:
            parsed = await self._lang.parse(message)
        except Exception as exc:
            log.error("pipeline.parse_failed", error=str(exc), phone_hash=phone_hash)
            return _error_response()

        log.info(
            "pipeline.parsed",
            lang=parsed.language_code,
            is_command=parsed.is_command,
            phone_hash=phone_hash,
        )

        # ── 2. Handle commands ────────────────────────────────────────────────
        if parsed.is_command:
            return await self._handle_command(parsed, user, phone_hash)

        english_query = parsed.english_text.strip()
        if not english_query:
            return _empty_input_response(parsed.language_code)

        # ── 3. Pre-embed for cache lookup (best-effort; skip if LLM down) ─────
        query_embedding: list[float] = []
        try:
            query_embedding = (await self._llm.embed([english_query]))[0]
        except Exception as exc:
            log.warning("pipeline.embed_failed", error=str(exc), phone_hash=phone_hash)

        # ── 4. Route (disease + intent classification) ────────────────────────
        try:
            route = await self._router.route(english_query, phone_hash)
        except EmergencyDetectedError:
            bot = BotResponse(
                text=_EMERGENCY_TEXT,
                language_code=parsed.language_code,
                sources=[],
                confidence="high",
                needs_professional=True,
                is_emergency=True,
                cache_hit=False,
            )
            await self._save_turns(phone_hash, parsed, bot.text, [])
            return bot
        except Exception as exc:
            log.error("pipeline.route_failed", error=str(exc), phone_hash=phone_hash)
            return _error_response()

        # ── 5. Semantic cache lookup ──────────────────────────────────────────
        # Never cache: personal queries, emergencies (already handled above)
        cache_eligible = not (route.is_personal or route.is_general_health and not route.disease_ids)
        cached: Optional[CachedResponse] = None
        if cache_eligible and query_embedding:
            try:
                cached = await self._cache.lookup(
                    english_query=english_query,
                    query_embedding=query_embedding,
                    disease_ids=route.disease_ids,
                    threshold=settings.SEMANTIC_CACHE_THRESHOLD,
                )
            except Exception as exc:
                log.warning("pipeline.cache_lookup_failed", error=str(exc), phone_hash=phone_hash)

        if cached:
            log.info("pipeline.cache_hit", phone_hash=phone_hash)
            bot = self._gen.from_cache(cached, parsed.language_code)
            if parsed.language_code != "en":
                try:
                    bot.text = await self._lang.translate_from_english(bot.text, parsed.language_code)
                    bot.language_code = parsed.language_code
                except Exception:
                    pass
            await self._save_turns(phone_hash, parsed, bot.text, route.disease_ids)
            return bot

        # ── 6. RAG retrieval ──────────────────────────────────────────────────
        try:
            retrieval = await self._rag.retrieve(english_query, route, phone_hash)
        except Exception as exc:
            log.error("pipeline.retrieval_failed", error=str(exc), phone_hash=phone_hash)
            return _error_response()

        # ── 7. Generate response ──────────────────────────────────────────────
        try:
            llm_resp, bot = await self._gen.generate(english_query, retrieval, route, phone_hash)
        except Exception as exc:
            log.error("pipeline.generation_failed", error=str(exc), phone_hash=phone_hash)
            return _error_response()

        # ── 8. Translate back to user's language ──────────────────────────────
        if parsed.language_code != "en":
            try:
                bot.text = await self._lang.translate_from_english(bot.text, parsed.language_code)
                bot.language_code = parsed.language_code
            except Exception as exc:
                log.warning("pipeline.translation_failed", error=str(exc), phone_hash=phone_hash)

        # ── 9. Store in semantic cache (enforce never-cache rules) ────────────
        should_cache = (
            cache_eligible
            and query_embedding
            and llm_resp.confidence != "low"
            and not llm_resp.needs_professional
            and not route.is_personal
        )
        if should_cache:
            try:
                entry = CachedResponse(
                    english_query=english_query,
                    disease_ids=",".join(sorted(route.disease_ids)),
                    query_intent=route.query_intent.value,
                    english_response=llm_resp.answer,
                    sources_json=CachedResponse.sources_to_json(bot.sources),
                    confidence=llm_resp.confidence,
                    created_at=int(time.time()),
                    expires_at=int(time.time()) + settings.CACHE_TTL_DAYS * 86_400,
                )
                await self._cache.store(english_query, query_embedding, entry)
            except Exception as exc:
                log.warning("pipeline.cache_store_failed", error=str(exc), phone_hash=phone_hash)

        # ── 10. Persist conversation history ──────────────────────────────────
        await self._save_turns(phone_hash, parsed, bot.text, route.disease_ids)

        return bot

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_or_create_user(self, phone_number: str, timestamp: int) -> User:
        user = await self._users.get_user(phone_number)
        if user is None:
            user = User(phone_number=phone_number, created_at=timestamp, last_seen_at=timestamp)
        else:
            user = user.model_copy(
                update={"last_seen_at": timestamp, "message_count": user.message_count + 1}
            )
        await self._users.upsert_user(user)
        return user

    async def _handle_command(
        self, parsed: ParsedMessage, user: User, phone_hash: str
    ) -> BotResponse:
        cmd = parsed.command or ""
        args = parsed.command_args

        if cmd == UserCommand.HELP.value:
            text = _HELP_TEXT
        elif cmd == UserCommand.ABOUT.value:
            text = _ABOUT_TEXT
        elif cmd == UserCommand.LANGUAGE.value:
            lang = args[0].lower() if args else "en"
            if lang in settings.SUPPORTED_LANGUAGES:
                updated = user.model_copy(update={"language_code": lang})
                await self._users.upsert_user(updated)
                text = f"✅ Language set to *{lang}*."
            else:
                supported = ", ".join(settings.SUPPORTED_LANGUAGES)
                text = f"❌ Unsupported language. Choose: {supported}"
        elif cmd == UserCommand.AUDIO.value:
            updated = user.model_copy(update={"audio_mode": not user.audio_mode})
            await self._users.upsert_user(updated)
            state = "enabled" if updated.audio_mode else "disabled"
            text = f"🔊 Audio responses {state}."
        elif cmd == UserCommand.FEEDBACK.value:
            text = "📝 Thank you! Please share your feedback as a message in this chat."
        else:
            text = _HELP_TEXT

        log.info("pipeline.command", command=cmd, phone_hash=phone_hash)
        return BotResponse(
            text=text,
            language_code=parsed.language_code,
            sources=[],
            confidence="high",
            needs_professional=False,
            is_emergency=False,
            cache_hit=False,
        )

    async def _save_turns(
        self,
        phone_hash: str,
        parsed: ParsedMessage,
        bot_text: str,
        disease_ids: list[str],
    ) -> None:
        now = int(time.time())
        user_content = parsed.english_text or parsed.original.text or ""
        try:
            await self._convs.save_turn(
                phone_hash,
                ConversationTurn(
                    role="user",
                    content=user_content,
                    timestamp=now,
                    disease_ids=disease_ids,
                    language_code=parsed.language_code,
                ),
            )
            await self._convs.save_turn(
                phone_hash,
                ConversationTurn(
                    role="assistant",
                    content=bot_text,
                    timestamp=now,
                    disease_ids=disease_ids,
                    language_code=parsed.language_code,
                ),
            )
        except Exception as exc:
            log.warning("pipeline.save_turns_failed", error=str(exc), phone_hash=phone_hash)


# ── Fallback responses ────────────────────────────────────────────────────────

def _error_response() -> BotResponse:
    return BotResponse(
        text=(
            "⚠️ I'm having trouble processing your request right now. "
            "Please try again in a moment, or contact your nearest health facility."
        ),
        language_code="en",
        sources=[],
        confidence="low",
        needs_professional=False,
        is_emergency=False,
    )


def _rate_limit_response() -> BotResponse:
    return BotResponse(
        text="⏳ You're sending messages too quickly. Please wait a moment and try again.",
        language_code="en",
        sources=[],
        confidence="high",
        needs_professional=False,
        is_emergency=False,
    )


def _empty_input_response(language_code: str) -> BotResponse:
    return BotResponse(
        text="Please send a health question and I'll do my best to help. Type /help for options.",
        language_code=language_code,
        sources=[],
        confidence="high",
        needs_professional=False,
        is_emergency=False,
    )
