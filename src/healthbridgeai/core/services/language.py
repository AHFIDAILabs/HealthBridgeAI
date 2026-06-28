"""LanguageService — language detection, translation, and command extraction."""
from __future__ import annotations

import re

import structlog

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import LanguageNotSupportedError
from healthbridgeai.core.models.message import (
    InboundMessage,
    ParsedMessage,
    UserCommand,
)
from healthbridgeai.core.ports.llm import ILLMClient

log = structlog.get_logger(__name__)

_COMMAND_RE = re.compile(r"^/(\w+)(?:\s+(.*))?$", re.DOTALL)
_SUPPORTED_COMMANDS = {c.value for c in UserCommand}

_DETECT_SYSTEM = (
    "You are a language-detection assistant for a Nigerian health chatbot.\n"
    "Detect the primary language of the user's message.\n"
    "Supported: en (English), yo (Yoruba), ig (Igbo), ha (Hausa), pidgin (Nigerian Pidgin).\n"
    "Reply with ONLY the language code — one of: en yo ig ha pidgin.\n"
    "If uncertain or outside the list, reply: en"
)

_TRANSLATE_SYSTEM = (
    "You are a medical translator for a Nigerian health chatbot. "
    "Translate accurately, preserving medical terminology. "
    "Return only the translation — no preamble, no explanation."
)

_LANG_NAMES = {
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "pidgin": "Nigerian Pidgin",
}


class LanguageService:
    """Handles language detection, translation, and command parsing."""

    def __init__(self, llm: ILLMClient) -> None:
        self._llm = llm

    async def detect(self, text: str) -> str:
        """Return BCP-47 language code. Falls back to 'en' on any failure."""
        try:
            raw = await self._llm.complete(
                system=_DETECT_SYSTEM,
                user=text[:500],
                temperature=0.0,
                max_tokens=10,
            )
            code = raw.strip().lower().split()[0]
            return code if code in settings.SUPPORTED_LANGUAGES else "en"
        except Exception:
            log.warning("language.detect.failed", text_preview=text[:60])
            return "en"

    async def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate from a supported Nigerian language to English."""
        if source_lang == "en":
            return text
        lang_name = _LANG_NAMES.get(source_lang, source_lang)
        return await self._llm.complete(
            system=_TRANSLATE_SYSTEM,
            user=f"Translate this {lang_name} health message to English:\n\n{text}",
            temperature=0.1,
            max_tokens=1024,
        )

    async def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate an English response back to the user's language.

        Citation markers like [1] and [2] must be preserved as-is.
        """
        if target_lang == "en":
            return text
        lang_name = _LANG_NAMES.get(target_lang, target_lang)
        return await self._llm.complete(
            system=_TRANSLATE_SYSTEM,
            user=(
                f"Translate this health information to {lang_name}. "
                "Preserve numbered citations exactly as they appear, e.g. [1], [2].\n\n"
                + text
            ),
            temperature=0.1,
            max_tokens=2048,
        )

    async def parse(self, message: InboundMessage) -> ParsedMessage:
        """Full parse: extract commands, detect language, translate to English."""
        raw_text = (message.text or "").strip()

        # Command extraction (/help, /language yo, /audio, …)
        cmd_match = _COMMAND_RE.match(raw_text)
        if cmd_match:
            cmd_name = cmd_match.group(1).lower()
            raw_args = cmd_match.group(2) or ""
            cmd_args = raw_args.split() if raw_args.strip() else []
            if cmd_name in _SUPPORTED_COMMANDS:
                return ParsedMessage(
                    original=message,
                    language_code="en",
                    english_text=raw_text,
                    is_command=True,
                    command=cmd_name,
                    command_args=cmd_args,
                )

        lang_code = await self.detect(raw_text) if raw_text else "en"

        if lang_code not in settings.SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(lang_code, settings.SUPPORTED_LANGUAGES)

        english_text = (
            await self.translate_to_english(raw_text, lang_code)
            if raw_text and lang_code != "en"
            else raw_text
        )

        return ParsedMessage(
            original=message,
            language_code=lang_code,
            english_text=english_text,
        )
