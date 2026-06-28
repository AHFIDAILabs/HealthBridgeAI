"""AudioTranscriber — N-ATLAS → Whisper cascade for WhatsApp voice messages."""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Optional

import structlog

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import AudioError

log = structlog.get_logger(__name__)

# Supported language codes for transcription
_NATLAS_LANGS = {"ha", "ig", "yo"}
_NATLAS_MODELS = {"ha": "Hausa-ASR", "ig": "Igbo-ASR", "yo": "Yoruba-ASR"}


class AudioTranscriber:
    """
    Speech-to-text with two-tier cascade:
        1. N-ATLAS (NCAIR1/*-ASR on HuggingFace) — Nigerian-language ASR
        2. Whisper large-v3 — multilingual fallback

    Returns (transcribed_text, detected_language_code).
    All heavy calls run in a thread pool via run_in_executor.
    """

    async def transcribe(self, audio_bytes: bytes, lang_hint: str = "auto") -> tuple[str, str]:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio (OGG Opus from WhatsApp)
            lang_hint:   BCP-47 language code or 'auto'

        Returns:
            (text, language_code) — language_code matches SUPPORTED_LANGUAGES
        """
        ogg_path = await self._save_tmp(audio_bytes)
        try:
            return await self._transcribe_file(ogg_path, lang_hint)
        finally:
            _safe_remove(ogg_path)

    async def _transcribe_file(self, path: str, lang_hint: str) -> tuple[str, str]:
        loop = asyncio.get_event_loop()

        # N-ATLAS for known Nigerian languages (requires HuggingFace token)
        if lang_hint in _NATLAS_LANGS and settings.HUGGINGFACE_TOKEN:
            try:
                text = await loop.run_in_executor(None, self._natlas_sync, path, lang_hint)
                if text:
                    return text, lang_hint
            except Exception as exc:
                log.warning("audio.natlas.failed", lang=lang_hint, error=str(exc))

        # Whisper fallback — auto-detect if lang_hint is 'auto' or unsupported
        try:
            text, detected = await loop.run_in_executor(
                None, self._whisper_sync, path, lang_hint
            )
            # Map Whisper language codes to our supported set
            mapped = _map_lang(detected)
            return text, mapped
        except Exception as exc:
            log.error("audio.whisper.failed", error=str(exc))
            raise AudioError(f"All transcription methods failed: {exc}") from exc

    # ── N-ATLAS (sync) ────────────────────────────────────────────────────────

    def _natlas_sync(self, path: str, lang: str) -> str:
        import librosa
        from transformers import pipeline

        model_name = _NATLAS_MODELS[lang]
        pipe = pipeline(
            "automatic-speech-recognition",
            model=f"NCAIR1/{model_name}",
            token=settings.HUGGINGFACE_TOKEN,
        )
        audio, _ = librosa.load(path, sr=16000)
        result = pipe(audio)
        return (result.get("text") or "").strip()

    # ── Whisper (sync) ────────────────────────────────────────────────────────

    def _whisper_sync(self, path: str, lang_hint: str) -> tuple[str, str]:
        import whisper  # openai-whisper
        model = whisper.load_model("large-v3")
        options = {}
        if lang_hint not in ("auto", "") and lang_hint in settings.SUPPORTED_LANGUAGES:
            options["language"] = lang_hint
        result = model.transcribe(path, fp16=False, **options)
        return (result.get("text") or "").strip(), result.get("language", "en")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _save_tmp(self, audio_bytes: bytes) -> str:
        import uuid
        path = os.path.join(tempfile.gettempdir(), f"hb_audio_{uuid.uuid4().hex}.ogg")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path


def _map_lang(whisper_code: str) -> str:
    """Map Whisper/ISO 639-1 code to our supported language codes."""
    mapping = {
        "yoruba": "yo", "yo": "yo",
        "igbo": "ig", "ig": "ig",
        "hausa": "ha", "ha": "ha",
        "english": "en", "en": "en",
    }
    return mapping.get(whisper_code.lower(), "en")


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
