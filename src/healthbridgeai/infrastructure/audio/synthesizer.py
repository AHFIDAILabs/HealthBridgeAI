"""AudioSynthesizer — YarnGPT → MMS → gTTS cascade, uploads result to GCS."""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from typing import Optional

import httpx
import structlog

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import AudioError

log = structlog.get_logger(__name__)

_YARNGPT_URL = "https://yarngpt.ai/api/v1/tts"
_YARNGPT_VOICE = "Idera"  # supports en, yo, ig, ha

# Map internal language codes to gTTS language codes and TLDs
_GTTS_MAP = {
    "en": ("en", "com.ng"),
    "yo": ("yo", "com.ng"),
    "ig": ("ig", "com.ng"),
    "ha": ("ha", "com.ng"),
    "pidgin": ("en", "com.ng"),  # gTTS has no pidgin — use English voice
}


class AudioSynthesizer:
    """
    Text-to-speech with three-tier cascade:
        1. YarnGPT — primary (Nigerian voices, all supported languages)
        2. MMS (facebook/mms-tts-*) — secondary for yo/ha (local model)
        3. gTTS — last resort (global voices)

    synthesize() is fully async; heavy sync calls run in thread pool.
    Returns a local OGG Opus file path ready for upload to GCS.
    """

    async def synthesize(self, text: str, lang: str) -> str:
        """Synthesize speech to a temp OGG file and return its path."""
        text = text.strip()
        if not text:
            raise AudioError("Empty text provided for TTS")

        # 1. YarnGPT
        if settings.YARNGPT_API_KEY and lang in ("en", "yo", "ig", "ha"):
            try:
                return await self._yarngpt(text, lang)
            except Exception as exc:
                log.warning("audio.yarngpt.failed", lang=lang, error=str(exc))

        # 2. MMS (yo / ha only, local model)
        if lang in ("yo", "ha") and settings.HUGGINGFACE_TOKEN:
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._mms_sync, text, lang
                )
            except Exception as exc:
                log.warning("audio.mms.failed", lang=lang, error=str(exc))

        # 3. gTTS fallback
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._gtts_sync, text, lang
            )
        except Exception as exc:
            log.error("audio.gtts.failed", lang=lang, error=str(exc))
            raise AudioError(f"All TTS providers failed for lang={lang}") from exc

    # ── YarnGPT ───────────────────────────────────────────────────────────────

    async def _yarngpt(self, text: str, lang: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                _YARNGPT_URL,
                headers={"Authorization": f"Bearer {settings.YARNGPT_API_KEY}"},
                json={"text": text, "voice": _YARNGPT_VOICE},
            )
        if resp.status_code != 200:
            raise AudioError(f"YarnGPT HTTP {resp.status_code}: {resp.text[:200]}")

        wav_path = _tmp_path("yarngpt", ".wav")
        with open(wav_path, "wb") as f:
            f.write(resp.content)
        return _wav_to_ogg(wav_path)

    # ── MMS (sync, runs in executor) ──────────────────────────────────────────

    def _mms_sync(self, text: str, lang: str) -> str:
        import torch
        import torchaudio
        from transformers import AutoTokenizer, VitsModel

        lang_map = {"yo": "yor", "ha": "hau"}
        model_name = f"facebook/mms-tts-{lang_map[lang]}"
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        wav_path = _tmp_path("mms", ".wav")
        torchaudio.save(wav_path, waveform, sample_rate=16000)
        return _wav_to_ogg(wav_path)

    # ── gTTS (sync, runs in executor) ─────────────────────────────────────────

    def _gtts_sync(self, text: str, lang: str) -> str:
        from gtts import gTTS
        tts_lang, tts_tld = _GTTS_MAP.get(lang, ("en", "com.ng"))
        tts = gTTS(text=text, lang=tts_lang, tld=tts_tld, slow=False)
        mp3_path = _tmp_path("gtts", ".mp3")
        tts.save(mp3_path)
        return _mp3_to_ogg(mp3_path)


# ── Audio conversion helpers ──────────────────────────────────────────────────

def _tmp_path(prefix: str, suffix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"hb_{prefix}_{uuid.uuid4().hex}{suffix}")


def _wav_to_ogg(wav_path: str) -> str:
    ogg_path = wav_path.replace(".wav", ".ogg")
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(
            ogg_path, format="ogg", codec="libopus", parameters=["-strict", "-2"]
        )
        os.remove(wav_path)
        return ogg_path
    except Exception:
        return wav_path  # return WAV if conversion fails


def _mp3_to_ogg(mp3_path: str) -> str:
    ogg_path = mp3_path.replace(".mp3", ".ogg")
    try:
        from pydub import AudioSegment
        AudioSegment.from_mp3(mp3_path).export(
            ogg_path, format="ogg", codec="libopus", parameters=["-strict", "-2"]
        )
        os.remove(mp3_path)
        return ogg_path
    except Exception:
        return mp3_path
