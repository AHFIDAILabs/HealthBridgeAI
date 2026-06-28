"""
WhatChimp WhatsApp adapter — implements IMessagingProvider.

WhatChimp is a WhatsApp Business Solution Provider (BSP) with a proprietary HTTP API.
All provider-specific details are encapsulated here; the core pipeline has no
knowledge of WhatChimp — it only sees the IMessagingProvider Protocol.

Configuration (via Settings):
    WHATCHAMP_API_KEY       — bearer token for outbound API calls
    WHATCHAMP_API_URL       — base URL, default https://api.whatchamp.com/v1
    WHATCHAMP_PHONE_NUMBER  — E.164 sender number (your WhatsApp Business number)
    WHATCHAMP_WEBHOOK_SECRET — HMAC-SHA256 key used to verify inbound events

Full implementation in Step 5 (infrastructure layer).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Optional

import httpx
import structlog

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import ExternalServiceError, WebhookAuthError
from healthbridgeai.core.models.message import InboundMessage, MediaInfo, MessageType

log = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class WhatChampAdapter:
    """Generic HTTP adapter wrapping the WhatChimp REST API."""

    def __init__(self) -> None:
        self._base_url = settings.WHATCHAMP_API_URL.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.WHATCHAMP_API_KEY}",
            "Content-Type": "application/json",
        }
        self._phone = settings.WHATCHAMP_PHONE_NUMBER
        self._secret = settings.WHATCHAMP_WEBHOOK_SECRET

    # ── Inbound ───────────────────────────────────────────────────────────────

    async def validate_webhook(self, raw_body: bytes, headers: dict) -> bool:
        """Verify the HMAC-SHA256 signature sent by WhatChimp."""
        signature = headers.get("x-whatchamp-signature", "")
        expected = hmac.new(
            self._secret.encode(),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            log.warning("whatchamp.webhook_signature_invalid")
            raise WebhookAuthError("WhatChimp signature mismatch")
        return True

    async def parse_webhook(
        self, payload: bytes, headers: dict
    ) -> Optional[InboundMessage]:
        """
        Parse a WhatChimp inbound webhook event into InboundMessage.

        WhatChimp payload shape is proprietary. This stub returns None until
        the full adapter is implemented in Step 5.  The shape below is a
        reasonable assumption for a WhatsApp Cloud API-compatible BSP.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.error("whatchamp.parse_webhook.invalid_json")
            return None

        # TODO(Step 5): map WhatChimp-specific fields to InboundMessage
        # Placeholder — prevents import errors during integration tests
        return None

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def send_text(self, to: str, text: str) -> str:
        """Send a plain-text WhatsApp message. Returns the provider message ID."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/messages",
                headers=self._headers,
                json={
                    "to": to,
                    "from": self._phone,
                    "type": "text",
                    "text": {"body": text},
                },
            )
        if resp.status_code >= 400:
            raise ExternalServiceError(
                "WhatChimp",
                f"send_text failed: {resp.status_code} {resp.text[:200]}",
            )
        return resp.json().get("message_id", "")

    async def send_audio(self, to: str, audio_url: str, caption: str = "") -> str:
        """Send an audio message by URL. Returns the provider message ID."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/messages",
                headers=self._headers,
                json={
                    "to": to,
                    "from": self._phone,
                    "type": "audio",
                    "audio": {"url": audio_url},
                    "caption": caption,
                },
            )
        if resp.status_code >= 400:
            raise ExternalServiceError(
                "WhatChimp",
                f"send_audio failed: {resp.status_code} {resp.text[:200]}",
            )
        return resp.json().get("message_id", "")

    async def download_media(self, media_id: str) -> bytes:
        """Download binary media by WhatChimp media ID."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{self._base_url}/media/{media_id}",
                headers=self._headers,
            )
        if resp.status_code >= 400:
            raise ExternalServiceError(
                "WhatChimp",
                f"download_media failed: {resp.status_code} {resp.text[:200]}",
            )
        return resp.content
