"""
WhatChimp WhatsApp adapter — implements IMessagingProvider.

WhatChimp is a WhatsApp Business Solution Provider (BSP) built on Meta's
WhatsApp Cloud API. Key facts from their API documentation:

  Base URL  : https://app.whatchimp.com/api/v1
  Auth      : apiToken passed as a form-data field (not a Bearer header).
              The Upload Media endpoint additionally accepts
              Authorization: Bearer <apiToken>.
  Webhooks  : WhatChimp forwards the standard Meta WhatsApp Cloud API
              webhook payload unchanged, verified with X-Hub-Signature-256
              (HMAC-SHA256 of the raw body using the app secret).

Endpoints used:
  POST /whatsapp/send          — send text message
  POST /whatsapp/send/file     — send media (audio, image, video, document)
  POST /whatsapp/upload/media  — upload bytes, get media_id back
  Media download               — inbound audio arrives with a Meta media URL;
                                 we download it directly using httpx.

Configuration (via Settings):
  WHATCHAMP_API_KEY          — your WhatChimp API token
  WHATCHAMP_API_URL          — base URL (default: https://app.whatchimp.com/api/v1)
  WHATCHAMP_PHONE_NUMBER     — E.164 sender number  (e.g. +2348012345678)
  WHATCHAMP_PHONE_NUMBER_ID  — Meta phone_number_id (required in all API calls)
  WHATCHAMP_WEBHOOK_SECRET   — app secret for X-Hub-Signature-256 (optional;
                               if empty, signature check is skipped with a warning)
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

_SEND_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_MEDIA_TIMEOUT = httpx.Timeout(60.0, connect=10.0)   # uploads / large downloads

# ── Webhook ───────────────────────────────────────────────────────────────────
# WhatChimp uses the standard Meta signature format:
#   X-Hub-Signature-256: sha256=<hex-digest>
_SIG_HEADER = "x-hub-signature-256"
_SIG_PREFIX = "sha256="


class WhatChampAdapter:
    """HTTP adapter for the WhatChimp WhatsApp BSP REST API."""

    def __init__(self) -> None:
        self._base = settings.WHATCHAMP_API_URL.rstrip("/")
        self._api_key = settings.WHATCHAMP_API_KEY
        self._phone_number_id = settings.WHATCHAMP_PHONE_NUMBER_ID
        self._phone = settings.WHATCHAMP_PHONE_NUMBER
        self._secret = settings.WHATCHAMP_WEBHOOK_SECRET  # may be empty

    # ─────────────────────────────────────────────────────────────────────────
    # Inbound
    # ─────────────────────────────────────────────────────────────────────────

    async def validate_webhook(self, raw_body: bytes, headers: dict) -> bool:
        """
        Verify X-Hub-Signature-256 sent by Meta / WhatChimp.

        If WHATCHAMP_WEBHOOK_SECRET is empty (not configured), the check is
        skipped and a warning is logged — useful during local development but
        must be configured in production.
        """
        if not self._secret:
            log.warning(
                "whatchamp.webhook_secret_not_set",
                hint="Set WHATCHAMP_WEBHOOK_SECRET to enable signature verification",
            )
            return True

        raw_sig = headers.get(_SIG_HEADER, "")
        if not raw_sig.startswith(_SIG_PREFIX):
            log.warning("whatchamp.webhook_missing_signature")
            raise WebhookAuthError("Missing X-Hub-Signature-256 header")

        received = raw_sig[len(_SIG_PREFIX):]
        expected = hmac.new(
            self._secret.encode(),
            raw_body,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(received, expected):
            log.warning("whatchamp.webhook_signature_invalid")
            raise WebhookAuthError("X-Hub-Signature-256 mismatch")

        return True

    async def parse_webhook(
        self, payload: bytes, headers: dict
    ) -> Optional[InboundMessage]:
        """
        Parse a WhatChimp/Meta webhook event into InboundMessage.

        WhatChimp forwards the standard Meta WhatsApp Cloud API payload:

          {
            "object": "whatsapp_business_account",
            "entry": [{
              "changes": [{
                "value": {
                  "metadata": {"phone_number_id": "..."},
                  "messages": [{
                    "id": "wamid.xxx",
                    "from": "2348012345678",
                    "timestamp": "1700000000",
                    "type": "text" | "audio" | "image" | ...,
                    "text": {"body": "..."},           -- for text
                    "audio": {"id": "...", "mime_type": "..."},  -- for audio
                  }],
                  "statuses": [...]   -- delivery receipts, not messages
                }
              }]
            }]
          }

        Returns None for non-message events (status updates, read receipts).
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.error("whatchamp.parse_webhook.invalid_json")
            return None

        # ── Navigate the Meta envelope ────────────────────────────────────────
        try:
            value = data["entry"][0]["changes"][0]["value"]
        except (KeyError, IndexError):
            log.debug("whatchamp.parse_webhook.no_value_block")
            return None

        messages = value.get("messages", [])
        if not messages:
            # Status update / read receipt — acknowledge silently
            return None

        msg = messages[0]
        msg_type_raw = msg.get("type", "")
        from_number = "+" + msg.get("from", "").lstrip("+")
        message_id = msg.get("id", "")
        timestamp = int(msg.get("timestamp", 0))

        # ── Map to MessageType ────────────────────────────────────────────────
        if msg_type_raw == "text":
            return InboundMessage(
                message_id=message_id,
                from_number=from_number,
                type=MessageType.TEXT,
                text=msg["text"]["body"],
                timestamp=timestamp,
            )

        if msg_type_raw == "audio":
            audio = msg.get("audio", {})
            return InboundMessage(
                message_id=message_id,
                from_number=from_number,
                type=MessageType.AUDIO,
                media=MediaInfo(
                    media_id=audio.get("id", ""),
                    mime_type=audio.get("mime_type", "audio/ogg"),
                    url=audio.get("url"),  # included if WhatChimp pre-fetches the URL
                ),
                timestamp=timestamp,
            )

        if msg_type_raw in {"image", "video", "document", "sticker"}:
            media_block = msg.get(msg_type_raw, {})
            return InboundMessage(
                message_id=message_id,
                from_number=from_number,
                type=MessageType.IMAGE if msg_type_raw == "image" else MessageType.TEXT,
                media=MediaInfo(
                    media_id=media_block.get("id", ""),
                    mime_type=media_block.get("mime_type", ""),
                    url=media_block.get("url"),
                ),
                timestamp=timestamp,
            )

        # Unsupported type (location, contacts, reaction, etc.)
        log.info("whatchamp.parse_webhook.unsupported_type", msg_type=msg_type_raw)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Outbound
    # ─────────────────────────────────────────────────────────────────────────

    async def send_text(self, to: str, text: str) -> str:
        """
        Send a plain-text WhatsApp message via WhatChimp.

        POST /whatsapp/send
          apiToken, phone_number_id, message, phone_number (no leading +)
        """
        phone_number = to.lstrip("+")
        async with httpx.AsyncClient(timeout=_SEND_TIMEOUT) as client:
            resp = await client.post(
                f"{self._base}/whatsapp/send",
                data={
                    "apiToken": self._api_key,
                    "phone_number_id": self._phone_number_id,
                    "message": text,
                    "phone_number": phone_number,
                },
            )

        body = _parse_response(resp, "send_text")
        return body.get("wa_message_id", "")

    async def send_audio(self, to: str, audio_url: str, caption: str = "") -> str:
        """
        Send an audio file by public HTTPS URL via WhatChimp.

        POST /whatsapp/send/file
          apiToken, phone_number_id, phone_number, media_url, media_type=audio
        """
        phone_number = to.lstrip("+")
        data: dict = {
            "apiToken": self._api_key,
            "phone_number_id": self._phone_number_id,
            "phone_number": phone_number,
            "media_url": audio_url,
            "media_type": "audio",
        }
        if caption:
            data["media_caption_text"] = caption

        async with httpx.AsyncClient(timeout=_SEND_TIMEOUT) as client:
            resp = await client.post(
                f"{self._base}/whatsapp/send/file",
                data=data,
            )

        body = _parse_response(resp, "send_audio")
        return body.get("wa_message_id", "")

    async def upload_media(self, file_bytes: bytes, mime_type: str, filename: str) -> tuple[str, str]:
        """
        Upload raw bytes to WhatChimp and return (media_id, media_type).

        POST /whatsapp/upload/media  (multipart/form-data)
          Authorization: Bearer <apiToken>
          phone_number_id, media_file
        """
        async with httpx.AsyncClient(timeout=_MEDIA_TIMEOUT) as client:
            resp = await client.post(
                f"{self._base}/whatsapp/upload/media",
                headers={"Authorization": f"Bearer {self._api_key}"},
                data={"phone_number_id": self._phone_number_id},
                files={"media_file": (filename, file_bytes, mime_type)},
            )

        body = _parse_response(resp, "upload_media")
        return body.get("media_id", ""), body.get("media_type", "")

    async def download_media(self, media_id_or_url: str) -> bytes:
        """
        Download inbound media (e.g. voice note) from a URL or Meta media ID.

        WhatChimp forwards Meta media IDs in webhook events. If WhatChimp
        has pre-resolved the URL it will be stored in MediaInfo.media_url;
        pass that directly here. If only a raw Meta media ID is available,
        callers should use the Meta Graph API directly (requires access_token).

        This implementation handles two cases:
          - If the argument starts with "http", download directly.
          - Otherwise treat as an opaque reference and attempt to download
            from WhatChimp's platform URL pattern.
        """
        async with httpx.AsyncClient(timeout=_MEDIA_TIMEOUT) as client:
            if media_id_or_url.startswith("http"):
                resp = await client.get(
                    media_id_or_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    follow_redirects=True,
                )
            else:
                # Fallback: assume WhatChimp exposes media via the upload endpoint ID
                resp = await client.get(
                    f"{self._base}/whatsapp/media/{media_id_or_url}",
                    params={"apiToken": self._api_key},
                )

        if resp.status_code >= 400:
            raise ExternalServiceError(
                "WhatChimp",
                f"download_media failed: {resp.status_code} — {resp.text[:200]}",
            )
        return resp.content


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_response(resp: httpx.Response, operation: str) -> dict:
    """
    Raise ExternalServiceError on HTTP errors or WhatChimp status=0 responses.
    Returns the parsed JSON body on success.
    """
    if resp.status_code >= 400:
        raise ExternalServiceError(
            "WhatChimp",
            f"{operation} HTTP {resp.status_code}: {resp.text[:300]}",
        )
    try:
        body = resp.json()
    except Exception:
        raise ExternalServiceError("WhatChimp", f"{operation} returned non-JSON: {resp.text[:200]}")

    if str(body.get("status", "1")) == "0":
        raise ExternalServiceError(
            "WhatChimp",
            f"{operation} error: {body.get('message', 'unknown')}",
        )
    return body
