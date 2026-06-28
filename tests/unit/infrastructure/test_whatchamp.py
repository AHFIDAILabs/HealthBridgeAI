"""
Unit tests for WhatChampAdapter.

Pure-logic tests (validate_webhook, parse_webhook) require no HTTP mocking.
Outbound tests (send_text, send_audio, download_media) mock httpx.AsyncClient.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from healthbridgeai.core.exceptions import ExternalServiceError, WebhookAuthError
from healthbridgeai.core.models.message import MessageType
from healthbridgeai.infrastructure.messaging.whatchamp import WhatChampAdapter


# ── Fixtures ──────────────────────────────────────────────────────────────────

_SECRET = "test-app-secret"
_BOT_NUMBER = "+2348000000000"
_USER_NUMBER = "+2348012345678"

# Minimal valid Meta/WhatChimp webhook envelope
_TEXT_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [{
        "changes": [{
            "value": {
                "metadata": {
                    "display_phone_number": _BOT_NUMBER,
                    "phone_number_id": "987654321",
                },
                "messages": [{
                    "id": "wamid.test001",
                    "from": "2348012345678",   # Meta omits the leading +
                    "timestamp": "1700000000",
                    "type": "text",
                    "text": {"body": "What are the symptoms of TB?"},
                }],
            }
        }]
    }]
}

_AUDIO_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [{
        "changes": [{
            "value": {
                "metadata": {
                    "display_phone_number": _BOT_NUMBER,
                    "phone_number_id": "987654321",
                },
                "messages": [{
                    "id": "wamid.audio001",
                    "from": "2348012345678",
                    "timestamp": "1700000001",
                    "type": "audio",
                    "audio": {
                        "id": "media-abc123",
                        "mime_type": "audio/ogg; codecs=opus",
                        "url": "https://cdn.whatsapp.net/media-abc123",
                    },
                }],
            }
        }]
    }]
}

_STATUS_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [{
        "changes": [{
            "value": {
                "metadata": {"display_phone_number": _BOT_NUMBER},
                "statuses": [{"id": "wamid.sent001", "status": "delivered"}],
                # No "messages" key — this is a delivery receipt
            }
        }]
    }]
}


@pytest.fixture
def adapter():
    """WhatChampAdapter with settings injected from conftest env defaults."""
    return WhatChampAdapter()


def _sign(body: bytes, secret: str = _SECRET) -> str:
    """Compute the correct X-Hub-Signature-256 for a payload."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _httpx_mock(status: int = 200, body: dict | None = None, content: bytes = b"data"):
    """Return a pre-wired AsyncClient mock for patching httpx.AsyncClient."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.text = json.dumps(body) if body else ""
    mock_resp.content = content
    if body is not None:
        mock_resp.json.return_value = body
    else:
        mock_resp.json.side_effect = ValueError("no json body")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return mock_client


# ── validate_webhook ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_webhook_valid_signature(adapter):
    body = b'{"event": "test"}'
    sig = _sign(body)
    adapter._secret = _SECRET
    result = await adapter.validate_webhook(body, {"x-hub-signature-256": sig})
    assert result is True


@pytest.mark.asyncio
async def test_validate_webhook_invalid_signature_raises(adapter):
    body = b'{"event": "test"}'
    adapter._secret = _SECRET
    with pytest.raises(WebhookAuthError):
        await adapter.validate_webhook(body, {"x-hub-signature-256": "sha256=deadbeef"})


@pytest.mark.asyncio
async def test_validate_webhook_missing_header_raises(adapter):
    adapter._secret = _SECRET
    with pytest.raises(WebhookAuthError):
        await adapter.validate_webhook(b"body", {})


@pytest.mark.asyncio
async def test_validate_webhook_no_secret_returns_true(adapter):
    """Empty WHATCHAMP_WEBHOOK_SECRET → skip check (dev mode)."""
    adapter._secret = ""
    result = await adapter.validate_webhook(b"body", {})
    assert result is True


@pytest.mark.asyncio
async def test_validate_webhook_wrong_prefix_raises(adapter):
    adapter._secret = _SECRET
    with pytest.raises(WebhookAuthError):
        await adapter.validate_webhook(b"body", {"x-hub-signature-256": "md5=abc"})


# ── parse_webhook — text ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_webhook_text_returns_inbound_message(adapter):
    raw = json.dumps(_TEXT_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})

    assert msg is not None
    assert msg.type == MessageType.TEXT
    assert msg.text == "What are the symptoms of TB?"


@pytest.mark.asyncio
async def test_parse_webhook_text_from_number_has_plus(adapter):
    """Meta strips the leading +; adapter must restore it."""
    raw = json.dumps(_TEXT_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})
    assert msg.from_number == _USER_NUMBER


@pytest.mark.asyncio
async def test_parse_webhook_text_to_number_extracted(adapter):
    """to_number is extracted from metadata.display_phone_number."""
    raw = json.dumps(_TEXT_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})
    assert msg.to_number == _BOT_NUMBER


@pytest.mark.asyncio
async def test_parse_webhook_text_message_id(adapter):
    raw = json.dumps(_TEXT_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})
    assert msg.message_id == "wamid.test001"


@pytest.mark.asyncio
async def test_parse_webhook_text_timestamp(adapter):
    raw = json.dumps(_TEXT_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})
    assert msg.timestamp == 1700000000


# ── parse_webhook — audio ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_webhook_audio_returns_inbound_message(adapter):
    raw = json.dumps(_AUDIO_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})

    assert msg is not None
    assert msg.type == MessageType.AUDIO
    assert msg.media is not None
    assert msg.media.media_id == "media-abc123"


@pytest.mark.asyncio
async def test_parse_webhook_audio_url_preserved(adapter):
    """If WhatChimp pre-fetches the media URL, it should be stored."""
    raw = json.dumps(_AUDIO_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})
    assert msg.media.url == "https://cdn.whatsapp.net/media-abc123"


@pytest.mark.asyncio
async def test_parse_webhook_audio_mime_type(adapter):
    raw = json.dumps(_AUDIO_PAYLOAD).encode()
    msg = await adapter.parse_webhook(raw, {})
    assert "ogg" in msg.media.mime_type


# ── parse_webhook — edge cases ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_webhook_status_update_returns_none(adapter):
    """Delivery receipts have no messages key → return None."""
    raw = json.dumps(_STATUS_PAYLOAD).encode()
    assert await adapter.parse_webhook(raw, {}) is None


@pytest.mark.asyncio
async def test_parse_webhook_empty_messages_returns_none(adapter):
    payload = {
        "entry": [{"changes": [{"value": {"metadata": {}, "messages": []}}]}]
    }
    raw = json.dumps(payload).encode()
    assert await adapter.parse_webhook(raw, {}) is None


@pytest.mark.asyncio
async def test_parse_webhook_invalid_json_returns_none(adapter):
    assert await adapter.parse_webhook(b"not-json!!", {}) is None


@pytest.mark.asyncio
async def test_parse_webhook_malformed_envelope_returns_none(adapter):
    """Missing the standard entry/changes structure → None."""
    raw = json.dumps({"object": "whatsapp_business_account"}).encode()
    assert await adapter.parse_webhook(raw, {}) is None


@pytest.mark.asyncio
async def test_parse_webhook_unsupported_type_returns_none(adapter):
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "metadata": {"display_phone_number": _BOT_NUMBER},
                    "messages": [{
                        "id": "wamid.loc001",
                        "from": "2348012345678",
                        "timestamp": "1700000000",
                        "type": "location",
                        "location": {"latitude": 6.45, "longitude": 3.39},
                    }],
                }
            }]
        }]
    }
    raw = json.dumps(payload).encode()
    assert await adapter.parse_webhook(raw, {}) is None


# ── send_text ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_text_returns_message_id(adapter):
    mock_client = _httpx_mock(body={"status": 1, "wa_message_id": "wamid.sent001"})
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        result = await adapter.send_text("+2348012345678", "Hello!")
    assert result == "wamid.sent001"


@pytest.mark.asyncio
async def test_send_text_strips_leading_plus(adapter):
    """Phone number sent to WhatChimp API must NOT have leading +."""
    mock_client = _httpx_mock(body={"status": 1, "wa_message_id": "wamid.001"})
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        await adapter.send_text("+2348012345678", "Hi")
    _, kwargs = mock_client.post.call_args
    sent_data = kwargs.get("data") or mock_client.post.call_args.args[0] if mock_client.post.call_args.args else {}
    # Extract from positional or keyword call_args
    call_data = mock_client.post.call_args.kwargs.get("data", {})
    assert not call_data.get("phone_number", "").startswith("+")


@pytest.mark.asyncio
async def test_send_text_includes_required_fields(adapter):
    mock_client = _httpx_mock(body={"status": 1, "wa_message_id": "wamid.001"})
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        await adapter.send_text("+2348012345678", "Test message")
    call_data = mock_client.post.call_args.kwargs.get("data", {})
    assert "apiToken" in call_data
    assert "phone_number_id" in call_data
    assert "message" in call_data
    assert call_data["message"] == "Test message"


@pytest.mark.asyncio
async def test_send_text_raises_on_http_error(adapter):
    mock_client = _httpx_mock(status=400, body={"error": "bad request"})
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        with pytest.raises(ExternalServiceError):
            await adapter.send_text("+2348012345678", "Hi")


@pytest.mark.asyncio
async def test_send_text_raises_on_status_zero(adapter):
    """WhatChimp returns HTTP 200 but status=0 when the message fails."""
    mock_client = _httpx_mock(body={"status": 0, "message": "Invalid phone number"})
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        with pytest.raises(ExternalServiceError, match="Invalid phone number"):
            await adapter.send_text("+2348099999999", "Hi")


@pytest.mark.asyncio
async def test_send_text_raises_on_non_json_response(adapter):
    mock_client = _httpx_mock(status=200)  # no body → json() raises
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        with pytest.raises(ExternalServiceError):
            await adapter.send_text("+2348012345678", "Hi")


# ── download_media ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_download_media_uses_url_directly(adapter):
    """If argument starts with 'http', do a direct GET to that URL."""
    mock_client = _httpx_mock(content=b"audio-bytes")
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        data = await adapter.download_media("https://cdn.whatsapp.net/media-abc123")
    assert data == b"audio-bytes"
    url_called = mock_client.get.call_args.args[0]
    assert url_called == "https://cdn.whatsapp.net/media-abc123"


@pytest.mark.asyncio
async def test_download_media_uses_id_path_for_non_url(adapter):
    """Non-URL media_id → falls back to /whatsapp/media/<id> path."""
    mock_client = _httpx_mock(content=b"audio-data")
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        await adapter.download_media("media-abc123")
    url_called = mock_client.get.call_args.args[0]
    assert "media-abc123" in url_called
    assert not url_called.startswith("https://cdn")


@pytest.mark.asyncio
async def test_download_media_raises_on_4xx(adapter):
    mock_client = _httpx_mock(status=404, content=b"not found")
    with patch("healthbridgeai.infrastructure.messaging.whatchamp.httpx.AsyncClient",
               return_value=mock_client):
        with pytest.raises(ExternalServiceError):
            await adapter.download_media("https://cdn.whatsapp.net/missing")
