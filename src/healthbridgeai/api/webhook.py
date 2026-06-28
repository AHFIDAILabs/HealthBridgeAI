"""Webhook router — validates WhatChimp events and publishes to Pub/Sub."""
from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, Response

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import WebhookAuthError

log = structlog.get_logger(__name__)

router = APIRouter(tags=["webhook"])


@router.get("/webhook", status_code=200)
async def webhook_verify(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
    hub_challenge: str = Query("", alias="hub.challenge"),
) -> Response:
    """
    Meta webhook verification challenge (GET).

    When you register a webhook URL in the Meta App Dashboard (or via WhatChimp),
    Meta sends a GET request to confirm ownership:

        GET /webhook?hub.mode=subscribe
                    &hub.verify_token=<your-token>
                    &hub.challenge=<random-string>

    We must respond with hub.challenge as plain text and status 200.
    The verify token must match WHATCHAMP_WEBHOOK_VERIFY_TOKEN in settings.
    """
    if hub_mode != "subscribe":
        raise HTTPException(status_code=400, detail="hub.mode must be 'subscribe'")

    expected = settings.WHATCHAMP_WEBHOOK_VERIFY_TOKEN
    if not expected:
        log.error("webhook.verify.token_not_configured")
        raise HTTPException(status_code=500, detail="Webhook verify token not configured")

    if hub_verify_token != expected:
        log.warning("webhook.verify.token_mismatch")
        raise HTTPException(status_code=403, detail="hub.verify_token mismatch")

    log.info("webhook.verify.ok")
    return Response(content=hub_challenge, media_type="text/plain")


@router.post("/webhook", status_code=200)
async def webhook(request: Request) -> Response:
    """
    Inbound WhatChimp webhook receiver.

    Flow:
      1. Validate HMAC-SHA256 signature — reject non-WhatChimp requests with 401
      2. Parse raw body into InboundMessage via the messaging adapter
      3. Publish serialised InboundMessage to Pub/Sub inbound topic
      4. Return 200 in < 2 seconds regardless of pipeline outcome

    Non-message events (delivery receipts, status callbacks) are acknowledged
    silently with 200 — only messages with parseable content are published.
    """
    raw_body = await request.body()
    headers = dict(request.headers)
    messaging = request.app.state.messaging

    # ── 1. Signature validation ───────────────────────────────────────────────
    try:
        await messaging.validate_webhook(raw_body, headers)
    except WebhookAuthError:
        log.warning("webhook.signature_rejected")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    except Exception as exc:
        log.error("webhook.validation_error", error=str(exc))
        raise HTTPException(status_code=401, detail="Webhook validation failed")

    # ── 2. Parse message ──────────────────────────────────────────────────────
    try:
        message = await messaging.parse_webhook(raw_body, headers)
    except Exception as exc:
        log.error("webhook.parse_error", error=str(exc))
        return Response(status_code=200)  # acknowledge; do not retry

    if message is None:
        # Status update, read receipt, or unsupported event type
        return Response(status_code=200)

    # ── 3. Publish to Pub/Sub ─────────────────────────────────────────────────
    try:
        payload = json.dumps(message.model_dump()).encode("utf-8")
        future = request.app.state.publisher.publish(
            request.app.state.topic_path,
            data=payload,
            # Pub/Sub message attributes (for filtering/routing if needed)
            message_id=message.message_id,
            message_type=message.type.value,
        )
        future.result(timeout=5)
        log.info(
            "webhook.published",
            message_id=message.message_id,
            message_type=message.type.value,
        )
    except Exception as exc:
        # Publish failure is logged but we still return 200 — the processor
        # has a dead-letter queue; retrying here would duplicate messages.
        log.error(
            "webhook.publish_failed",
            error=str(exc),
            message_id=message.message_id,
        )

    return Response(status_code=200)
