"""
Unit tests for /health and /ready endpoints.

Uses FastAPI TestClient with manually-set app.state to avoid a real lifespan
(no Pub/Sub, Pinecone, or WhatChimp credentials needed).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from healthbridgeai.api.health import router


# ── Helpers ───────────────────────────────────────────────────────────────────

_GOOD_TOPIC = "projects/test-project/topics/inbound-messages"
_GOOD_KEY = "whatchamp-api-key"
_GOOD_STATS = {"total_vector_count": 1500}


def _make_app(
    pinecone_stats: dict | None = None,
    pinecone_error: Exception | None = None,
    topic_path: str = _GOOD_TOPIC,
    messaging_key: str | None = _GOOD_KEY,
    omit_messaging: bool = False,
    omit_topic: bool = False,
) -> FastAPI:
    """Build a minimal FastAPI app with health router and mocked app.state."""
    app = FastAPI()
    app.include_router(router)

    mock_pinecone = AsyncMock()
    if pinecone_error:
        mock_pinecone.describe_index.side_effect = pinecone_error
    else:
        mock_pinecone.describe_index.return_value = pinecone_stats or _GOOD_STATS
    app.state.pinecone = mock_pinecone

    if not omit_topic:
        app.state.topic_path = topic_path

    if not omit_messaging:
        mock_messaging = MagicMock()
        mock_messaging._api_key = messaging_key or ""
        app.state.messaging = mock_messaging

    return app


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "version" in body


# ── /ready — happy path ───────────────────────────────────────────────────────

def test_ready_all_healthy_returns_200():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_body_all_healthy():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["checks"]["pinecone"].startswith("ok")
    assert body["checks"]["pubsub"].startswith("ok")
    assert body["checks"]["whatchamp"] == "ok"
    assert "version" in body


def test_ready_pinecone_check_includes_vector_count():
    app = _make_app(pinecone_stats={"total_vector_count": 42000})
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert "42000" in body["checks"]["pinecone"]


def test_ready_pubsub_check_includes_topic_path():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert _GOOD_TOPIC in body["checks"]["pubsub"]


# ── /ready — Pinecone failure ─────────────────────────────────────────────────

def test_ready_pinecone_error_returns_503():
    app = _make_app(pinecone_error=RuntimeError("index not found"))
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_pinecone_error_body():
    app = _make_app(pinecone_error=ConnectionError("timeout"))
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert body["status"] == "not_ready"
    assert "error" in body["checks"]["pinecone"]


def test_ready_pinecone_error_other_checks_still_reported():
    """A Pinecone failure must not short-circuit the other checks."""
    app = _make_app(pinecone_error=RuntimeError("boom"))
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert "pubsub" in body["checks"]
    assert "whatchamp" in body["checks"]


# ── /ready — Pub/Sub failure ──────────────────────────────────────────────────

def test_ready_missing_topic_path_returns_503():
    app = _make_app(topic_path="")
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_omitted_topic_path_returns_503():
    app = _make_app(omit_topic=True)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_missing_topic_body():
    app = _make_app(topic_path="")
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert body["status"] == "not_ready"
    assert "error" in body["checks"]["pubsub"]


# ── /ready — WhatChimp failure ────────────────────────────────────────────────

def test_ready_missing_messaging_returns_503():
    app = _make_app(omit_messaging=True)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_empty_api_key_returns_503():
    app = _make_app(messaging_key="")
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_empty_api_key_body():
    app = _make_app(messaging_key="")
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert body["status"] == "not_ready"
    assert "error" in body["checks"]["whatchamp"]


# ── /ready — multiple failures ────────────────────────────────────────────────

def test_ready_all_failing_returns_503():
    app = _make_app(
        pinecone_error=RuntimeError("down"),
        topic_path="",
        messaging_key="",
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_all_failing_body_reports_all_checks():
    app = _make_app(
        pinecone_error=RuntimeError("down"),
        topic_path="",
        messaging_key="",
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/ready").json()
    assert body["status"] == "not_ready"
    assert "error" in body["checks"]["pinecone"]
    assert "error" in body["checks"]["pubsub"]
    assert "error" in body["checks"]["whatchamp"]
