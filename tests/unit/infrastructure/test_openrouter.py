"""
Unit tests for OpenRouterClient.

BGE-M3 loading is bypassed by mocking _embed_sync / _embed_sparse_sync directly
on the instance. OpenAI / instructor API calls are mocked at the attribute level
after construction to avoid real HTTP connections.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from healthbridgeai.core.exceptions import LLMError
from healthbridgeai.infrastructure.llm.openrouter import OpenRouterClient


# ── Helpers ────────────────────────────────────────────────────────────────────

class _Answer(BaseModel):
    """Minimal Pydantic model used as structured-output target in tests."""
    answer: str


def _mock_choice(content: str = "Here is the answer.") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    return choice


@pytest.fixture
def client() -> OpenRouterClient:
    """
    OpenRouterClient with AsyncOpenAI and instructor patched during __init__
    so no real HTTP clients are created, then replaced with controllable mocks.
    """
    with patch("healthbridgeai.infrastructure.llm.openrouter.AsyncOpenAI"), \
         patch("healthbridgeai.infrastructure.llm.openrouter.instructor"):
        c = OpenRouterClient()

    # Replace with controllable async mocks
    c._raw = AsyncMock()
    c._instructor = AsyncMock()

    # Bypass BGE-M3 loading in embed / embed_sparse tests
    c._embed_sync = lambda texts: [[0.1] * 1024 for _ in texts]
    c._embed_sparse_sync = lambda texts: [
        {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]} for _ in texts
    ]
    return c


# ── structured() ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_structured_returns_pydantic_model(client):
    client._instructor.chat.completions.create = AsyncMock(
        return_value=_Answer(answer="TB is spread by air.")
    )
    result = await client.structured(
        system="You are a health assistant.",
        user="How is TB spread?",
        response_model=_Answer,
    )
    assert isinstance(result, _Answer)
    assert "TB" in result.answer


@pytest.mark.asyncio
async def test_structured_raises_llm_error_on_failure(client):
    client._instructor.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("API unavailable")
    )
    with pytest.raises(LLMError):
        await client.structured("sys", "user", _Answer)


@pytest.mark.asyncio
async def test_structured_uses_default_model(client, monkeypatch):
    monkeypatch.setattr(
        "healthbridgeai.infrastructure.llm.openrouter.settings",
        MagicMock(LLM_PRIMARY_MODEL="default-model", LLM_TIMEOUT_SECONDS=30),
    )
    client._instructor.chat.completions.create = AsyncMock(
        return_value=_Answer(answer="ok")
    )
    await client.structured("sys", "user", _Answer)
    call_kwargs = client._instructor.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "default-model"


@pytest.mark.asyncio
async def test_structured_uses_explicit_model(client):
    client._instructor.chat.completions.create = AsyncMock(
        return_value=_Answer(answer="ok")
    )
    await client.structured("sys", "user", _Answer, model="anthropic/claude-haiku")
    call_kwargs = client._instructor.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-haiku"


@pytest.mark.asyncio
async def test_structured_includes_system_and_user_messages(client):
    client._instructor.chat.completions.create = AsyncMock(
        return_value=_Answer(answer="ok")
    )
    await client.structured("SYSTEM PROMPT", "USER PROMPT", _Answer)
    messages = client._instructor.chat.completions.create.call_args.kwargs["messages"]
    roles = {m["role"]: m["content"] for m in messages}
    assert roles.get("system") == "SYSTEM PROMPT"
    assert roles.get("user") == "USER PROMPT"


# ── complete() ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_returns_text(client):
    mock_resp = MagicMock()
    mock_resp.choices = [_mock_choice("This is a complete answer.")]
    client._raw.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await client.complete("system", "user")
    assert result == "This is a complete answer."


@pytest.mark.asyncio
async def test_complete_returns_empty_string_when_content_is_none(client):
    mock_resp = MagicMock()
    mock_resp.choices = [_mock_choice(None)]   # type: ignore[arg-type]
    client._raw.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await client.complete("system", "user")
    assert result == ""


@pytest.mark.asyncio
async def test_complete_retries_on_transient_failure(client):
    """One transient failure should be retried; second attempt succeeds."""
    mock_resp = MagicMock()
    mock_resp.choices = [_mock_choice("recovered")]

    call_count = 0
    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient network error")
        return mock_resp

    client._raw.chat.completions.create = _side_effect

    with patch("asyncio.sleep"):   # suppress tenacity's inter-retry wait
        result = await client.complete("system", "user")

    assert result == "recovered"
    assert call_count == 2


@pytest.mark.asyncio
async def test_complete_uses_default_model(client, monkeypatch):
    monkeypatch.setattr(
        "healthbridgeai.infrastructure.llm.openrouter.settings",
        MagicMock(LLM_PRIMARY_MODEL="default-model", LLM_TIMEOUT_SECONDS=30),
    )
    mock_resp = MagicMock()
    mock_resp.choices = [_mock_choice("ok")]
    client._raw.chat.completions.create = AsyncMock(return_value=mock_resp)

    await client.complete("sys", "user")
    call_kwargs = client._raw.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "default-model"


# ── embed() ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_returns_2d_list(client):
    result = await client.embed(["What is TB?"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)


@pytest.mark.asyncio
async def test_embed_single_text_shape(client):
    result = await client.embed(["What is TB?"])
    assert len(result) == 1
    assert len(result[0]) == 1024


@pytest.mark.asyncio
async def test_embed_multiple_texts(client):
    texts = ["TB question", "HIV question", "Malaria question"]
    result = await client.embed(texts)
    assert len(result) == 3
    assert all(len(vec) == 1024 for vec in result)


# ── embed_sparse() ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_sparse_returns_list_of_dicts(client):
    result = await client.embed_sparse(["TB question"])
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "indices" in result[0]
    assert "values" in result[0]


@pytest.mark.asyncio
async def test_embed_sparse_indices_are_ints(client):
    result = await client.embed_sparse(["TB question"])
    assert all(isinstance(i, int) for i in result[0]["indices"])


@pytest.mark.asyncio
async def test_embed_sparse_values_are_floats(client):
    result = await client.embed_sparse(["TB question"])
    assert all(isinstance(v, float) for v in result[0]["values"])


@pytest.mark.asyncio
async def test_embed_sparse_multiple_texts(client):
    texts = ["TB", "HIV", "Malaria"]
    result = await client.embed_sparse(texts)
    assert len(result) == 3
