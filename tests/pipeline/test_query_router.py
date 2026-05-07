from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from news_agent.pipeline.query_router import QueryFilters, extract_filters


@pytest.fixture
async def isolated_db():
    yield


def _response(payload: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])


@pytest.mark.asyncio
async def test_extract_filters_parses_source_list(monkeypatch):
    async def fake_completion(**_):
        return _response(
            '{"sources":["openai","anthropic"],"topics":["evals"],"entities":["OpenAI"],"hours":48,"is_ticker":false}'
        )

    monkeypatch.setattr(
        "news_agent.pipeline.query_router.litellm.acompletion",
        fake_completion,
    )

    filters = await extract_filters("News from OpenAI and Anthropic last 48 hours about evals")

    assert filters.sources == ["openai", "anthropic"]
    assert filters.topics == ["evals"]
    assert filters.hours == 48


@pytest.mark.asyncio
async def test_extract_filters_detects_ticker(monkeypatch):
    async def fake_completion(**_):
        return _response('{"sources":[],"topics":[],"entities":["NVDA"],"hours":null,"is_ticker":false}')

    monkeypatch.setattr(
        "news_agent.pipeline.query_router.litellm.acompletion",
        fake_completion,
    )

    filters = await extract_filters("What happened to $NVDA this week?")

    assert filters.is_ticker is True


@pytest.mark.asyncio
async def test_extract_filters_times_out_returns_empty(monkeypatch):
    async def hang(**_):
        await asyncio.sleep(1)

    monkeypatch.setattr("news_agent.pipeline.query_router.litellm.acompletion", hang)
    monkeypatch.setattr("news_agent.pipeline.query_router.settings.smart_filter_timeout_seconds", 0.01)

    filters = await extract_filters("Latest OpenAI news")

    assert filters == QueryFilters()


@pytest.mark.asyncio
async def test_extract_filters_handles_malformed_json(monkeypatch):
    async def fake_completion(**_):
        return _response("not json")

    monkeypatch.setattr(
        "news_agent.pipeline.query_router.litellm.acompletion",
        fake_completion,
    )

    filters = await extract_filters("Latest OpenAI news")

    assert filters == QueryFilters()
