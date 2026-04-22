from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from tests.conftest import hours_ago, make_item


@pytest.fixture
async def isolated_db():
    yield


@pytest.mark.asyncio
async def test_smart_filter_flag_off_disables_routing(monkeypatch):
    import news_agent.web.app as app_module

    called = {"router": 0}

    @asynccontextmanager
    async def fake_get_session():
        yield object()

    async def fake_extract_filters(_topic):
        called["router"] += 1
        raise AssertionError("router should not be called")

    monkeypatch.setattr(app_module, "get_session", fake_get_session)
    monkeypatch.setattr(app_module, "render", lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr("news_agent.config.settings.smart_filter_enabled", False)
    monkeypatch.setattr("news_agent.pipeline.query_router.extract_filters", fake_extract_filters)
    monkeypatch.setattr("news_agent.storage.repository.NewsRepository.search", AsyncMock(return_value=[]))
    monkeypatch.setattr(app_module, "get_preference_scores", AsyncMock(return_value={}))
    monkeypatch.setattr(app_module, "apply_preference_boost", lambda items, prefs: items)

    resp = await app_module.panel_fragment(topic="openai")

    assert resp == "ok"
    assert called["router"] == 0


@pytest.mark.asyncio
async def test_smart_filter_fallback_on_empty(monkeypatch):
    import news_agent.web.app as app_module
    from news_agent.pipeline.query_router import QueryFilters

    item = make_item(url="https://example.com/filter-fallback", title="Fallback story", published_at=hours_ago(1))
    calls: list[QueryFilters | None] = []

    @asynccontextmanager
    async def fake_get_session():
        yield object()

    async def fake_extract_filters(_topic):
        return QueryFilters(sources=["github"])

    async def fake_search(self, query, **kwargs):
        calls.append(kwargs.get("filters"))
        return [] if kwargs.get("filters") is not None else [item]

    monkeypatch.setattr(app_module, "get_session", fake_get_session)
    monkeypatch.setattr(app_module, "render", lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr("news_agent.config.settings.smart_filter_enabled", True)
    monkeypatch.setattr("news_agent.pipeline.query_router.extract_filters", fake_extract_filters)
    monkeypatch.setattr("news_agent.storage.repository.NewsRepository.search", fake_search)
    monkeypatch.setattr(app_module, "get_preference_scores", AsyncMock(return_value={}))
    monkeypatch.setattr(app_module, "apply_preference_boost", lambda items, prefs: items)
    monkeypatch.setattr("news_agent.pipeline.ranker.rank_by_query", lambda topic, items: items)

    resp = await app_module.panel_fragment(topic="openai")

    assert resp == "ok"
    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[1] is None
