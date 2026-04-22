from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_agent.models import NewsItem
from news_agent.pipeline.query_router import QueryFilters
from news_agent.storage.repository import NewsRepository


@pytest.fixture
async def isolated_db():
    yield


class _FakeRow:
    def __init__(self, item: NewsItem):
        self.id = item.id
        self.title = item.title
        self.published_at = item.published_at
        self._item = item

    def to_pydantic(self) -> NewsItem:
        return self._item


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: self._rows)


@pytest.mark.asyncio
async def test_search_with_filters_restricts_sources(monkeypatch):
    item = NewsItem(
        source="openai",
        topic="ai",
        title="OpenAI ships a new evals update",
        url="https://example.com/filter-openai",
        content="OpenAI released a new evals package.",
        published_at=datetime.utcnow(),
        raw_score=0.5,
    )
    executed = {}

    async def fake_execute(statement):
        executed["sql"] = str(statement)
        return _FakeResult([_FakeRow(item)])

    repo = NewsRepository(session=SimpleNamespace(execute=fake_execute))
    monkeypatch.setattr(repo, "bm25_search", AsyncMock(return_value=[item.id]))
    monkeypatch.setattr(
        "news_agent.pipeline.vector_search.semantic_search",
        AsyncMock(return_value=[item.id]),
    )

    results = await repo.search(
        "OpenAI evals",
        hours=24,
        limit=10,
        filters=QueryFilters(sources=["openai"]),
    )

    assert [row.source for row in results] == ["openai"]
    assert "news_items.source IN" in executed["sql"]
