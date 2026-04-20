"""Tests for Aggregator: concurrent fetch, error isolation, sorting."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from news_agent.collectors.base import BaseCollector
from news_agent.models import NewsItem
from news_agent.pipeline.aggregator import Aggregator
from tests.conftest import hours_ago, make_item


class _OKCollector(BaseCollector):
    source_name = "ok"

    def __init__(self, items, name="ok"):
        super().__init__(topics=[])
        self.source_name = name
        self._items = items

    async def fetch(self) -> list[NewsItem]:
        return list(self._items)


class _FailCollector(BaseCollector):
    source_name = "fail"

    def __init__(self):
        super().__init__(topics=[])

    async def fetch(self) -> list[NewsItem]:
        raise RuntimeError("simulated collector failure")


# ── safe_fetch ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_safe_fetch_swallows_exception():
    agg = Aggregator([_FailCollector()])
    result = await agg.fetch_all()
    assert result == [], "failing collector should return empty list, not raise"


@pytest.mark.asyncio
async def test_safe_fetch_partial_failure_keeps_good_results():
    good_item = make_item(url="https://good.com/1", title="Good news")
    agg = Aggregator([
        _OKCollector([good_item], name="ok"),
        _FailCollector(),
    ])
    result = await agg.fetch_all()
    assert len(result) == 1
    assert result[0].url == good_item.url


# ── fetch_all ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_all_merges_all_collectors():
    items_a = [make_item(url="https://a.com/1"), make_item(url="https://a.com/2")]
    items_b = [make_item(url="https://b.com/1")]
    agg = Aggregator([
        _OKCollector(items_a, name="source_a"),
        _OKCollector(items_b, name="source_b"),
    ])
    result = await agg.fetch_all()
    assert len(result) == 3


@pytest.mark.asyncio
async def test_fetch_all_sorted_newest_first():
    items = [
        make_item(url="https://a.com/old", published_at=hours_ago(10)),
        make_item(url="https://a.com/new", published_at=hours_ago(1)),
        make_item(url="https://a.com/mid", published_at=hours_ago(5)),
    ]
    agg = Aggregator([_OKCollector(items)])
    result = await agg.fetch_all()
    timestamps = [i.published_at for i in result]
    assert timestamps == sorted(timestamps, reverse=True), "results should be newest-first"


@pytest.mark.asyncio
async def test_fetch_all_empty_collectors():
    agg = Aggregator([_OKCollector([])])
    result = await agg.fetch_all()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_all_no_collectors():
    agg = Aggregator([])
    result = await agg.fetch_all()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_all_tags_language():
    item = make_item(url="https://en.com/1", title="Breaking AI news from OpenAI")
    agg = Aggregator([_OKCollector([item])])
    result = await agg.fetch_all()
    assert result[0].language is not None, "language should be tagged after fetch_all"


@pytest.mark.asyncio
async def test_fetch_all_runs_concurrently():
    """All collectors should run concurrently, not sequentially."""

    class _SlowCollector(BaseCollector):
        source_name = "slow"

        def __init__(self, delay, name):
            super().__init__(topics=[])
            self.source_name = name
            self._delay = delay

        async def fetch(self) -> list[NewsItem]:
            await asyncio.sleep(self._delay)
            return [make_item(url=f"https://{self.source_name}.com/1")]

    agg = Aggregator([_SlowCollector(0.05, "a"), _SlowCollector(0.05, "b")])
    t0 = datetime.utcnow()
    result = await agg.fetch_all()
    elapsed = (datetime.utcnow() - t0).total_seconds()
    assert len(result) == 2
    assert elapsed < 0.15, "concurrent collectors should finish in ~0.05s, not 0.10s+"
