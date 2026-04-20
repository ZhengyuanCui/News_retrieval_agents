"""Tests for BaseCollector: rate limiting, score normalization, topic filtering."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from news_agent.collectors.base import BaseCollector
from news_agent.models import NewsItem
from tests.conftest import make_item


class _FakeCollector(BaseCollector):
    source_name = "fake"
    rate_limit_delay = 0.05  # fast for tests

    def __init__(self, topics=None, items=None):
        super().__init__(topics)
        self._items = items or []

    async def fetch(self) -> list[NewsItem]:
        return list(self._items)


# ── normalize_score ───────────────────────────────────────────────────────────

class TestNormalizeScore:
    def test_midpoint(self):
        assert BaseCollector.normalize_score(50, 0, 100) == 0.5

    def test_min_value_is_zero(self):
        assert BaseCollector.normalize_score(0, 0, 100) == 0.0

    def test_max_value_is_one(self):
        assert BaseCollector.normalize_score(100, 0, 100) == 1.0

    def test_clamps_below_min(self):
        assert BaseCollector.normalize_score(-10, 0, 100) == 0.0

    def test_clamps_above_max(self):
        assert BaseCollector.normalize_score(150, 0, 100) == 1.0

    def test_degenerate_range_returns_zero(self):
        # max_val == min_val → undefined; returns 0.0 to avoid division by zero
        assert BaseCollector.normalize_score(5, 5, 5) == 0.0

    def test_max_less_than_min_returns_zero(self):
        assert BaseCollector.normalize_score(5, 10, 5) == 0.0


# ── rate limiting ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_enforces_delay():
    c = _FakeCollector()
    c.rate_limit_delay = 0.1
    t0 = datetime.utcnow()
    await c._rate_limit()  # first call — no sleep
    await c._rate_limit()  # second call — should sleep ~0.1s
    elapsed = (datetime.utcnow() - t0).total_seconds()
    assert elapsed >= 0.08, "rate_limit should have slept at least ~0.1s"


@pytest.mark.asyncio
async def test_rate_limit_no_sleep_if_enough_time_passed():
    c = _FakeCollector()
    c.rate_limit_delay = 0.05
    await c._rate_limit()
    await asyncio.sleep(0.1)  # wait longer than rate_limit_delay
    t0 = datetime.utcnow()
    await c._rate_limit()
    elapsed = (datetime.utcnow() - t0).total_seconds()
    assert elapsed < 0.05, "no sleep needed — enough time already passed"


# ── topic filtering ───────────────────────────────────────────────────────────

class TestTopicInit:
    def test_none_uses_default_topics(self):
        class _C(BaseCollector):
            source_name = "c"
            default_topics = ["ai", "markets"]
            async def fetch(self): return []
        c = _C(topics=None)
        assert c.topics == ["ai", "markets"]

    def test_empty_list_overrides_defaults(self):
        class _C(BaseCollector):
            source_name = "c"
            default_topics = ["ai", "markets"]
            async def fetch(self): return []
        c = _C(topics=[])
        assert c.topics == []

    def test_explicit_topics_override_defaults(self):
        class _C(BaseCollector):
            source_name = "c"
            default_topics = ["ai"]
            async def fetch(self): return []
        c = _C(topics=["crypto"])
        assert c.topics == ["crypto"]


# ── tag_language ──────────────────────────────────────────────────────────────

def test_tag_language_sets_language_field():
    item = make_item(title="Artificial intelligence news", content="OpenAI released a new model.")
    result = BaseCollector.tag_language(item)
    assert result.language == "en"
    assert result is item  # mutates in place and returns same object


def test_tag_languages_processes_all_items():
    items = [
        make_item(url="https://example.com/1", title="AI news"),
        make_item(url="https://example.com/2", title="More AI news"),
    ]
    result = BaseCollector.tag_languages(items)
    assert len(result) == 2
    assert all(i.language is not None for i in result)


# ── is_enabled default ────────────────────────────────────────────────────────

def test_is_enabled_default_true():
    c = _FakeCollector()
    assert c.is_enabled() is True


# ── fetch_keyword default ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_keyword_default_returns_empty():
    c = _FakeCollector()
    result = await c.fetch_keyword("bitcoin")
    assert result == []
