"""Shared fixtures for the test suite."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from news_agent.models import NewsItem


def make_item(**kwargs) -> NewsItem:
    """Factory for NewsItem with sensible defaults."""
    defaults = dict(
        source="rss",
        topic="ai",
        title="Default test title",
        url="https://example.com/article",
        content="Some content about artificial intelligence and machine learning.",
        published_at=datetime.utcnow(),
        raw_score=0.5,
        relevance_score=5.0,
    )
    defaults.update(kwargs)
    return NewsItem(**defaults)


def hours_ago(n: float) -> datetime:
    return datetime.utcnow() - timedelta(hours=n)
