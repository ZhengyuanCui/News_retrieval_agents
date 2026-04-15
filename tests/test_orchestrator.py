from datetime import datetime

from news_agent.models import NewsItem
from news_agent.orchestrator import _is_digest_topic


def make_item(**kwargs) -> NewsItem:
    defaults = dict(
        source="reddit",
        topic="ai",
        title="Test title",
        url="https://example.com/article",
        content="Some content about AI",
        published_at=datetime.utcnow(),
        raw_score=0.5,
    )
    defaults.update(kwargs)
    return NewsItem(**defaults)


def test_is_digest_topic_rejects_source_backed_topic():
    items = [make_item(source="techmeme", topic="techmeme")]
    assert _is_digest_topic("techmeme", items) is False


def test_is_digest_topic_allows_keyword_topic_from_real_sources():
    items = [
        make_item(source="google-news", topic="openai"),
        make_item(source="reddit", topic="openai", url="https://example.com/article-2"),
    ]
    assert _is_digest_topic("openai", items) is True
