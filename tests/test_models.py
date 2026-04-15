"""Basic smoke tests — no API calls required."""
from datetime import datetime

from news_agent.models import NewsItem
from news_agent.pipeline.deduplicator import Deduplicator, normalize_url


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


def test_news_item_id_is_stable():
    item = make_item()
    assert item.id == item.id  # computed_field is stable
    assert len(item.id) == 16


def test_news_item_id_depends_on_source_and_url():
    a = make_item(source="reddit", url="https://example.com/a")
    b = make_item(source="twitter", url="https://example.com/a")
    assert a.id != b.id


def test_normalize_url_strips_tracking():
    url = "https://example.com/article?utm_source=twitter&utm_medium=social&id=123"
    assert "utm_source" not in normalize_url(url)
    assert "id=123" in normalize_url(url)


def test_url_dedup_exact():
    dedup = Deduplicator(strategy="url_only")
    items = [
        make_item(url="https://example.com/a", raw_score=0.3),
        make_item(url="https://example.com/a?utm_source=x", raw_score=0.8),  # same after normalization
        make_item(url="https://example.com/b", raw_score=0.5),
    ]
    result = dedup.deduplicate(items)
    non_dupes = [i for i in result if not i.is_duplicate]
    assert len(non_dupes) == 2


def test_tfidf_dedup():
    dedup = Deduplicator(strategy="tfidf", threshold=0.9)
    items = [
        make_item(title="OpenAI releases GPT-5 model", content="OpenAI announced GPT-5 today with major improvements"),
        make_item(title="OpenAI releases GPT-5 model", content="OpenAI announced GPT-5 today with major improvements", url="https://example.com/b"),
        make_item(title="Stock market hits record high", content="S&P 500 reached all time high", url="https://example.com/c"),
    ]
    result = dedup.deduplicate(items)
    non_dupes = [i for i in result if not i.is_duplicate]
    assert len(non_dupes) <= 2  # near-identical pair should collapse
