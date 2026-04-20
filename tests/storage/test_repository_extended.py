"""
Extended repository tests covering: exists, get_by_id, update_url,
clear_all, count_unanalyzed, upsert_digest / get_digest / delete_digest,
update_collector_state / get_all_collector_states, and get_recent language filter.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from news_agent.storage.database import get_session, init_db
from news_agent.storage.repository import NewsRepository
from tests.conftest import hours_ago, make_item


# ── exists / get_by_id ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exists_returns_true_for_stored_item():
    item = make_item(url="https://example.com/exists-true", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        assert await repo.exists(item.id) is True


@pytest.mark.asyncio
async def test_exists_returns_false_for_missing_item():
    async with get_session() as session:
        repo = NewsRepository(session)
        assert await repo.exists("nonexistent0000") is False


@pytest.mark.asyncio
async def test_get_by_id_returns_item():
    item = make_item(
        url="https://example.com/get-by-id",
        title="Get by ID test article",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        retrieved = await repo.get_by_id(item.id)
    assert retrieved is not None
    assert retrieved.id == item.id
    assert retrieved.title == item.title


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing():
    async with get_session() as session:
        repo = NewsRepository(session)
        result = await repo.get_by_id("doesnotexist0000")
    assert result is None


# ── update_url ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_url_persists_resolved_url():
    item = make_item(
        url="https://news.google.com/redirect/original",
        title="Google News redirect article",
        published_at=hours_ago(1),
    )
    real_url = "https://techcrunch.com/real-article"
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        await repo.update_url(item.id, real_url)

    async with get_session() as session2:
        repo2 = NewsRepository(session2)
        retrieved = await repo2.get_by_id(item.id)
    assert retrieved.url == real_url


# ── clear_all ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_all_removes_all_items():
    async with get_session() as session:
        repo = NewsRepository(session)
        for i in range(3):
            await repo.upsert(make_item(url=f"https://example.com/clear-{i}", published_at=hours_ago(1)))
        counts = await repo.clear_all()
    assert counts["items"] >= 3


@pytest.mark.asyncio
async def test_clear_all_leaves_empty_db():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(make_item(url="https://example.com/clear-check", published_at=hours_ago(1)))
        await repo.clear_all()
        stats = await repo.get_stats()
    assert stats["total_items"] == 0


# ── count_unanalyzed ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_count_unanalyzed_counts_items_without_summary():
    async with get_session() as session:
        repo = NewsRepository(session)
        for i in range(4):
            await repo.upsert(make_item(url=f"https://example.com/uncount-{i}", published_at=hours_ago(1)))
        count = await repo.count_unanalyzed()
    assert count >= 4


@pytest.mark.asyncio
async def test_count_unanalyzed_excludes_analyzed():
    item = make_item(url="https://example.com/analyzed-count", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        item.summary = "Done."
        await repo.update_analysis_many([item])
        count_after = await repo.count_unanalyzed()
    assert item.id not in [i.id for i in await _get_unanalyzed(repo)]


async def _get_unanalyzed(repo):
    return await repo.get_unanalyzed(limit=500)


# ── language filter ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_recent_filters_by_language():
    en_item = make_item(
        url="https://example.com/en-lang",
        title="English language article",
        language="en",
        published_at=hours_ago(1),
    )
    zh_item = make_item(
        url="https://example.com/zh-lang",
        title="Chinese language article",
        language="zh",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(en_item)
        await repo.upsert(zh_item)
        results = await repo.get_recent(hours=24, languages=["en"])
    ids = [r.id for r in results]
    assert en_item.id in ids
    assert zh_item.id not in ids


# ── upsert_digest / get_digest / delete_digest ────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_and_get_digest():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest("2025-01-15", "openai", "Headline\nBullet 1\nBullet 2", item_count=5)
        digest = await repo.get_digest("2025-01-15", "openai")
    assert digest is not None
    assert "Headline" in digest.content
    assert digest.item_count == 5


@pytest.mark.asyncio
async def test_get_digest_case_insensitive_topic():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest("2025-01-15", "OpenAI", "Test headline", item_count=3)
        digest = await repo.get_digest("2025-01-15", "openai")
    assert digest is not None


@pytest.mark.asyncio
async def test_get_digest_falls_back_to_most_recent():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest("2025-01-10", "stocks", "Old digest", item_count=2)
        # No digest for today — should fall back to most recent
        digest = await repo.get_digest("2025-06-01", "stocks")
    assert digest is not None
    assert "Old digest" in digest.content


@pytest.mark.asyncio
async def test_get_digest_returns_none_when_no_digest():
    async with get_session() as session:
        repo = NewsRepository(session)
        digest = await repo.get_digest("2020-01-01", "no_such_topic_xyz")
    assert digest is None


@pytest.mark.asyncio
async def test_delete_digest_removes_entry():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest("2025-03-01", "ai_delete_test", "Some digest", item_count=10)

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.delete_digest("2025-03-01", "ai_delete_test")

    async with get_session() as session:
        repo = NewsRepository(session)
        digest = await repo.get_digest("2025-03-01", "ai_delete_test")
    assert digest is None


@pytest.mark.asyncio
async def test_upsert_digest_updates_existing():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest("2025-04-01", "crypto", "First version", item_count=5)
        await repo.upsert_digest("2025-04-01", "crypto", "Updated version", item_count=12)
        digest = await repo.get_digest("2025-04-01", "crypto")
    assert "Updated version" in digest.content
    assert digest.item_count == 12


# ── collector state ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_collector_state_creates_new_entry():
    from datetime import datetime
    run_time = datetime.utcnow()
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.update_collector_state(
            "rss",
            last_run=run_time,
            items_fetched=42,
        )
        states = await repo.get_all_collector_states()
    names = [s.source for s in states]
    assert "rss" in names
    rss = next(s for s in states if s.source == "rss")
    assert rss.items_fetched == 42


@pytest.mark.asyncio
async def test_update_collector_state_accumulates_items_fetched():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.update_collector_state("twitter", items_fetched=10)
        await repo.update_collector_state("twitter", items_fetched=5)
        states = await repo.get_all_collector_states()
    tw = next((s for s in states if s.source == "twitter"), None)
    assert tw is not None
    assert tw.items_fetched == 15


@pytest.mark.asyncio
async def test_update_collector_state_records_error():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.update_collector_state("github", last_error="Rate limit exceeded")
        states = await repo.get_all_collector_states()
    gh = next((s for s in states if s.source == "github"), None)
    assert gh is not None
    assert "Rate limit" in gh.last_error


@pytest.mark.asyncio
async def test_update_collector_state_set_disabled():
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.update_collector_state("youtube", is_enabled=False)
        states = await repo.get_all_collector_states()
    yt = next((s for s in states if s.source == "youtube"), None)
    assert yt is not None
    assert yt.is_enabled is False


@pytest.mark.asyncio
async def test_get_all_collector_states_returns_all():
    async with get_session() as session:
        repo = NewsRepository(session)
        for src in ("src_a", "src_b", "src_c"):
            await repo.update_collector_state(src, items_fetched=1)
        states = await repo.get_all_collector_states()
    sources = {s.source for s in states}
    assert {"src_a", "src_b", "src_c"}.issubset(sources)
