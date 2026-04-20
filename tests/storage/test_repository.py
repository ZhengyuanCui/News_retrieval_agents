"""
Repository integration tests against a real in-memory SQLite database.

Tests cover: get_recent(), prune_old_items(), update_analysis_many() field
protection, get_unanalyzed(), upsert idempotency, get_stats(), and
mark_duplicate().
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from news_agent.storage.database import get_session, init_db
from news_agent.storage.repository import NewsRepository
from tests.conftest import hours_ago, make_item


# ── get_recent ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_recent_returns_items_within_window():
    await init_db()
    fresh = make_item(
        url="https://example.com/fresh-repo",
        title="Fresh AI news for repository test",
        published_at=hours_ago(2),
        topic="ai",
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(fresh)
        results = await repo.get_recent(hours=24)
    ids = [r.id for r in results]
    assert fresh.id in ids


@pytest.mark.asyncio
async def test_get_recent_excludes_old_items():
    await init_db()
    old = make_item(
        url="https://example.com/old-repo",
        title="Old AI article for repository test",
        published_at=hours_ago(200),
        topic="ai",
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(old)
        results = await repo.get_recent(hours=24)
    ids = [r.id for r in results]
    assert old.id not in ids, "200h-old article should be outside 24h window"


@pytest.mark.asyncio
async def test_get_recent_filters_by_topic():
    await init_db()
    ai_item = make_item(
        url="https://example.com/ai-topic",
        title="AI topic filter test",
        topic="ai",
        published_at=hours_ago(1),
    )
    crypto_item = make_item(
        url="https://example.com/crypto-topic",
        title="Crypto topic filter test",
        topic="crypto",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(ai_item)
        await repo.upsert(crypto_item)
        results = await repo.get_recent(hours=24, topic="ai")
    ids = [r.id for r in results]
    assert ai_item.id in ids
    assert crypto_item.id not in ids


@pytest.mark.asyncio
async def test_get_recent_excludes_duplicates_by_default():
    await init_db()
    original = make_item(
        url="https://example.com/original-dedup",
        title="Original article for dedup test",
        published_at=hours_ago(1),
    )
    dup = make_item(
        url="https://example.com/duplicate-dedup",
        title="Duplicate article for dedup test",
        published_at=hours_ago(1),
        is_duplicate=True,
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(original)
        await repo.upsert(dup)
        results = await repo.get_recent(hours=24)
    ids = [r.id for r in results]
    assert original.id in ids
    assert dup.id not in ids, "duplicates should be excluded by default"


@pytest.mark.asyncio
async def test_get_recent_include_duplicates_flag():
    await init_db()
    dup = make_item(
        url="https://example.com/dup-flag-test",
        title="Dup flag test article",
        published_at=hours_ago(1),
        is_duplicate=True,
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(dup)
        results = await repo.get_recent(hours=24, include_duplicates=True)
    ids = [r.id for r in results]
    assert dup.id in ids


# ── prune_old_items ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prune_old_items_removes_old():
    await init_db()
    old = make_item(
        url="https://example.com/prunable",
        title="Prunable old article for prune test",
        published_at=hours_ago(200),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(old)
        from sqlalchemy import update
        from news_agent.models import NewsItemORM
        old_fetched = datetime.utcnow() - timedelta(days=10)
        await session.execute(
            update(NewsItemORM)
            .where(NewsItemORM.id == old.id)
            .values(fetched_at=old_fetched)
        )
        await session.flush()
        deleted = await repo.prune_old_items(retention_days=7)
    assert deleted >= 1


@pytest.mark.asyncio
async def test_prune_old_items_keeps_recent():
    await init_db()
    recent = make_item(
        url="https://example.com/keep-recent",
        title="Keep recent article for prune test",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(recent)
        deleted = await repo.prune_old_items(retention_days=7)
    assert deleted == 0


# ── update_analysis_many ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_analysis_many_fills_analysis_fields():
    await init_db()
    item = make_item(
        url="https://example.com/analysis-test",
        title="Analysis fields test article",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        item.summary = "This is the LLM summary."
        item.relevance_score = 8.5
        item.sentiment = "positive"
        item.tags = ["ai", "technology"]
        item.key_entities = ["OpenAI"]
        await repo.update_analysis_many([item])
        retrieved = await repo.get_by_id(item.id)
    assert retrieved.summary == "This is the LLM summary."
    assert retrieved.relevance_score == 8.5
    assert retrieved.sentiment == "positive"


@pytest.mark.asyncio
async def test_update_analysis_many_does_not_overwrite_analysis_on_re_upsert():
    """Re-upserting (plain fetch) must NOT clobber already-set analysis fields."""
    await init_db()
    item = make_item(
        url="https://example.com/protect-analysis",
        title="Protect analysis fields test",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        item.summary = "Protected summary."
        item.relevance_score = 9.0
        await repo.update_analysis_many([item])
        # Now re-upsert (simulates next fetch cycle)
        item.summary = None
        item.relevance_score = None
        await repo.upsert(item)
        retrieved = await repo.get_by_id(item.id)
    assert retrieved.summary == "Protected summary.", "summary must survive a plain re-upsert"
    assert retrieved.relevance_score == 9.0


@pytest.mark.asyncio
async def test_update_analysis_many_skips_items_without_summary():
    """Items with summary=None should not overwrite existing analysis."""
    await init_db()
    item = make_item(
        url="https://example.com/skip-no-summary",
        title="Skip no-summary article",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        item.summary = "Initial summary."
        item.relevance_score = 7.0
        await repo.update_analysis_many([item])
        item.summary = None
        item.relevance_score = 3.0
        await repo.update_analysis_many([item])  # should skip
        retrieved = await repo.get_by_id(item.id)
    assert retrieved.summary == "Initial summary."


# ── get_unanalyzed ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_unanalyzed_returns_items_without_summary():
    await init_db()
    no_summary = make_item(
        url="https://example.com/unanalyzed",
        title="Unanalyzed article for test",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(no_summary)
        unanalyzed = await repo.get_unanalyzed(limit=100)
    ids = [i.id for i in unanalyzed]
    assert no_summary.id in ids


@pytest.mark.asyncio
async def test_get_unanalyzed_excludes_already_analyzed():
    await init_db()
    analyzed = make_item(
        url="https://example.com/already-analyzed",
        title="Already analyzed article for test",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(analyzed)
        analyzed.summary = "Done."
        analyzed.relevance_score = 7.0
        await repo.update_analysis_many([analyzed])
        unanalyzed = await repo.get_unanalyzed(limit=100)
    ids = [i.id for i in unanalyzed]
    assert analyzed.id not in ids


@pytest.mark.asyncio
async def test_get_unanalyzed_excludes_duplicates():
    await init_db()
    dup = make_item(
        url="https://example.com/dup-unanalyzed",
        title="Dup unanalyzed article for test",
        published_at=hours_ago(1),
        is_duplicate=True,
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(dup)
        unanalyzed = await repo.get_unanalyzed(limit=100)
    ids = [i.id for i in unanalyzed]
    assert dup.id not in ids


# ── upsert idempotency ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_idempotent_same_item():
    await init_db()
    item = make_item(
        url="https://example.com/idempotent",
        title="Idempotent upsert test article",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        await repo.upsert(item)  # second upsert — same data
        results = await repo.get_recent(hours=24)
    matching = [r for r in results if r.id == item.id]
    assert len(matching) == 1, "duplicate upsert should not create two rows"


# ── get_stats ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stats_returns_total_count():
    await init_db()
    async with get_session() as session:
        repo = NewsRepository(session)
        stats = await repo.get_stats()
    assert "total_items" in stats
    assert isinstance(stats["total_items"], int)
    assert "by_topic" in stats


# ── mark_duplicate ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mark_duplicate_sets_flag():
    await init_db()
    original = make_item(url="https://example.com/orig-dup-flag", published_at=hours_ago(1))
    dup = make_item(url="https://example.com/the-dup-flag", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(original)
        await repo.upsert(dup)
        await repo.mark_duplicate(dup.id, original.id)
        retrieved = await repo.get_by_id(dup.id)
    assert retrieved.is_duplicate is True
    assert retrieved.duplicate_of == original.id
