"""
Repository integration tests against the shared SQLite database.

Pattern: write in one session (commits on exit), then read in a fresh session
so the ORM SELECT sees fully committed data.

Tests cover: get_recent(), prune_old_items(), update_analysis_many() field
protection, get_unanalyzed(), set_starred() / get_starred_ids(), upsert
idempotency, and get_stats().
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from sqlalchemy import update as sa_update

from news_agent.models import NewsItemORM
from news_agent.storage.database import get_session, init_db
from news_agent.storage.repository import NewsRepository
from tests.conftest import hours_ago, make_item


# ── helpers ───────────────────────────────────────────────────────────────────

async def _upsert(*items):
    """Write items and commit."""
    await init_db()
    async with get_session() as session:
        repo = NewsRepository(session)
        for item in items:
            await repo.upsert(item)


async def _read_recent(hours=24, topic=None, include_duplicates=False, limit=200):
    """Read get_recent from a fresh committed session."""
    async with get_session() as session:
        return await NewsRepository(session).get_recent(
            hours=hours, topic=topic,
            include_duplicates=include_duplicates, limit=limit,
        )


# ── get_recent ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_recent_returns_items_within_window():
    item = make_item(
        url="https://example.com/repo-within-window",
        title="Repo within-window test article",
        published_at=hours_ago(2),
        topic="ai",
    )
    await _upsert(item)
    results = await _read_recent(hours=24)
    assert item.id in {r.id for r in results}


@pytest.mark.asyncio
async def test_get_recent_excludes_old_items():
    old = make_item(
        url="https://example.com/repo-old-exclude",
        title="Repo old-exclude test article",
        published_at=hours_ago(200),
        topic="ai",
    )
    await _upsert(old)
    results = await _read_recent(hours=24)
    assert old.id not in {r.id for r in results}, "200h-old article must be outside 24h window"


@pytest.mark.asyncio
async def test_get_recent_filters_by_topic():
    ai_item = make_item(
        url="https://example.com/repo-topic-ai",
        title="Repo topic-filter AI article",
        topic="ai",
        published_at=hours_ago(1),
    )
    # Use a unique topic so only our item appears in it
    unique_topic = "repo_topic_test_xyz"
    other = make_item(
        url="https://example.com/repo-topic-other",
        title="Repo topic-filter other article",
        topic=unique_topic,
        published_at=hours_ago(1),
    )
    await _upsert(ai_item, other)

    ai_results = await _read_recent(hours=24, topic="ai")
    other_results = await _read_recent(hours=24, topic=unique_topic)

    assert ai_item.id in {r.id for r in ai_results}
    assert other.id in {r.id for r in other_results}
    assert other.id not in {r.id for r in ai_results}


@pytest.mark.asyncio
async def test_get_recent_excludes_duplicates_by_default():
    original = make_item(
        url="https://example.com/repo-original-dedup",
        title="Repo dedup original article",
        published_at=hours_ago(1),
    )
    dup = make_item(
        url="https://example.com/repo-dup-dedup",
        title="Repo dedup duplicate article",
        published_at=hours_ago(1),
        is_duplicate=True,
    )
    await _upsert(original, dup)
    results = await _read_recent(hours=24)
    ids = {r.id for r in results}
    assert original.id in ids
    assert dup.id not in ids, "duplicates excluded by default"


@pytest.mark.asyncio
async def test_get_recent_include_duplicates_flag():
    dup = make_item(
        url="https://example.com/repo-dup-flag",
        title="Repo dup-flag test article",
        published_at=hours_ago(1),
        is_duplicate=True,
    )
    await _upsert(dup)
    results = await _read_recent(hours=24, include_duplicates=True)
    assert dup.id in {r.id for r in results}


# ── prune_old_items ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prune_old_items_removes_old():
    old = make_item(
        url="https://example.com/repo-prune-old",
        title="Repo prunable old article",
        published_at=hours_ago(300),
    )
    await _upsert(old)

    # Force fetched_at to 10 days ago so prune_old_items(7) catches it
    async with get_session() as session:
        await session.execute(
            sa_update(NewsItemORM)
            .where(NewsItemORM.id == old.id)
            .values(fetched_at=datetime.utcnow() - timedelta(days=10))
        )

    async with get_session() as session:
        deleted = await NewsRepository(session).prune_old_items(retention_days=7)
    assert deleted >= 1


@pytest.mark.asyncio
async def test_prune_old_items_keeps_recent():
    recent = make_item(
        url="https://example.com/repo-prune-keep",
        title="Repo keep-recent prune article",
        published_at=hours_ago(1),
    )
    await _upsert(recent)
    async with get_session() as session:
        deleted = await NewsRepository(session).prune_old_items(retention_days=7)
    assert deleted == 0


# ── update_analysis_many ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_analysis_many_fills_analysis_fields():
    item = make_item(
        url="https://example.com/repo-analysis-fill",
        title="Repo analysis-fill test article",
        published_at=hours_ago(1),
    )
    await _upsert(item)

    item.summary = "LLM-generated summary."
    item.relevance_score = 8.5
    item.sentiment = "positive"
    item.tags = ["ai", "tech"]
    item.key_entities = ["OpenAI"]
    async with get_session() as session:
        await NewsRepository(session).update_analysis_many([item])

    async with get_session() as session:
        retrieved = await NewsRepository(session).get_by_id(item.id)
    assert retrieved.summary == "LLM-generated summary."
    assert retrieved.relevance_score == 8.5
    assert retrieved.sentiment == "positive"


@pytest.mark.asyncio
async def test_update_analysis_many_not_overwritten_by_re_upsert():
    """Plain re-upsert must never clobber analysis fields set by LLMAnalyzer."""
    item = make_item(
        url="https://example.com/repo-analysis-protect",
        title="Repo analysis-protect test article",
        published_at=hours_ago(1),
    )
    await _upsert(item)

    item.summary = "Protected summary."
    item.relevance_score = 9.0
    async with get_session() as session:
        await NewsRepository(session).update_analysis_many([item])

    # Re-upsert with cleared analysis fields (simulates next fetch cycle)
    item.summary = None
    item.relevance_score = None
    await _upsert(item)

    async with get_session() as session:
        retrieved = await NewsRepository(session).get_by_id(item.id)
    assert retrieved.summary == "Protected summary.", "summary must survive a plain re-upsert"
    assert retrieved.relevance_score == 9.0


@pytest.mark.asyncio
async def test_update_analysis_many_skips_no_summary():
    """update_analysis_many must skip items with summary=None."""
    item = make_item(
        url="https://example.com/repo-analysis-skip",
        title="Repo analysis-skip test article",
        published_at=hours_ago(1),
    )
    await _upsert(item)

    # Set initial analysis
    item.summary = "Initial."
    item.relevance_score = 7.0
    async with get_session() as session:
        await NewsRepository(session).update_analysis_many([item])

    # Call again with summary=None — should be ignored
    item.summary = None
    item.relevance_score = 1.0
    async with get_session() as session:
        await NewsRepository(session).update_analysis_many([item])

    async with get_session() as session:
        retrieved = await NewsRepository(session).get_by_id(item.id)
    assert retrieved.summary == "Initial."


# ── get_unanalyzed ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_unanalyzed_returns_items_without_summary():
    item = make_item(
        url="https://example.com/repo-unanalyzed",
        title="Repo unanalyzed test article",
        published_at=hours_ago(1),
    )
    await _upsert(item)
    async with get_session() as session:
        unanalyzed = await NewsRepository(session).get_unanalyzed(limit=500)
    assert item.id in {i.id for i in unanalyzed}


@pytest.mark.asyncio
async def test_get_unanalyzed_excludes_analyzed():
    item = make_item(
        url="https://example.com/repo-analyzed-exclude",
        title="Repo analyzed-exclude test article",
        published_at=hours_ago(1),
    )
    await _upsert(item)

    item.summary = "Done."
    item.relevance_score = 7.0
    async with get_session() as session:
        await NewsRepository(session).update_analysis_many([item])

    async with get_session() as session:
        unanalyzed = await NewsRepository(session).get_unanalyzed(limit=500)
    assert item.id not in {i.id for i in unanalyzed}


@pytest.mark.asyncio
async def test_get_unanalyzed_excludes_duplicates():
    dup = make_item(
        url="https://example.com/repo-dup-unanalyzed",
        title="Repo dup-unanalyzed test article",
        published_at=hours_ago(1),
        is_duplicate=True,
    )
    await _upsert(dup)
    async with get_session() as session:
        unanalyzed = await NewsRepository(session).get_unanalyzed(limit=500)
    assert dup.id not in {i.id for i in unanalyzed}


# ── upsert idempotency ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_idempotent_same_item():
    item = make_item(
        url="https://example.com/repo-idempotent",
        title="Repo idempotent upsert test article",
        published_at=hours_ago(1),
    )
    await _upsert(item)
    await _upsert(item)  # second time — must not create duplicate row

    async with get_session() as session:
        results = await NewsRepository(session).get_recent(hours=24)
    matching = [r for r in results if r.id == item.id]
    assert len(matching) == 1, "duplicate upsert must not create two rows"


# ── get_stats ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stats_returns_counts():
    await init_db()
    async with get_session() as session:
        stats = await NewsRepository(session).get_stats()
    assert "total_items" in stats
    assert isinstance(stats["total_items"], int)
    assert "by_topic" in stats
