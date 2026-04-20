"""
FastAPI web layer tests.

Uses httpx ASGITransport so no server is started. The isolated_db fixture
(autouse in conftest.py) ensures all DB access goes to a fresh in-memory
SQLite database — production news.db is never touched.

Startup/shutdown events are NOT triggered (ASGITransport bypasses lifespan),
so no scheduler, background fetch, or model warmup happens during tests.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from tests.conftest import hours_ago, make_item


@pytest_asyncio.fixture
async def client(isolated_db):
    """Async HTTP client wired to the FastAPI app with the test DB."""
    import news_agent.web.app as app_module
    from news_agent.storage.database import get_session
    app_module.get_session = get_session  # type: ignore[attr-defined]

    from news_agent.web.app import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── GET / ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_home_page_returns_html(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_home_page_with_topic(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(
        url="https://example.com/openai-web",
        title="OpenAI releases GPT-5 model",
        topic="openai",
        published_at=hours_ago(2),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.get("/?topic1=openai&hours=24")
    assert resp.status_code == 200


# ── GET /api/stats ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_returns_json(client):
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_items" in data
    assert isinstance(data["total_items"], int)


@pytest.mark.asyncio
async def test_stats_reflects_inserted_items(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(url="https://example.com/stats-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.get("/api/stats")
    assert resp.json()["total_items"] >= 1


# ── POST /api/interaction — upvote / downvote ─────────────────────────────────

@pytest.mark.asyncio
async def test_interaction_upvote(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/upvote-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.post(
        "/api/interaction",
        json={"item_id": item.id, "action": "upvote"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM)
            .where(UserInteractionORM.item_id == item.id, UserInteractionORM.action == "upvote")
        )
        assert result.scalar_one_or_none() is not None, "upvote interaction should be recorded"


@pytest.mark.asyncio
async def test_interaction_unupvote(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/unupvote-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    await client.post("/api/interaction", json={"item_id": item.id, "action": "upvote"})
    resp = await client.post("/api/interaction", json={"item_id": item.id, "action": "unupvote"})
    assert resp.status_code == 200

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM)
            .where(UserInteractionORM.item_id == item.id, UserInteractionORM.action == "unupvote")
        )
        assert result.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_interaction_downvote(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/downvote-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.post(
        "/api/interaction",
        json={"item_id": item.id, "action": "downvote"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM)
            .where(UserInteractionORM.item_id == item.id, UserInteractionORM.action == "downvote")
        )
        assert result.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_interaction_undownvote(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(url="https://example.com/undownvote-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    await client.post("/api/interaction", json={"item_id": item.id, "action": "downvote"})
    resp = await client.post("/api/interaction", json={"item_id": item.id, "action": "undownvote"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_downvote_cancels_existing_upvote(client, isolated_db):
    """Downvoting an already-upvoted item should automatically cancel the upvote."""
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/mutual-exclusion-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    await client.post("/api/interaction", json={"item_id": item.id, "action": "upvote"})
    await client.post("/api/interaction", json={"item_id": item.id, "action": "downvote"})

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM)
            .where(UserInteractionORM.item_id == item.id)
            .order_by(UserInteractionORM.created_at)
        )
        actions = [r.action for r in result.scalars()]

    assert "unupvote" in actions, "downvote should auto-cancel the prior upvote"
    assert "downvote" in actions


@pytest.mark.asyncio
async def test_upvote_cancels_existing_downvote(client, isolated_db):
    """Upvoting an already-downvoted item should automatically cancel the downvote."""
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/upvote-cancels-downvote", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    await client.post("/api/interaction", json={"item_id": item.id, "action": "downvote"})
    await client.post("/api/interaction", json={"item_id": item.id, "action": "upvote"})

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM)
            .where(UserInteractionORM.item_id == item.id)
            .order_by(UserInteractionORM.created_at)
        )
        actions = [r.action for r in result.scalars()]

    assert "undownvote" in actions, "upvote should auto-cancel the prior downvote"
    assert "upvote" in actions


@pytest.mark.asyncio
async def test_downvote_does_not_cancel_if_already_unupvoted(client, isolated_db):
    """If upvote was already cancelled (unupvote), downvoting should not add another unupvote."""
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/no-double-cancel", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    await client.post("/api/interaction", json={"item_id": item.id, "action": "upvote"})
    await client.post("/api/interaction", json={"item_id": item.id, "action": "unupvote"})
    await client.post("/api/interaction", json={"item_id": item.id, "action": "downvote"})

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM)
            .where(
                UserInteractionORM.item_id == item.id,
                UserInteractionORM.action == "unupvote",
            )
        )
        unupvote_count = len(result.scalars().all())

    assert unupvote_count == 1, "should not record a duplicate unupvote"


# ── GET /api/interaction — click and read ────────────────────────────────────

@pytest.mark.asyncio
async def test_interaction_click(client, isolated_db):
    resp = await client.post(
        "/api/interaction",
        json={"item_id": "nonexistentid0000", "action": "click"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_interaction_read_with_seconds(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(url="https://example.com/read-seconds-test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.post(
        "/api/interaction",
        json={"item_id": item.id, "action": "read", "read_seconds": 90.0},
    )
    assert resp.status_code == 200


# ── GET /api/fetch/status ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_status_no_topic(client):
    resp = await client.get("/api/fetch/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert data["running"] is False


@pytest.mark.asyncio
async def test_fetch_status_with_topic(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(url="https://example.com/status-ai", topic="ai_status_test", published_at=hours_ago(1))
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.get("/api/fetch/status?topic=ai_status_test")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert data["count"] >= 1


# ── POST /api/fetch ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_no_keyword_returns_not_started(client):
    resp = await client.post("/api/fetch")
    assert resp.status_code == 200
    assert resp.json()["started"] is False


@pytest.mark.asyncio
async def test_fetch_with_keyword_starts(client):
    from news_agent import orchestrator as orch_mod
    with patch.object(orch_mod, "run_keyword_fetch", new=AsyncMock(return_value=None)):
        resp = await client.post("/api/fetch?keyword=openai&force=true")
    assert resp.status_code == 200
    assert "started" in resp.json()


# ── GET /api/panel ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_panel_no_topic_returns_html(client, isolated_db):
    resp = await client.get("/api/panel")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_panel_with_topic_returns_html(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(
        url="https://example.com/panel-topic-test",
        topic="panel_topic",
        title="Panel topic test article about AI",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.get("/api/panel?topic=panel_topic&hours=24")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# ── GET /api/digest-fragment ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_digest_fragment_no_topic_returns_empty(client):
    resp = await client.get("/api/digest-fragment")
    assert resp.status_code == 200
    assert resp.text == ""


@pytest.mark.asyncio
async def test_digest_fragment_no_digest_returns_empty_html(client):
    resp = await client.get("/api/digest-fragment?topic=nonexistenttopic")
    assert resp.status_code == 200


# ── GET /api/debug/topic/{topic} ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_debug_topic_returns_json(client, isolated_db):
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(
        url="https://example.com/debug-topic-test",
        topic="debug_topic",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    resp = await client.get("/api/debug/topic/debug_topic")
    assert resp.status_code == 200
    data = resp.json()
    assert data["topic"] == "debug_topic"
    assert data["total_items"] >= 1


@pytest.mark.asyncio
async def test_debug_topic_empty_topic(client):
    resp = await client.get("/api/debug/topic/no_such_topic_xyz")
    assert resp.status_code == 200
    assert resp.json()["total_items"] == 0


# ── Podcast endpoints ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_podcast_status_not_ready(client):
    resp = await client.get("/api/podcast/ai/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is False
    assert data["generating"] is False


@pytest.mark.asyncio
async def test_podcast_audio_404_when_not_ready(client):
    resp = await client.get("/api/podcast/nonexistent_topic/audio")
    assert resp.status_code == 404
