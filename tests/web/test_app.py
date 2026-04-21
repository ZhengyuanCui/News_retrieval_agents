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


# ── Default topics (server-side persistence for newsletter) ──────────────────

@pytest.mark.asyncio
async def test_default_topics_initially_empty(client):
    resp = await client.get("/api/default-topics")
    assert resp.status_code == 200
    assert resp.json() == {"topic1": "", "topic2": ""}


@pytest.mark.asyncio
async def test_default_topics_post_persists(client):
    """POST then GET — the topics must survive so the newsletter CLI/cron
    running in a separate process can read them."""
    resp = await client.post(
        "/api/default-topics",
        json={"topic1": "ai", "topic2": "stocks"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    # Read via the API
    resp = await client.get("/api/default-topics")
    assert resp.json() == {"topic1": "ai", "topic2": "stocks"}

    # Read via the newsletter pipeline (same path the CLI / scheduler uses)
    from news_agent.pipeline.newsletter import get_default_topics
    assert await get_default_topics() == ["ai", "stocks"]


@pytest.mark.asyncio
async def test_default_topics_post_overwrites(client):
    await client.post("/api/default-topics", json={"topic1": "ai", "topic2": "stocks"})
    await client.post("/api/default-topics", json={"topic1": "crypto", "topic2": ""})

    resp = await client.get("/api/default-topics")
    assert resp.json() == {"topic1": "crypto", "topic2": ""}

    from news_agent.pipeline.newsletter import get_default_topics
    assert await get_default_topics() == ["crypto"]


@pytest.mark.asyncio
async def test_default_topics_post_trims_whitespace(client):
    resp = await client.post(
        "/api/default-topics",
        json={"topic1": "  ai  ", "topic2": "  stocks  "},
    )
    assert resp.status_code == 200

    from news_agent.pipeline.newsletter import get_default_topics
    assert await get_default_topics() == ["ai", "stocks"]


@pytest.mark.asyncio
async def test_default_topics_post_accepts_blank_both_fields(client):
    """Saving blank defaults clears the server-side topics list."""
    await client.post("/api/default-topics", json={"topic1": "ai", "topic2": "stocks"})
    await client.post("/api/default-topics", json={"topic1": "", "topic2": ""})

    resp = await client.get("/api/default-topics")
    assert resp.json() == {"topic1": "", "topic2": ""}

    from news_agent.pipeline.newsletter import get_default_topics
    assert await get_default_topics() == []


# ── Save-defaults side-effect isolation ──────────────────────────────────────
# Saving defaults must be a pure write: it must NOT touch the news_items table,
# must NOT touch the digests table, must NOT trigger a fetch cycle, and must
# NOT invalidate any cached state. These tests lock that invariant in.

@pytest.mark.asyncio
async def test_saving_defaults_does_not_modify_news_items(client, isolated_db):
    """Saving defaults must not add, remove, or modify any news_items rows."""
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import NewsItemORM
    from sqlalchemy import select

    item = make_item(
        url="https://example.com/preserved",
        title="Preexisting item",
        topic="ai",
        summary="Original summary, must not change",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)

    async def snapshot_items():
        async with get_session() as session:
            rows = (await session.execute(select(NewsItemORM))).scalars().all()
            return [(r.id, r.title, r.summary, r.topic) for r in rows]

    before = await snapshot_items()

    resp = await client.post(
        "/api/default-topics",
        json={"topic1": "ai", "topic2": "stocks"},
    )
    assert resp.status_code == 200

    after = await snapshot_items()
    assert before == after, "news_items must not change when defaults are saved"


@pytest.mark.asyncio
async def test_saving_defaults_does_not_regenerate_digest(client, isolated_db):
    """A pre-existing digest for the just-saved topic must stay exactly as-is."""
    from datetime import datetime
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    today = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest(
            date=today, topic="ai",
            content="ORIGINAL DIGEST — must not be regenerated",
            item_count=7,
        )
        existing_before = await repo.get_digest(today, "ai")
        assert existing_before is not None
        original_generated_at = existing_before.generated_at
        original_content = existing_before.content

    await client.post("/api/default-topics", json={"topic1": "ai", "topic2": ""})

    async with get_session() as session:
        repo = NewsRepository(session)
        after = await repo.get_digest(today, "ai")
    assert after is not None
    assert after.content == original_content
    assert after.generated_at == original_generated_at
    assert after.item_count == 7


@pytest.mark.asyncio
async def test_saving_defaults_does_not_trigger_fetch(client):
    """POSTing defaults must not spin up a keyword-fetch background task."""
    from news_agent.web import app as app_module

    before = set(app_module._keyword_fetching)
    resp = await client.post(
        "/api/default-topics",
        json={"topic1": "some-brand-new-keyword", "topic2": ""},
    )
    assert resp.status_code == 200
    after = set(app_module._keyword_fetching)
    assert after == before, "saving defaults must not start a keyword fetch"


@pytest.mark.asyncio
async def test_saving_defaults_does_not_invalidate_vector_index(client):
    """The semantic vector index must not be invalidated by a pure settings write."""
    import news_agent.pipeline.vector_search as vs

    calls = {"n": 0}
    original = vs.invalidate_index

    def counting_invalidate():
        calls["n"] += 1
        return original()

    try:
        vs.invalidate_index = counting_invalidate
        await client.post(
            "/api/default-topics",
            json={"topic1": "ai", "topic2": "stocks"},
        )
    finally:
        vs.invalidate_index = original

    assert calls["n"] == 0, "saving defaults should not invalidate the vector index"


# ── Client-side JS invariants (inline in digest.html) ────────────────────────
# These assert the rendered HTML/JS has the properties that make saveDefaultTopics
# a pure save — no navigation, no panel reload — so a future refactor that
# reintroduces window.location.reload() inside the save path trips the test.

@pytest.mark.asyncio
async def test_digest_page_save_defaults_js_does_not_navigate(client):
    """saveDefaultTopics() must not call window.location = or window.location.reload."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Isolate just the saveDefaultTopics function body.
    start = html.index("async function saveDefaultTopics(")
    end = html.index("function showSettingsToast", start)
    save_fn_src = html[start:end]

    forbidden_patterns = [
        "window.location =",
        "window.location=",
        "window.location.href",
        "window.location.reload",
        "location.reload",
    ]
    for pat in forbidden_patterns:
        assert pat not in save_fn_src, (
            f"saveDefaultTopics must not navigate (found {pat!r}); "
            "saving defaults should be a pure save."
        )


@pytest.mark.asyncio
async def test_digest_page_save_defaults_js_posts_to_server(client):
    """saveDefaultTopics() must POST to /api/default-topics with keepalive so
    the request survives any subsequent page navigation."""
    resp = await client.get("/")
    html = resp.text

    start = html.index("async function saveDefaultTopics(")
    end = html.index("function showSettingsToast", start)
    save_fn_src = html[start:end]

    assert "/api/default-topics" in save_fn_src
    assert "method: 'POST'" in save_fn_src or "method:\"POST\"" in save_fn_src
    assert "keepalive" in save_fn_src, (
        "POST must use keepalive:true so it isn't cancelled by navigation"
    )
    assert "await fetch" in save_fn_src, (
        "fetch must be awaited so the save completes before the function returns"
    )


@pytest.mark.asyncio
async def test_digest_page_save_defaults_js_shows_toast(client):
    """saveDefaultTopics() should give the user feedback via a toast,
    since the visual page state doesn't change."""
    resp = await client.get("/")
    html = resp.text

    start = html.index("async function saveDefaultTopics(")
    end = html.index("function showSettingsToast", start)
    save_fn_src = html[start:end]

    assert "showSettingsToast" in save_fn_src
    assert "toggleSettings()" in save_fn_src


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


# ── Newsletter audio endpoints ────────────────────────────────────────────────
#
# These endpoints back the <audio> player embedded in the daily newsletter.
# They must serve the MP3 with Content-Disposition: inline (so Gmail /
# Chrome / Safari play it instead of downloading) and support HTTP Range
# requests (so HTMLMediaElement can seek/stream progressively).  The
# companion /newsletter/player/<slug> endpoint renders a small HTML page
# so the "Play briefing in browser" link never opens a .mp3 URL directly,
# which some browsers auto-download regardless of the response headers.

def _seed_audio_file(content: bytes = b"\xff\xfb" + b"\x00" * 2000) -> str:
    """Drop a fake MP3 into the configured newsletter_audio_dir and return
    its filename.  The first two bytes are a valid MPEG frame sync so any
    client-side sniff that does care sees an audio file."""
    import news_agent.web.app as app_module
    app_module._newsletter_audio_dir.mkdir(parents=True, exist_ok=True)
    fname = "ai-briefing-test.mp3"
    (app_module._newsletter_audio_dir / fname).write_bytes(content)
    return fname


@pytest.mark.asyncio
async def test_newsletter_audio_serves_inline_with_range_support(client):
    fname = _seed_audio_file()
    resp = await client.get(f"/newsletter/audio/{fname}")
    assert resp.status_code == 200
    # The critical headers for inline playback
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.headers["content-disposition"].startswith("inline;")
    assert "attachment" not in resp.headers["content-disposition"].lower()
    assert resp.headers["accept-ranges"] == "bytes"


@pytest.mark.asyncio
async def test_newsletter_audio_range_request_returns_206(client):
    fname = _seed_audio_file(b"\xff\xfb" + b"X" * 1000)
    resp = await client.get(
        f"/newsletter/audio/{fname}", headers={"Range": "bytes=0-99"}
    )
    assert resp.status_code == 206
    assert resp.headers["content-range"].startswith("bytes 0-99/")
    assert resp.headers["content-length"] == "100"
    assert len(resp.content) == 100


@pytest.mark.asyncio
async def test_newsletter_audio_unsatisfiable_range_returns_416(client):
    fname = _seed_audio_file(b"tiny")
    resp = await client.get(
        f"/newsletter/audio/{fname}", headers={"Range": "bytes=9999-"}
    )
    assert resp.status_code == 416
    assert resp.headers["content-range"] == "bytes */4"


@pytest.mark.asyncio
async def test_newsletter_audio_404_for_missing_file(client):
    resp = await client.get("/newsletter/audio/does-not-exist.mp3")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_newsletter_audio_rejects_path_traversal(client):
    # A simple "../secret" is normalised by httpx/starlette before it
    # reaches our handler, so we test the in-handler defense via a name
    # containing a literal slash in a single path segment.
    resp = await client.get("/newsletter/audio/..%2Fsecret.mp3")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_newsletter_player_returns_html_with_embedded_audio(client):
    fname = _seed_audio_file()
    # slug without .mp3 — this is what the email links to
    slug = fname.removesuffix(".mp3")
    resp = await client.get(f"/newsletter/player/{slug}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "<audio" in body and "controls" in body
    # The <audio src> points at the streaming endpoint, NOT the raw file
    assert f'src="/newsletter/audio/{fname}"' in body
    # No attachment-style disposition on the HTML page itself
    assert "attachment" not in resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_newsletter_player_accepts_slug_with_mp3_suffix(client):
    """Back-compat: older emails link to /newsletter/player/<file>.mp3."""
    fname = _seed_audio_file()
    resp = await client.get(f"/newsletter/player/{fname}")
    assert resp.status_code == 200
    assert "<audio" in resp.text


@pytest.mark.asyncio
async def test_newsletter_player_404_for_missing_file(client):
    resp = await client.get("/newsletter/player/does-not-exist")
    assert resp.status_code == 404
