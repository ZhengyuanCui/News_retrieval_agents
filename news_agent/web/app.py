from __future__ import annotations

from datetime import datetime
from pathlib import Path

import re

import jinja2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from news_agent.preference import record_interaction, recompute_preferences
from news_agent.storage import get_session, init_db
from news_agent.storage.repository import NewsRepository

app = FastAPI(title="News Agent")

# Track background fetch state
_fetch_running = False
_podcast_generating: set[str] = set()
_podcast_errors: dict[str, str] = {}
# In-memory podcast cache: {topic: {"audio": bytes, "hours": float}}
_podcast_cache: dict[str, dict] = {}
# Background tasks to cancel on shutdown
_background_tasks: set = set()

BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

# Use jinja2 directly to avoid Python 3.14 cache-key bug in starlette's wrapper
def _strip_html(text: str) -> str:
    """Remove HTML tags from text (for RSS content that contains markup)."""
    return re.sub(r"<[^>]+>", " ", text or "").strip()


_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(BASE / "templates")),
    autoescape=jinja2.select_autoescape(["html"]),
    auto_reload=True,
    cache_size=0,  # disable cache to avoid the hashability issue
)
_jinja_env.filters["strip_html"] = _strip_html


def _bold_md(text: str) -> "markupsafe.Markup":
    """Convert **text** markdown to <strong>text</strong>, return as safe HTML."""
    from markupsafe import Markup, escape
    escaped = escape(text)
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(escaped))
    return Markup(result)


_jinja_env.filters["bold_md"] = _bold_md

import json as _json
_jinja_env.filters["tojson"] = lambda v: _json.dumps(v)


def _parse_digest(raw: str | None) -> dict | None:
    """Parse stored digest string into {headline, bullets}."""
    if not raw:
        return None
    # Legacy pipe-delimited format
    if "|||" in raw:
        parts = raw.split("|||")
        return {"headline": parts[0], "bullets": [b for b in parts[1:] if b.strip()]}
    # New plain-text format: first line = headline, rest = bullets
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return None
    return {"headline": lines[0], "bullets": lines[1:]}


def render(template_name: str, **ctx) -> HTMLResponse:
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**ctx))


def _track(coro):
    """Create a tracked background task that is cancelled on shutdown."""
    import asyncio
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


_scheduler = None


@app.on_event("startup")
async def startup():
    global _scheduler
    await init_db()

    # Migrate legacy hardcoded topic labels to source-based labels
    from sqlalchemy import update as _update, text as _text
    from news_agent.models import NewsItemORM
    async with get_session() as session:
        await session.execute(
            _update(NewsItemORM)
            .where(NewsItemORM.topic.in_(["ai", "stocks", "tech"]))
            .values(topic=NewsItemORM.source)
        )
        await session.commit()

    # Add language column if it doesn't exist yet (one-time migration)
    from news_agent.storage.database import engine as _engine
    async with _engine.begin() as conn:
        try:
            await conn.execute(_text(
                "ALTER TABLE news_items ADD COLUMN language TEXT NOT NULL DEFAULT 'en'"
            ))
        except Exception:
            pass  # column already exists
    # Remove any leftover podcast files from previous disk-based implementation
    legacy_dir = BASE.parent.parent / "data" / "podcasts"
    if legacy_dir.exists():
        for f in legacy_dir.glob("*.mp3"):
            f.unlink(missing_ok=True)

    # Pre-warm ML models in a background thread so the first search doesn't
    # pay the model-loading cost (~5-15s for sentence-transformers + spam classifier)
    import asyncio as _asyncio
    def _warmup_models():
        try:
            from news_agent.spam import warmup as _spam_warmup
            _spam_warmup()
        except Exception:
            pass
        try:
            from news_agent.pipeline.deduplicator import Deduplicator as _Dedup
            _Dedup()._load_semantic_model()
        except Exception:
            pass
    loop = _asyncio.get_event_loop()
    loop.run_in_executor(None, _warmup_models)

    # Start the background scheduler (fetch every N hours, digest at 08:00, prune at 03:00)
    from news_agent.scheduler import build_scheduler
    _scheduler = build_scheduler()
    _scheduler.start()

    # Run an initial fetch immediately in the background
    from news_agent.scheduler import _fetch_job
    _track(_fetch_job())


@app.on_event("shutdown")
async def shutdown():
    import asyncio
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    for task in list(_background_tasks):
        task.cancel()
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

def _parse_languages(langs: str) -> list[str] | None:
    """Parse comma-separated language codes into a list, or None if empty/all."""
    if not langs or langs.lower() == "all":
        return None
    return [l.strip() for l in langs.split(",") if l.strip()]


@app.get("/", response_class=HTMLResponse)
async def digest_page(hours: float = 24, topic1: str = "", topic2: str = "", langs: str = ""):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    languages = _parse_languages(langs)

    async with get_session() as session:
        repo = NewsRepository(session)
        items1 = await repo.search(topic1, hours=hours, languages=languages) if topic1 else await repo.get_recent(hours=hours, limit=60, languages=languages)
        items2 = await repo.search(topic2, hours=hours, languages=languages) if topic2 else await repo.get_recent(hours=hours, limit=60, languages=languages)
        starred = await repo.get_starred_ids()
        digest1 = await repo.get_digest(today, topic1.lower()) if topic1 else None
        digest2 = await repo.get_digest(today, topic2.lower()) if topic2 else None

    return render(
        "digest.html",
        items1=items1,
        items2=items2,
        topic1=topic1,
        topic2=topic2,
        starred_ids=starred,
        hours=hours,
        langs=langs,
        digest1=_parse_digest(digest1.content if digest1 else None),
        digest2=_parse_digest(digest2.content if digest2 else None),
        date=datetime.utcnow().strftime("%B %d, %Y"),
    )


@app.get("/search", response_class=HTMLResponse)
async def search_page(q: str = "", hours: float = 24):
    items = []
    starred = set()
    if q.strip():
        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.search(q.strip(), hours=hours)
            starred = await repo.get_starred_ids()
        # Always kick off a background fetch to find more/better results
        if q.strip() not in _keyword_fetching:
            from news_agent.orchestrator import run_keyword_fetch
            async def _bg():
                _keyword_fetching.add(q.strip())
                try:
                    await run_keyword_fetch(q.strip())
                finally:
                    _keyword_fetching.discard(q.strip())
            _track(_bg())

    return render(
        "search.html",
        q=q,
        items=items,
        starred_ids=starred,
        hours=hours,
        date=datetime.utcnow().strftime("%B %d, %Y"),
    )


# ── Interaction API ───────────────────────────────────────────────────────────

class InteractionPayload(BaseModel):
    item_id: str
    action: str          # "click" | "read" | "star" | "unstar"
    read_seconds: float | None = None


@app.post("/api/interaction")
async def log_interaction(payload: InteractionPayload):
    async with get_session() as session:
        await record_interaction(session, payload.item_id, payload.action, payload.read_seconds)
        if payload.action in ("star", "unstar"):
            repo = NewsRepository(session)
            await repo.set_starred(payload.item_id, payload.action == "star")
        await recompute_preferences(session)
    return {"ok": True}


_keyword_fetching: set[str] = set()  # keywords currently being fetched


_keyword_last_fetch: dict[str, datetime] = {}
_KEYWORD_COOLDOWN_SECONDS = 300  # don't re-fetch same keyword within 5 minutes

@app.post("/api/fetch")
async def trigger_fetch(keyword: str = "", force: bool = False):
    """Fetch news for a keyword. Pass force=true to bypass the cooldown (e.g. manual refresh)."""
    if not keyword:
        return {"started": False, "reason": "keyword required"}
    if keyword in _keyword_fetching:
        return {"started": False, "reason": "already fetching"}

    if not force:
        last = _keyword_last_fetch.get(keyword.lower())
        if last and (datetime.utcnow() - last).total_seconds() < _KEYWORD_COOLDOWN_SECONDS:
            return {"started": False, "reason": "cooldown"}

    from news_agent.orchestrator import run_keyword_fetch

    async def _run():
        _keyword_fetching.add(keyword)
        try:
            await run_keyword_fetch(keyword)
            _keyword_last_fetch[keyword.lower()] = datetime.utcnow()
        finally:
            _keyword_fetching.discard(keyword)

    _track(_run())
    return {"started": True}


@app.get("/api/fetch/status")
async def fetch_status(topic: str = "", hours: float = 24):
    """Return current item count and whether a fetch is running for this topic."""
    count = 0
    if topic:
        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.search(topic, hours=hours)
            count = len(items)
    return {"running": topic in _keyword_fetching, "count": count}


@app.get("/api/digest-fragment", response_class=HTMLResponse)
async def digest_fragment(topic: str = "", hours: float = 24):
    """Return the digest-summary HTML for a topic (empty string if none exists yet)."""
    if not topic:
        return HTMLResponse("")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        repo = NewsRepository(session)
        digest = await repo.get_digest(today, topic.lower())
    parsed = _parse_digest(digest.content if digest else None)
    return render("partials/digest_summary.html", digest=parsed)


@app.get("/api/digest-stream/{topic}")
async def digest_stream(topic: str, hours: float = 24):
    """Stream digest generation via SSE. Serves cached digest instantly if available,
    otherwise streams live from Claude and stores when complete."""
    import json as _json

    today = datetime.utcnow().strftime("%Y-%m-%d")

    async def event_stream():
        async with get_session() as session:
            repo = NewsRepository(session)
            existing = await repo.get_digest(today, topic.lower())

        from news_agent.pipeline.analyzer import ClaudeAnalyzer
        from news_agent.config import settings as _cfg

        if not _cfg.anthropic_api_key:
            yield f"data: {_json.dumps({'t': 'ANTHROPIC_API_KEY not set.'})}\n\n"
            yield f"data: {_json.dumps({'done': True})}\n\n"
            return

        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.get_recent(hours=hours, topic=topic, limit=30)

        if existing:
            # Delete cached error messages from DB so they can never be re-served.
            cached_content = existing.content or ""
            if cached_content.startswith("Digest generation failed") or "[Error:" in cached_content:
                async with get_session() as session:
                    repo = NewsRepository(session)
                    await repo.delete_digest(today, topic.lower())
                existing = None
                logger.info("Deleted bad cached digest for '%s'", topic)

        stale_fallback = None  # cached content to fall back to if regeneration fails
        if existing:
            cached_count = existing.item_count or 0
            current_count = len(items)
            # Only regenerate if item count more than doubled — normal fetches bring
            # incremental updates, not wholesale replacements.
            is_stale = current_count > cached_count * 2 and current_count - cached_count > 20
            if not is_stale:
                text = existing.content
                chunk_size = 8
                for i in range(0, len(text), chunk_size):
                    yield f"data: {_json.dumps({'t': text[i:i+chunk_size]})}\n\n"
                yield f"data: {_json.dumps({'done': True})}\n\n"
                return
            stale_fallback = existing.content
            logger.info(
                "Digest cache stale for '%s': cached=%d items, current=%d — regenerating",
                topic, cached_count, current_count,
            )

        if not items:
            yield f"data: {_json.dumps({'t': f'No recent {topic} news found.'})}\n\n"
            yield f"data: {_json.dumps({'done': True})}\n\n"
            return

        analyzer = ClaudeAnalyzer()
        full_text = ""
        try:
            async for chunk in analyzer.generate_digest_stream(items, topic):
                full_text += chunk
                yield f"data: {_json.dumps({'t': chunk})}\n\n"
        except Exception as e:
            logger.error("Digest stream failed for '%s': %s", topic, e)
            # Fall back to cached digest rather than showing a raw error
            if stale_fallback:
                logger.info("Falling back to stale cached digest for '%s'", topic)
                chunk_size = 8
                for i in range(0, len(stale_fallback), chunk_size):
                    yield f"data: {_json.dumps({'t': stale_fallback[i:i+chunk_size]})}\n\n"

        yield f"data: {_json.dumps({'done': True})}\n\n"

        # Store the completed digest
        if full_text.strip():
            async with get_session() as session:
                repo = NewsRepository(session)
                await repo.upsert_digest(today, topic, full_text.strip(), len(items))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/panel", response_class=HTMLResponse)
async def panel_fragment(topic: str = "", hours: float = 24, langs: str = ""):
    """Return just the news-list inner HTML for a single panel (used for live updates)."""
    import asyncio
    from news_agent.collectors.rss import _resolve_url

    languages = _parse_languages(langs)
    async with get_session() as session:
        repo = NewsRepository(session)
        items = await repo.search(topic, hours=hours, languages=languages) if topic else await repo.get_recent(hours=hours, limit=60, languages=languages)
        starred = await repo.get_starred_ids()

    # Resolve any stale Google News redirect URLs and persist the real URL
    google_items = [i for i in items if "news.google.com" in i.url]
    if google_items:
        resolved = await asyncio.gather(*[_resolve_url(i.url) for i in google_items])
        changed = [(item, url) for item, url in zip(google_items, resolved) if url != item.url]
        if changed:
            async with get_session() as session:
                repo = NewsRepository(session)
                for item, real_url in changed:
                    item.url = real_url
                    await repo.update_url(item.id, real_url)

    return render("partials/panel_items.html", topic=topic, items=items, starred_ids=starred)


@app.post("/api/podcast/{topic}")
async def generate_podcast(topic: str, hours: float = 24):
    """Generate a podcast for a topic in the background."""
    import asyncio
    from news_agent.config import settings as _settings

    if topic in _podcast_generating:
        return {"started": False, "reason": "already generating"}

    if not _settings.anthropic_api_key:
        return {"started": False, "reason": "ANTHROPIC_API_KEY not set in .env"}

    # If cached podcast was for a different hours window, invalidate it
    cached = _podcast_cache.get(topic)
    if cached and cached["hours"] != hours:
        del _podcast_cache[topic]
        _podcast_errors.pop(topic, None)

    async with get_session() as session:
        repo = NewsRepository(session)
        items = await repo.get_recent(hours=hours, topic=topic, limit=30)

    if not items:
        return {"started": False, "reason": "No news items found — run a fetch first"}

    async def _run():
        from news_agent.pipeline.podcast import PodcastGenerator
        import logging as _log
        import tempfile, os
        try:
            _podcast_generating.add(topic)
            gen = PodcastGenerator()
            # Generate to a temp file, then read into memory
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            await asyncio.get_event_loop().run_in_executor(
                None, gen.generate, items, topic, tmp_path
            )
            _podcast_cache[topic] = {"audio": tmp_path.read_bytes(), "hours": hours}
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            _log.getLogger(__name__).error("Podcast generation failed: %s", e)
            _podcast_errors[topic] = str(e)
        finally:
            _podcast_generating.discard(topic)

    _podcast_errors.pop(topic, None)
    _track(_run())
    return {"started": True}


@app.get("/api/podcast/{topic}/status")
async def podcast_status(topic: str, hours: float = 24):
    cached = _podcast_cache.get(topic)
    ready = cached is not None and cached["hours"] == hours
    return {
        "generating": topic in _podcast_generating,
        "ready": ready,
        "url": f"/api/podcast/{topic}/audio" if ready else None,
        "error": _podcast_errors.get(topic),
    }


@app.get("/api/podcast/{topic}/audio")
async def podcast_audio(topic: str):
    cached = _podcast_cache.get(topic)
    if not cached:
        raise HTTPException(404, "Podcast not ready yet")
    return Response(
        content=cached["audio"],
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'inline; filename="{topic}-briefing.mp3"'},
    )


@app.get("/api/stats")
async def get_stats():
    async with get_session() as session:
        repo = NewsRepository(session)
        return await repo.get_stats()


@app.get("/api/debug/topic/{topic}")
async def debug_topic(topic: str, hours: float = 24):
    """Diagnostic: show item count, analysis state, and digest status for a topic."""
    from sqlalchemy import select, func
    from news_agent.models import NewsItemORM, DigestORM
    today = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        # Item counts
        all_q = await session.execute(
            select(func.count()).select_from(NewsItemORM)
            .where(NewsItemORM.topic.ilike(topic))
        )
        dup_q = await session.execute(
            select(func.count()).select_from(NewsItemORM)
            .where(NewsItemORM.topic.ilike(topic), NewsItemORM.is_duplicate == True)
        )
        analyzed_q = await session.execute(
            select(func.count()).select_from(NewsItemORM)
            .where(NewsItemORM.topic.ilike(topic), NewsItemORM.summary != None)
        )
        # Digests for this topic
        digests_q = await session.execute(
            select(DigestORM).where(DigestORM.topic == topic.lower())
            .order_by(DigestORM.generated_at.desc()).limit(5)
        )
        digests = [{"id": d.id, "date": d.date, "item_count": d.item_count,
                    "generated_at": str(d.generated_at),
                    "content_preview": d.content[:80]} for d in digests_q.scalars()]
    return {
        "topic": topic,
        "total_items": all_q.scalar(),
        "duplicate_items": dup_q.scalar(),
        "analyzed_items": analyzed_q.scalar(),
        "digests": digests,
        "today": today,
    }
