from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import html
import re

import jinja2

logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from news_agent.preference import apply_preference_boost, get_preference_scores, record_interaction, recompute_preferences
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

# Serve newsletter MP3s so the <audio> player embedded in the daily email
# can stream them instead of forcing a download.
#
# We intentionally do NOT use StaticFiles here — its default
# Content-Disposition and the .mp3 URL suffix cause some browsers / mail
# clients to save the file instead of playing it inline.  These two
# endpoints guarantee:
#   • Content-Type: audio/mpeg
#   • Content-Disposition: inline (never attachment)
#   • HTTP Range support so <audio> can seek/stream progressively
#   • A /newsletter/player/ landing page the "Play briefing" link can point
#     at so the URL the user clicks doesn't end in ".mp3" (which some
#     browsers / extensions auto-download).
from news_agent.config import settings as _settings  # local import avoids cycle

_PROJECT_ROOT = BASE.parent.parent
_newsletter_audio_dir = Path(_settings.newsletter_audio_dir)
if not _newsletter_audio_dir.is_absolute():
    _newsletter_audio_dir = _PROJECT_ROOT / _newsletter_audio_dir
_newsletter_audio_dir.mkdir(parents=True, exist_ok=True)


def _safe_audio_path(filename: str) -> Path:
    """Resolve ``filename`` inside the newsletter audio dir, or 404.

    Rejects path-traversal attempts (``..``, absolute paths) and anything
    that doesn't actually live under the configured audio directory.
    """
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(404)
    candidate = (_newsletter_audio_dir / filename).resolve()
    try:
        candidate.relative_to(_newsletter_audio_dir.resolve())
    except ValueError:
        raise HTTPException(404)
    if not candidate.is_file():
        raise HTTPException(404)
    return candidate


def _parse_range(header: str | None, file_size: int) -> tuple[int, int] | None:
    """Parse an HTTP ``Range: bytes=start-end`` header.  Returns None if
    the header is missing or malformed; raises HTTPException(416) if the
    range is unsatisfiable."""
    if not header or not header.startswith("bytes="):
        return None
    try:
        raw = header.split("=", 1)[1].split(",", 1)[0].strip()
        start_s, end_s = raw.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except (ValueError, IndexError):
        return None
    if start < 0 or end < start or start >= file_size:
        raise HTTPException(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)
    return start, end


@app.get("/newsletter/audio/{filename}")
async def serve_newsletter_audio(filename: str, request: Request) -> Response:
    """Stream a newsletter MP3 inline with Range support.

    The ``Content-Disposition: inline`` header is what convinces browsers
    and mail-client previewers to play the audio instead of prompting the
    user to save it.
    """
    path = _safe_audio_path(filename)
    file_size = path.stat().st_size
    rng = _parse_range(request.headers.get("range"), file_size)

    headers = {
        "Content-Type": "audio/mpeg",
        # `inline` is the critical bit.  `filename=` is fine — it just
        # names the resource, it does not force a download so long as the
        # disposition is `inline`.
        "Content-Disposition": f'inline; filename="{filename}"',
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }

    if rng is None:
        def _full():
            with path.open("rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk
        headers["Content-Length"] = str(file_size)
        return StreamingResponse(_full(), status_code=200, headers=headers, media_type="audio/mpeg")

    start, end = rng
    length = end - start + 1

    def _partial():
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    headers["Content-Length"] = str(length)
    return StreamingResponse(_partial(), status_code=206, headers=headers, media_type="audio/mpeg")


@app.get("/newsletter/player/{slug}", response_class=HTMLResponse)
async def newsletter_audio_player(slug: str) -> HTMLResponse:
    """Tiny HTML page that plays a briefing inline.

    The email's "Play briefing in browser" link points here.  The URL path
    intentionally does NOT end in ``.mp3`` — some browsers and browser
    extensions treat any ``.mp3`` URL opened in a new tab as a download
    even when the server sends ``Content-Disposition: inline``.  By
    serving a regular HTML page with an embedded <audio> element that
    streams from ``/newsletter/audio/<file>.mp3``, we guarantee the click
    lands on a playable page instead of triggering a save dialog.

    The slug may be the bare stem (preferred, e.g. ``ai-briefing-2026-01-01``)
    or include the ``.mp3`` extension for backward compatibility.
    """
    filename = slug if slug.lower().endswith(".mp3") else f"{slug}.mp3"
    path = _safe_audio_path(filename)
    audio_url = f"/newsletter/audio/{path.name}"
    safe_name = html.escape(path.name)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Briefing — {safe_name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
             Roboto, sans-serif; background: #f5f5f7; color: #1d1d1f;
             margin: 0; padding: 48px 16px; display: flex;
             justify-content: center; }}
    .card {{ background: #fff; border-radius: 12px; padding: 28px 32px;
             max-width: 520px; width: 100%;
             box-shadow: 0 4px 24px rgba(0,0,0,.06); }}
    h1 {{ margin: 0 0 6px 0; font-size: 18px; }}
    .meta {{ color: #86868b; font-size: 13px; margin-bottom: 18px;
             word-break: break-all; }}
    audio {{ width: 100%; }}
    .hint {{ margin-top: 16px; font-size: 12px; color: #86868b; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>&#127911; News briefing</h1>
    <div class="meta">{safe_name}</div>
    <audio controls autoplay preload="auto" src="{audio_url}"></audio>
    <div class="hint">Press play if autoplay is blocked by your browser.</div>
  </div>
</body>
</html>"""
    return HTMLResponse(page)

# Use jinja2 directly to avoid Python 3.14 cache-key bug in starlette's wrapper
def _strip_html(text: str) -> str:
    """Remove HTML tags from text (for RSS content that contains markup).
    Unescapes HTML entities first so &lt;a href="..."&gt; is also stripped.
    """
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def _has_llm_key() -> bool:
    from news_agent.config import settings as _s
    return bool(_s.llm_api_key or _s.anthropic_api_key)


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

    # One-time column migrations
    from news_agent.storage.database import engine as _engine
    async with _engine.begin() as conn:
        for ddl in (
            "ALTER TABLE news_items ADD COLUMN language TEXT NOT NULL DEFAULT 'en'",
            "ALTER TABLE news_items ADD COLUMN cluster_id TEXT",
        ):
            try:
                await conn.execute(_text(ddl))
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
            # Loading the shared model warms up both dedup and semantic ranking
            from news_agent.pipeline.embeddings import get_model
            get_model()
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
    import asyncio
    today = datetime.utcnow().strftime("%Y-%m-%d")
    languages = _parse_languages(langs)

    # Min 72h pool so episodic sources (curated YouTube, frontier-lab blogs) appear
    search_hours = max(hours, 72)
    async with get_session() as session:
        repo = NewsRepository(session)
        items1 = await repo.search(topic1, hours=search_hours, languages=languages) if topic1 else await repo.get_recent(hours=hours, limit=60, languages=languages)
        items2 = await repo.search(topic2, hours=search_hours, languages=languages) if topic2 else await repo.get_recent(hours=hours, limit=60, languages=languages)
        digest1 = await repo.get_digest(today, topic1.lower()) if topic1 else None
        digest2 = await repo.get_digest(today, topic2.lower()) if topic2 else None
        prefs = await get_preference_scores(session)

    items1 = apply_preference_boost(items1, prefs)
    items2 = apply_preference_boost(items2, prefs)

    from news_agent.pipeline.ranker import rank_by_query
    loop = asyncio.get_event_loop()
    if topic1 and items1:
        items1 = await loop.run_in_executor(None, rank_by_query, topic1, items1)
    if topic2 and items2:
        items2 = await loop.run_in_executor(None, rank_by_query, topic2, items2)

    return render(
        "digest.html",
        items1=items1,
        items2=items2,
        topic1=topic1,
        topic2=topic2,
        hours=hours,
        langs=langs,
        digest1=_parse_digest(digest1.content if digest1 else None),
        digest2=_parse_digest(digest2.content if digest2 else None),
        date=datetime.utcnow().strftime("%B %d, %Y"),
    )


# ── Interaction API ───────────────────────────────────────────────────────────

class InteractionPayload(BaseModel):
    item_id: str
    # "click" | "read" | "upvote" | "unupvote" | "downvote" | "undownvote"
    # Upvote and downvote are mutually exclusive — _cancel_opposite_vote() below
    # auto-records the cancel action before writing the new vote.
    action: str
    read_seconds: float | None = None


_VOTE_OPPOSITES = {"upvote": "downvote", "downvote": "upvote"}
_VOTE_CANCELS   = {"upvote": "unupvote", "downvote": "undownvote"}


async def _cancel_opposite_vote(session, item_id: str, action: str) -> None:
    """If the user already has the opposite vote active, cancel it first."""
    opposite = _VOTE_OPPOSITES.get(action)
    if not opposite:
        return
    from sqlalchemy import select as _sel
    from news_agent.models import UserInteractionORM as _UIO
    result = await session.execute(
        _sel(_UIO)
        .where(_UIO.item_id == item_id, _UIO.action.in_([opposite, _VOTE_CANCELS[opposite]]))
        .order_by(_UIO.created_at.desc())
        .limit(1)
    )
    last = result.scalar_one_or_none()
    if last and last.action == opposite:
        await record_interaction(session, item_id, _VOTE_CANCELS[opposite])


@app.post("/api/interaction")
async def log_interaction(payload: InteractionPayload):
    async with get_session() as session:
        await _cancel_opposite_vote(session, payload.item_id, payload.action)
        await record_interaction(session, payload.item_id, payload.action, payload.read_seconds)
        # Tombstone lifecycle: downvote hides the item across re-fetches;
        # upvote / undownvote clears the tombstone. Respects the feature flag.
        if _settings.dismiss_on_downvote:
            from news_agent.storage.repository import NewsRepository as _Repo
            repo = _Repo(session)
            if payload.action == "downvote":
                await repo.dismiss(payload.item_id, reason="downvote")
            elif payload.action in ("upvote", "undownvote"):
                await repo.undismiss(payload.item_id)
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
    """Return current item count and whether a fetch is running for this topic.

    Counts by the stored `topic` label (what run_keyword_fetch tagged the items
    with) rather than doing a content search — the UI uses this to poll for
    "N items have arrived for the keyword I just triggered", so it needs the
    canonical topic count, not BM25 relevance.
    """
    count = 0
    if topic:
        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.get_recent(topic=topic, hours=hours, limit=500)
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
        from news_agent.pipeline.analyzer import LLMAnalyzer, is_question

        if not _has_llm_key():
            yield f"data: {_json.dumps({'t': 'No LLM API key configured. Set LLM_API_KEY or ANTHROPIC_API_KEY in .env.'})}\n\n"
            yield f"data: {_json.dumps({'done': True})}\n\n"
            return

        # ── Question mode ────────────────────────────────────────────────────
        # If the topic looks like a natural-language question, retrieve
        # semantically relevant articles and answer the question directly.
        # Q&A responses are not cached — they're always generated fresh.
        if is_question(topic):
            # Use at least 168h so Q&A has enough context, but respect longer
            # ranges chosen by the user (e.g. 30-day slider).
            qa_hours = max(hours, 168)
            async with get_session() as session:
                repo = NewsRepository(session)
                items = await repo.search(topic, hours=qa_hours, limit=20)
            if not items:
                yield f"data: {_json.dumps({'t': 'No relevant news found to answer this question.'})}\n\n"
                yield f"data: {_json.dumps({'done': True})}\n\n"
                return
            analyzer = LLMAnalyzer()
            try:
                async for chunk in analyzer.answer_question_stream(topic, items):
                    yield f"data: {_json.dumps({'t': chunk})}\n\n"
            except Exception as e:
                logger.error("Q&A stream failed for %r: %s", topic, e)
                yield f"data: {_json.dumps({'t': f'Answer generation failed: {e}'})}\n\n"
            yield f"data: {_json.dumps({'done': True})}\n\n"
            return

        # ── Standard digest mode ─────────────────────────────────────────────
        async with get_session() as session:
            repo = NewsRepository(session)
            existing = await repo.get_digest(today, topic.lower())

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

        analyzer = LLMAnalyzer()
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
async def panel_fragment(
    topic: str = "",
    hours: float = 24,
    langs: str = "",
    alpha: float | None = None,
):
    """Return just the news-list inner HTML for a single panel (used for live updates).

    `alpha` (0..1) overrides the hybrid BM25/semantic blend for this request.
    Out-of-range values are clamped in NewsRepository.search. None leaves the
    configured default_hybrid_alpha in effect.
    """
    import asyncio
    from news_agent.collectors.rss import _resolve_url

    languages = _parse_languages(langs)
    from news_agent.pipeline.analyzer import is_question as _is_question
    # For question queries match the same window used by the Q&A digest stream
    # so the news list and the summary always draw from the same article set.
    effective_hours = max(hours, 168) if topic and _is_question(topic) else hours
    widened_hours: float | None = None  # set when the window was auto-expanded

    # Always search at least 72h so episodic sources (curated YouTube channels,
    # frontier-lab blogs) that post every 2-5 days are included in the pool.
    # rank_by_query re-orders by freshness + quality, so recent items still lead.
    search_hours = max(effective_hours, 72) if topic else effective_hours

    async with get_session() as session:
        repo = NewsRepository(session)
        if topic:
            items = await repo.search(
                topic, hours=search_hours, languages=languages, hybrid_alpha=alpha,
            )
        else:
            items = await repo.get_recent(hours=hours, limit=60, languages=languages)
        prefs = await get_preference_scores(session)

    items = apply_preference_boost(items, prefs)

    # Post-filter to requested window if there are enough items; otherwise keep
    # the wider pool and show the widened indicator to the user.
    if topic and not _is_question(topic) and search_hours > effective_hours:
        from datetime import timedelta as _td
        cutoff_dt = datetime.utcnow() - _td(hours=effective_hours)
        hours_filtered = [i for i in items if i.published_at >= cutoff_dt]
        if len(hours_filtered) >= 5:
            items = hours_filtered
        else:
            widened_hours = search_hours

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

    # Re-rank by semantic similarity to the search query
    if topic and items:
        from news_agent.pipeline.ranker import rank_by_query
        loop = asyncio.get_event_loop()
        items = await loop.run_in_executor(None, rank_by_query, topic, items)

    return render("partials/panel_items.html", topic=topic, items=items,
                  hours=hours, widened_hours=widened_hours)


@app.post("/api/podcast/{topic}")
async def generate_podcast(topic: str, hours: float = 24):
    """Generate a podcast for a topic in the background."""
    import asyncio
    from news_agent.config import settings as _settings

    if topic in _podcast_generating:
        return {"started": False, "reason": "already generating"}

    if not _has_llm_key():
        return {"started": False, "reason": "No LLM API key set in .env (LLM_API_KEY or ANTHROPIC_API_KEY)"}

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


class DefaultTopicsPayload(BaseModel):
    topic1: str = ""
    topic2: str = ""


@app.get("/api/default-topics")
async def get_default_topics():
    """Return the server-persisted default topics (used by the newsletter scheduler)."""
    async with get_session() as session:
        repo = NewsRepository(session)
        raw = await repo.get_setting("default_topics", "")
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return {
        "topic1": parts[0] if len(parts) > 0 else "",
        "topic2": parts[1] if len(parts) > 1 else "",
    }


@app.post("/api/default-topics")
async def set_default_topics(payload: DefaultTopicsPayload):
    """Persist the user's default topics server-side so the newsletter scheduler
    can pick them up. Called by the UI's Default Topics settings panel."""
    value = f"{payload.topic1.strip()}|{payload.topic2.strip()}"
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.set_setting("default_topics", value)
    return {"ok": True}


@app.post("/api/newsletter/send")
async def newsletter_send_now():
    """Trigger a one-off newsletter send right now (uses current default topics
    and SMTP settings). Returns the outcome synchronously so the UI can show it."""
    from news_agent.pipeline.newsletter import send_newsletter_now
    try:
        result = await send_newsletter_now()
        return {"ok": True, **result}
    except Exception as e:
        logger.exception("Newsletter send failed")
        return {"ok": False, "error": str(e)}


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
