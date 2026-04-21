"""Daily newsletter builder and sender.

Flow:
  1. Load default topics (NEWSLETTER_TOPICS override, else the UI-saved pair in
     the user_settings table).
  2. For each topic, fetch recent items from the DB (same query the web UI uses).
  3. Generate (or reuse today's) digest summary per topic.
  4. Render one HTML email containing both summaries + item lists.
  5. Generate a combined MP3 narration covering all topics.
  6. Email it via SMTP.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path

from news_agent.config import settings
from news_agent.emailer import EmailError, send_email
from news_agent.models import NewsItem
from news_agent.orchestrator import fetch_and_analyze_topics, generate_digest
from news_agent.pipeline.podcast import PodcastGenerator
from news_agent.storage import NewsRepository, get_session

logger = logging.getLogger(__name__)


# ── Topic resolution ─────────────────────────────────────────────────────────

async def get_default_topics() -> list[str]:
    """Return the list of topics the newsletter should cover.

    Priority: NEWSLETTER_TOPICS env var > UI-saved default_topics setting.
    Deduped, whitespace-trimmed, empties removed, order preserved.
    """
    if settings.newsletter_topics:
        seen: set[str] = set()
        out: list[str] = []
        for t in settings.newsletter_topics:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    async with get_session() as session:
        repo = NewsRepository(session)
        raw = await repo.get_setting("default_topics", "")
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return parts


# ── HTML rendering ───────────────────────────────────────────────────────────

_EMAIL_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f5f7; color: #1d1d1f; margin: 0; padding: 0; }
.wrapper { max-width: 680px; margin: 0 auto; background: #ffffff; }
.header { background: #1d1d1f; color: #fff; padding: 24px 28px; }
.header h1 { margin: 0; font-size: 22px; font-weight: 600; }
.header .date { color: #a1a1a6; font-size: 13px; margin-top: 4px; display: block; }
.topic-section { padding: 24px 28px; border-bottom: 1px solid #e5e5ea; }
.topic-title { margin: 0 0 14px 0; font-size: 18px; font-weight: 600; color: #1d1d1f; }
.digest-summary { background: #f5f5f7; border-left: 3px solid #0071e3; padding: 14px 16px;
                  border-radius: 4px; margin-bottom: 20px; }
.digest-summary .headline { font-weight: 600; margin-bottom: 8px; }
.digest-summary ul { margin: 8px 0 0 0; padding-left: 18px; }
.digest-summary li { margin-bottom: 6px; font-size: 14px; line-height: 1.5; }
.news-item { padding: 12px 0; border-bottom: 1px solid #f0f0f2; }
.news-item:last-child { border-bottom: none; }
.news-item .meta { font-size: 11px; color: #86868b; text-transform: uppercase;
                   letter-spacing: 0.4px; margin-bottom: 4px; }
.news-item .meta .source { background: #0071e3; color: #fff; padding: 2px 6px;
                           border-radius: 3px; margin-right: 6px; }
.news-item .meta .sentiment-positive { color: #0a7a30; }
.news-item .meta .sentiment-negative { color: #c0392b; }
.news-item a.title { color: #1d1d1f; text-decoration: none; font-weight: 600;
                     font-size: 15px; display: block; margin-bottom: 4px; }
.news-item a.title:hover { color: #0071e3; }
.news-item .summary { font-size: 13px; color: #3a3a3c; line-height: 1.5; margin: 6px 0 0 0; }
.news-item .tags { margin-top: 6px; }
.news-item .tag { display: inline-block; background: #f0f0f2; color: #3a3a3c;
                  font-size: 11px; padding: 2px 7px; border-radius: 10px; margin-right: 4px; }
.topic-audio { margin: 0 0 18px 0; padding: 12px 14px; background: #eef6ff;
               border: 1px solid #cfe4ff; border-radius: 6px;
               font-size: 13px; color: #0a3d7a; }
.topic-audio .label { font-weight: 600; }
.topic-audio audio { display: block; margin-top: 8px; }
.topic-audio a.listen { display: inline-block; margin-top: 6px; padding: 6px 12px;
                        background: #0071e3; color: #fff !important; text-decoration: none;
                        border-radius: 4px; font-weight: 600; font-size: 13px; }
.footer { padding: 18px 28px; color: #86868b; font-size: 12px; text-align: center; }
"""


def _strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_digest(raw: str | None) -> dict | None:
    if not raw:
        return None
    if "|||" in raw:
        parts = raw.split("|||")
        return {"headline": parts[0], "bullets": [b for b in parts[1:] if b.strip()]}
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return None
    return {"headline": lines[0], "bullets": lines[1:]}


def _render_bullet(text: str) -> str:
    """Convert **bold** markdown to <strong> (mirrors the web UI's bold_md filter)."""
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _render_topic_audio(
    audio_filename: str | None,
    content_id: str | None,
    audio_url: str | None = None,
    player_url: str | None = None,
) -> str:
    """Render the per-topic audio player.

    Preference order:
      1. A public ``audio_url`` (streams raw MP3 with
         ``Content-Disposition: inline``).  The embedded <audio> element
         fetches this URL.
      2. A ``player_url`` (e.g. /newsletter/player/<file>) that the
         "Play briefing in browser" link points at.  We never link
         directly to the .mp3 URL because many browsers / extensions
         auto-download any URL whose path ends in ``.mp3``.  The player
         page wraps the same audio stream in an <audio controls> element
         so clicking the link opens a tab that plays instead of saves.
      3. As a final fallback when no public URL is configured we embed a
         ``cid:`` reference to the attached MP3.  Apple Mail and some
         other clients will render <audio> from a cid: source; Gmail and
         Outlook unfortunately still force a download, which is a client
         limitation we cannot work around.
    """
    if audio_url:
        url = html.escape(audio_url)
        link_url = html.escape(player_url or audio_url)
        return f"""
    <div class="topic-audio">
      <div class="label">&#127911; Listen to this topic</div>
      <audio controls preload="none" src="{url}" style="width:100%;max-width:420px;margin-top:8px;">
      </audio>
      <div style="margin-top:8px;font-size:12px;">
        <a class="listen" href="{link_url}" target="_blank" rel="noopener">&#9654; Play briefing in browser</a>
      </div>
    </div>
    """
    if audio_filename and content_id:
        cid = html.escape(content_id)
        return f"""
    <div class="topic-audio">
      <div class="label">&#127911; Listen to this topic</div>
      <audio controls preload="none" src="cid:{cid}" style="width:100%;max-width:420px;margin-top:8px;">
        <a class="listen" href="cid:{cid}">&#9654; Play briefing</a>
      </audio>
    </div>
    """
    return ""


def _render_topic_section(
    topic: str,
    digest: dict | None,
    items: list[NewsItem],
    *,
    audio_filename: str | None = None,
    audio_content_id: str | None = None,
    audio_url: str | None = None,
    player_url: str | None = None,
) -> str:
    topic_label = html.escape(topic.title())

    if digest:
        bullets_html = "".join(f"<li>{_render_bullet(b)}</li>" for b in digest.get("bullets", []))
        digest_html = f"""
        <div class="digest-summary">
          <div class="headline">{_render_bullet(digest.get('headline', ''))}</div>
          {f'<ul>{bullets_html}</ul>' if bullets_html else ''}
        </div>
        """
    else:
        digest_html = (
            '<div class="digest-summary"><div class="headline">'
            "No summary available for this period yet.</div></div>"
        )

    audio_html = _render_topic_audio(
        audio_filename, audio_content_id, audio_url, player_url,
    )

    items_html_parts: list[str] = []
    for item in items:
        summary_src = item.summary or item.content[:300]
        summary = html.escape(_strip_html(summary_src))
        if not item.summary and len(item.content) > 300:
            summary += "…"

        sentiment_html = ""
        if item.sentiment:
            sent = html.escape(item.sentiment)
            sentiment_html = f'<span class="sentiment-{sent}">{sent}</span>'

        tags_html = ""
        if item.tags:
            tags_html = '<div class="tags">' + "".join(
                f'<span class="tag">{html.escape(t)}</span>' for t in item.tags[:5]
            ) + "</div>"

        published = item.published_at.strftime("%b %d, %H:%M UTC") if item.published_at else ""

        items_html_parts.append(f"""
        <div class="news-item">
          <div class="meta">
            <span class="source">{html.escape(item.source)}</span>
            {sentiment_html}
            <span>{html.escape(published)}</span>
          </div>
          <a class="title" href="{html.escape(item.url)}">{html.escape(item.title)}</a>
          <p class="summary">{summary}</p>
          {tags_html}
        </div>
        """)

    if not items_html_parts:
        items_html = '<p style="color:#86868b;font-size:13px">No recent items.</p>'
    else:
        items_html = "\n".join(items_html_parts)

    return f"""
    <div class="topic-section">
      <h2 class="topic-title">{topic_label} &middot; {len(items)} items</h2>
      {audio_html}
      {digest_html}
      {items_html}
    </div>
    """


def _render_email_html(
    *,
    date_str: str,
    sections: list[str],
    has_audio: bool = False,
    audio_topic_count: int = 0,
) -> str:
    # The per-topic sections render their own inline audio player, so no
    # top-of-email note is needed.  `has_audio` / `audio_topic_count` are
    # kept for backwards compatibility with callers and tests.
    del has_audio, audio_topic_count
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_EMAIL_CSS}</style></head>
<body>
  <div class="wrapper">
    <div class="header">
      <h1>News Digest</h1>
      <span class="date">{html.escape(date_str)}</span>
    </div>
    {body}
    <div class="footer">
      Generated by your News Retrieval Agent &middot; {html.escape(date_str)}
    </div>
  </div>
</body></html>
"""


def _render_email_text(date_str: str, parts: list[tuple[str, dict | None, list[NewsItem]]]) -> str:
    """Plain-text fallback body."""
    lines = [f"News Digest — {date_str}", "=" * 60, ""]
    for topic, digest, items in parts:
        lines.append(f"## {topic.upper()} ({len(items)} items)")
        if digest:
            lines.append(digest.get("headline", ""))
            for b in digest.get("bullets", []):
                lines.append(f"  • {re.sub(r'\\*\\*(.+?)\\*\\*', r'\\1', b)}")
        lines.append("")
        for item in items:
            lines.append(f"- [{item.source}] {item.title}")
            lines.append(f"  {item.url}")
            if item.summary:
                lines.append(f"  {_strip_html(item.summary)[:240]}")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


# ── Assembly ─────────────────────────────────────────────────────────────────

async def _gather_topic(topic: str, hours: int) -> tuple[dict | None, list[NewsItem]]:
    """Return (parsed_digest, recent_items) for one topic."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        repo = NewsRepository(session)
        items = await repo.get_recent(hours=hours, topic=topic, limit=30)
        stored = await repo.get_digest(today, topic.lower())

    digest_raw = stored.content if stored else None
    if not digest_raw and items and (settings.llm_api_key or settings.anthropic_api_key):
        try:
            digest_raw, _ = await generate_digest(topic, hours=hours)
        except Exception as e:
            logger.warning("Could not generate digest for '%s': %s", topic, e)

    return _parse_digest(digest_raw), items


def _generate_topic_audio(
    topic: str,
    items: list[NewsItem],
    output_path: Path,
) -> Path | None:
    """Build an MP3 briefing for a single topic. Returns path or None on failure."""
    if not items:
        return None
    gen = PodcastGenerator()
    try:
        return gen.generate(items, topic, output_path)
    except Exception as e:
        logger.error("Audio generation for topic '%s' failed: %s", topic, e, exc_info=True)
        return None


def _topic_slug(topic: str) -> str:
    """Make a topic string safe for filenames and Content-ID values."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", topic.strip().lower()).strip("-")
    return slug or "topic"


async def build_and_send_newsletter(
    *,
    topics: list[str] | None = None,
    hours: int | None = None,
    recipient: str | None = None,
    include_audio: bool | None = None,
    refresh: bool | None = None,
) -> dict:
    """Build and send the daily newsletter. Returns a summary dict.

    By default this first runs a live fetch for every configured topic and
    waits for LLM analysis to finish before composing the email, so the
    newsletter always reflects fresh, analyzed content.  Pass `refresh=False`
    to skip the fetch (useful in tests or when you want to send whatever is
    already in the DB).

    Raises EmailError if SMTP isn't configured or the send fails.
    """
    hours = hours if hours is not None else settings.newsletter_hours_lookback
    include_audio = (
        include_audio if include_audio is not None else settings.newsletter_include_audio
    )
    refresh = refresh if refresh is not None else True
    to_addr = recipient or settings.newsletter_email_to
    if not to_addr:
        raise EmailError(
            "No recipient configured — set NEWSLETTER_EMAIL_TO in .env "
            "or pass recipient= explicitly."
        )

    resolved_topics = topics if topics is not None else await get_default_topics()
    if not resolved_topics:
        async with get_session() as session:
            repo = NewsRepository(session)
            saved = await repo.get_setting("default_topics", "")
        raise EmailError(
            "No topics configured. Checked in order:\n"
            f"  1. --topics flag           → {topics!r}\n"
            f"  2. NEWSLETTER_TOPICS env   → {settings.newsletter_topics!r}\n"
            f"  3. UI-saved default_topics → {saved!r}\n"
            "Fix: pass --topics 'ai,stocks', or set NEWSLETTER_TOPICS in .env, "
            "or click ⚙ → Save & Apply in the web UI with two topics filled in."
        )

    logger.info("Building newsletter for topics=%s (last %dh, refresh=%s)",
                resolved_topics, hours, refresh)

    fetch_results: dict[str, dict] = {}
    if refresh:
        logger.info("Newsletter: fetching fresh items for %s before sending…",
                    resolved_topics)
        try:
            fetch_results = await fetch_and_analyze_topics(resolved_topics)
            logger.info("Newsletter: fetch complete — %s", fetch_results)
        except Exception as e:
            # If the pre-fetch fails we still try to send whatever is in the DB.
            logger.error("Newsletter pre-fetch failed (%s) — continuing with "
                         "cached DB items", e, exc_info=True)

    gathered = await asyncio.gather(*[_gather_topic(t, hours) for t in resolved_topics])
    parts: list[tuple[str, dict | None, list[NewsItem]]] = [
        (topic, digest, items) for topic, (digest, items) in zip(resolved_topics, gathered)
    ]

    total_items = sum(len(items) for _, _, items in parts)
    if total_items == 0:
        logger.warning("Newsletter has 0 items across all topics — sending anyway")

    # Audio — one MP3 per topic (synchronous generation, run in threads in parallel)
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    topic_audio: dict[str, dict] = {}  # topic -> {filename, content_id, bytes}
    can_generate_audio = include_audio and (
        settings.llm_api_key or settings.anthropic_api_key
    )

    # Resolve the directory where MP3s are persisted so the web app can
    # stream them.  We write each briefing to disk here regardless of
    # whether a public URL is configured — it's cheap, makes debugging
    # easier, and lets the app start serving the URL retroactively.
    audio_dir = Path(settings.newsletter_audio_dir)
    if not audio_dir.is_absolute():
        # Resolve relative paths against the project root (three levels up
        # from this file: news_agent/pipeline/newsletter.py → project root).
        project_root = Path(__file__).resolve().parents[2]
        audio_dir = project_root / audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    public_base = (settings.public_base_url or "").rstrip("/")

    async def _build_audio_for(topic: str, items: list[NewsItem]) -> None:
        if not items:
            return
        slug = _topic_slug(topic)
        filename = f"{slug}-briefing-{date_tag}.mp3"
        final_path = audio_dir / filename
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(
                None, _generate_topic_audio, topic, items, tmp_path,
            )
            if result_path and tmp_path.exists():
                data = tmp_path.read_bytes()
                final_path.write_bytes(data)
                # The player slug strips the ``.mp3`` suffix from the URL
                # path.  Browsers / extensions that auto-download any URL
                # ending in ``.mp3`` won't match this one — the player
                # endpoint adds the extension back internally when it
                # looks up the file on disk.
                player_slug = filename.removesuffix(".mp3")
                topic_audio[topic] = {
                    "filename": filename,
                    "content_id": f"audio-{slug}-{date_tag}",
                    "bytes": data,
                    "path": str(final_path),
                    "url": f"{public_base}/newsletter/audio/{filename}" if public_base else None,
                    "player_url": f"{public_base}/newsletter/player/{player_slug}" if public_base else None,
                }
        finally:
            tmp_path.unlink(missing_ok=True)

    if can_generate_audio:
        await asyncio.gather(*[
            _build_audio_for(t, items) for t, _, items in parts if items
        ])

    date_str = datetime.utcnow().strftime("%B %d, %Y")
    sections: list[str] = []
    for t, d, items in parts:
        info = topic_audio.get(t)
        sections.append(_render_topic_section(
            t, d, items,
            audio_filename=info["filename"] if info else None,
            audio_content_id=info["content_id"] if info else None,
            audio_url=info["url"] if info else None,
            player_url=info.get("player_url") if info else None,
        ))
    html_body = _render_email_html(
        date_str=date_str,
        sections=sections,
        has_audio=bool(topic_audio),
        audio_topic_count=len(topic_audio),
    )
    text_body = _render_email_text(date_str, parts)

    attachments: list[tuple] = []
    if settings.newsletter_attach_audio:
        for t, _, _items in parts:
            info = topic_audio.get(t)
            if info:
                attachments.append((
                    info["filename"], info["bytes"], "audio/mpeg", info["content_id"],
                ))

    subject = f"News Digest — {date_str} ({', '.join(t.title() for t in resolved_topics)})"
    send_email(
        to=to_addr,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        attachments=attachments,
    )

    summary = {
        "recipient": to_addr,
        "topics": resolved_topics,
        "items": total_items,
        "audio_included": bool(topic_audio),
        "audio_topics": sorted(topic_audio.keys()),
        "audio_files": [info["filename"] for info in topic_audio.values()],
        "audio_urls": [info["url"] for info in topic_audio.values() if info.get("url")],
        "refreshed": refresh,
        "fetch_results": fetch_results,
        "sent_at": datetime.utcnow().isoformat(),
    }
    logger.info("Newsletter sent: %s", summary)
    return summary


async def send_newsletter_now() -> dict:
    """Thin wrapper used by the /api/newsletter/send endpoint."""
    return await build_and_send_newsletter()


# ── Preview / formatting test ────────────────────────────────────────────────

# A valid ~1 KB silent MPEG-1 Layer III frame.  Used by the preview command
# so we can exercise the `<audio>` player end-to-end without running TTS.
# Base64-encoded; decoded on use.
_SILENT_MP3_B64 = (
    # Minimal MP3: ID3v2 header + one silent MPEG1 Layer3 frame (~26 ms).
    # Browsers accept this as a playable audio/mpeg resource.
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQwAAAAAAA"
    "AAAAAAAAAAAAAAAASW5mbwAAAA8AAAACAAAEQAB/f39/f39/f39/f39/f39/f39/f39/f39/"
    "f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/"
    "f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/f39/AAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAA//syxAADwAABpAAAACAAADSAAAAETEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVV//syxEADwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVV//syxIADwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVV"
)


def _dummy_silent_mp3_bytes() -> bytes:
    """Return a small, valid MP3 payload used for formatting previews.

    The file is real audio (a short silent frame) so every email client and
    browser's `<audio>` element treats it as playable — it just produces
    silence.
    """
    import base64
    return base64.b64decode(_SILENT_MP3_B64)


def _dummy_items(topic: str) -> list[NewsItem]:
    """Build a handful of obviously-fake NewsItem rows for the preview email.

    Covers the visual edge cases: bold markdown in digest, tags, sentiment,
    missing summary, long title, special characters.
    """
    now = datetime.utcnow()
    t = topic.strip() or "preview"
    return [
        NewsItem(
            source="rss",
            topic=t,
            title=f"[PREVIEW] Top story about {t.title()} — with **bold** emphasis",
            url="https://example.com/preview/1",
            content=f"Long-form body copy about {t} goes here. " * 4,
            summary=(
                f"Sample summary one for the <em>{t}</em> topic. "
                "This is dummy text used to preview newsletter formatting."
            ),
            published_at=now,
            raw_score=0.9,
            relevance_score=9.0,
            sentiment="positive",
            tags=["preview", "formatting", t],
            key_entities=["Example Corp", "Demo Inc"],
        ),
        NewsItem(
            source="reddit",
            topic=t,
            title=f"[PREVIEW] Negative-sentiment sample for {t.title()}",
            url="https://example.com/preview/2",
            content="",
            summary="Sample summary two — showcases negative sentiment styling.",
            published_at=now,
            raw_score=0.6,
            relevance_score=6.5,
            sentiment="negative",
            tags=["dummy"],
        ),
        NewsItem(
            source="github",
            topic=t,
            title=f"[PREVIEW] Neutral item for {t.title()} (no summary)",
            url="https://example.com/preview/3",
            content=(
                "This item intentionally has no summary so we can verify the "
                "fallback path that truncates the raw content instead. "
                * 5
            ),
            published_at=now,
            raw_score=0.3,
            relevance_score=4.0,
        ),
    ]


def _dummy_digest(topic: str) -> dict:
    t = topic.strip() or "preview"
    return {
        "headline": f"Preview headline for **{t.title()}**",
        "bullets": [
            f"First bullet about **{t}** with bold for emphasis.",
            "Second bullet covering a supporting point.",
            "Third bullet to test list rendering across clients.",
        ],
    }


async def build_and_send_preview_newsletter(
    *,
    topics: list[str] | None = None,
    recipient: str | None = None,
    include_audio: bool = True,
    audio_file: Path | str | None = None,
) -> dict:
    """Send a preview newsletter with dummy content + a silent MP3.

    Does NOT touch the fetch pipeline, the LLM, or the TTS engine — it's
    purely a formatting / delivery test.  Everything else (HTML rendering,
    attachment handling, public-URL embedding) is the same code path the
    real newsletter uses, so what you see here is what your users get.

    Args:
        topics: defaults to ["ai", "stocks"].  Two topics reproduces the
            multi-section layout; pass one for a single-topic view.
        recipient: override NEWSLETTER_EMAIL_TO.
        include_audio: embed the dummy audio player / attach the MP3.
        audio_file: use this MP3 instead of the built-in silent clip.
            Helpful for verifying a specific file plays in your client.

    Raises EmailError if SMTP isn't configured.
    """
    resolved_topics = topics or ["ai", "stocks"]
    to_addr = recipient or settings.newsletter_email_to
    if not to_addr:
        raise EmailError(
            "No recipient configured — set NEWSLETTER_EMAIL_TO in .env or "
            "pass recipient= explicitly."
        )

    # Build dummy items + digests per topic
    parts: list[tuple[str, dict | None, list[NewsItem]]] = [
        (t, _dummy_digest(t), _dummy_items(t)) for t in resolved_topics
    ]

    # Prepare dummy audio (persist to NEWSLETTER_AUDIO_DIR so the public
    # URL, if configured, actually resolves).
    audio_dir = Path(settings.newsletter_audio_dir)
    if not audio_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        audio_dir = project_root / audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    public_base = (settings.public_base_url or "").rstrip("/")

    if audio_file is not None:
        audio_bytes = Path(audio_file).read_bytes()
    else:
        audio_bytes = _dummy_silent_mp3_bytes()

    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    topic_audio: dict[str, dict] = {}
    if include_audio:
        for t in resolved_topics:
            slug = _topic_slug(t)
            filename = f"preview-{slug}-briefing-{date_tag}.mp3"
            final_path = audio_dir / filename
            final_path.write_bytes(audio_bytes)
            player_slug = filename.removesuffix(".mp3")
            topic_audio[t] = {
                "filename": filename,
                "content_id": f"audio-preview-{slug}-{date_tag}",
                "bytes": audio_bytes,
                "path": str(final_path),
                "url": f"{public_base}/newsletter/audio/{filename}" if public_base else None,
                "player_url": f"{public_base}/newsletter/player/{player_slug}" if public_base else None,
            }

    date_str = datetime.utcnow().strftime("%B %d, %Y")
    sections: list[str] = []
    for t, d, items in parts:
        info = topic_audio.get(t)
        sections.append(_render_topic_section(
            t, d, items,
            audio_filename=info["filename"] if info else None,
            audio_content_id=info["content_id"] if info else None,
            audio_url=info["url"] if info else None,
            player_url=info.get("player_url") if info else None,
        ))

    html_body = _render_email_html(
        date_str=date_str,
        sections=sections,
        has_audio=bool(topic_audio),
        audio_topic_count=len(topic_audio),
    )
    text_body = _render_email_text(date_str, parts)

    attachments: list[tuple] = []
    if include_audio and settings.newsletter_attach_audio:
        for t in resolved_topics:
            info = topic_audio.get(t)
            if info:
                attachments.append((
                    info["filename"], info["bytes"], "audio/mpeg", info["content_id"],
                ))

    subject = (
        f"[PREVIEW] News Digest — {date_str} "
        f"({', '.join(t.title() for t in resolved_topics)})"
    )
    send_email(
        to=to_addr,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        attachments=attachments,
    )

    return {
        "mode": "preview",
        "recipient": to_addr,
        "topics": resolved_topics,
        "items_per_topic": {t: len(items) for t, _, items in parts},
        "audio_included": bool(topic_audio),
        "audio_files": [info["filename"] for info in topic_audio.values()],
        "audio_urls": [info["url"] for info in topic_audio.values() if info.get("url")],
        "audio_attached": include_audio and settings.newsletter_attach_audio,
        "audio_source": "provided-file" if audio_file else ("silent-dummy" if include_audio else None),
        "sent_at": datetime.utcnow().isoformat(),
    }
