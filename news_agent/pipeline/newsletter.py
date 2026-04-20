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
.audio-note { background: #fff7e6; border: 1px solid #ffd48a; padding: 12px 16px;
              border-radius: 4px; margin: 20px 28px; font-size: 13px; color: #6b4f00; }
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


def _render_topic_section(topic: str, digest: dict | None, items: list[NewsItem]) -> str:
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
      {digest_html}
      {items_html}
    </div>
    """


def _render_email_html(
    *,
    date_str: str,
    sections: list[str],
    has_audio: bool,
) -> str:
    audio_note = (
        '<div class="audio-note">'
        "&#127911; An audio briefing is attached to this email. "
        "Open the MP3 attachment to listen to today's summary."
        "</div>"
        if has_audio else ""
    )
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_EMAIL_CSS}</style></head>
<body>
  <div class="wrapper">
    <div class="header">
      <h1>News Digest</h1>
      <span class="date">{html.escape(date_str)}</span>
    </div>
    {audio_note}
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


def _generate_combined_audio(
    topic_items: list[tuple[str, list[NewsItem]]],
    output_path: Path,
) -> Path | None:
    """Build a single MP3 covering every topic. Returns path or None on failure."""
    gen = PodcastGenerator()
    primary_topic = topic_items[0][0] if topic_items else "news"
    combined: list[NewsItem] = []
    for _, items in topic_items:
        combined.extend(items)
    if not combined:
        return None
    try:
        return gen.generate(combined, primary_topic, output_path)
    except Exception as e:
        logger.error("Audio generation failed: %s", e, exc_info=True)
        return None


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

    # Audio (optional; synchronous so run in a thread)
    audio_bytes: bytes | None = None
    audio_filename = f"news-briefing-{datetime.utcnow():%Y-%m-%d}.mp3"
    if include_audio and total_items > 0 and (
        settings.llm_api_key or settings.anthropic_api_key
    ):
        topic_items = [(t, items) for t, _, items in parts if items]
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(
                None, _generate_combined_audio, topic_items, tmp_path,
            )
            if result_path and tmp_path.exists():
                audio_bytes = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    date_str = datetime.utcnow().strftime("%B %d, %Y")
    sections = [_render_topic_section(t, d, items) for t, d, items in parts]
    html_body = _render_email_html(
        date_str=date_str, sections=sections, has_audio=audio_bytes is not None,
    )
    text_body = _render_email_text(date_str, parts)

    attachments: list[tuple[str, bytes, str]] = []
    if audio_bytes:
        attachments.append((audio_filename, audio_bytes, "audio/mpeg"))

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
        "audio_included": audio_bytes is not None,
        "refreshed": refresh,
        "fetch_results": fetch_results,
        "sent_at": datetime.utcnow().isoformat(),
    }
    logger.info("Newsletter sent: %s", summary)
    return summary


async def send_newsletter_now() -> dict:
    """Thin wrapper used by the /api/newsletter/send endpoint."""
    return await build_and_send_newsletter()
