from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool):
    """News Retrieval Agent — AI and stock market news aggregator."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


# ── fetch ─────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--topics", default="", show_default=True, help="Comma-separated topics (empty = fetch everything)")
def fetch(topics: str):
    """Run a single fetch cycle from all enabled sources."""
    from news_agent.storage import init_db
    from news_agent.orchestrator import run_fetch_cycle

    topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    async def _run():
        await init_db()
        with console.status("[bold green]Fetching news…"):
            summary = await run_fetch_cycle(topic_list)
        rprint(f"\n[bold green]Done![/] {summary}")

    asyncio.run(_run())


# ── schedule ──────────────────────────────────────────────────────────────────

@main.command()
def schedule():
    """Start the background scheduler (runs indefinitely)."""
    from news_agent.scheduler import run_scheduler
    asyncio.run(run_scheduler())


# ── digest ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--topic", default="ai", show_default=True)
@click.option("--hours", default=24, show_default=True, help="Look back N hours")
def digest(topic: str, hours: int):
    """Generate and print a Claude-powered digest for a topic."""
    from news_agent.storage import init_db
    from news_agent.orchestrator import generate_digest

    async def _run():
        await init_db()
        with console.status(f"[bold cyan]Generating {topic} digest…"):
            text, items = await generate_digest(topic, hours)
        console.rule(f"[bold]{topic.upper()} Digest — {datetime.utcnow().strftime('%Y-%m-%d')}")
        console.print(text)
        console.print(f"\n[dim]Based on {len(items)} items from the last {hours}h[/dim]")

    asyncio.run(_run())


# ── export ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--format", "fmt", default="markdown", type=click.Choice(["json", "markdown"]), show_default=True)
@click.option("--date", default=None, help="Date to export (YYYY-MM-DD, default: today)")
@click.option("--hours", default=24, show_default=True)
def export(fmt: str, date: str | None, hours: int):
    """Export news items to JSON or Markdown."""
    from news_agent.storage import init_db, get_session, Exporter
    from news_agent.storage.repository import NewsRepository
    from news_agent.orchestrator import generate_digest

    async def _run():
        await init_db()
        date_str = date or datetime.utcnow().strftime("%Y-%m-%d")
        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.get_recent(hours=hours)

        exporter = Exporter()
        if fmt == "json":
            path = exporter.export_json(items, date_str)
        else:
            # Try to get digests from DB, generate if missing
            from news_agent.storage.repository import NewsRepository as _Repo
            async with get_session() as session:
                repo = _Repo(session)
                ai_d = await repo.get_digest(date_str, "ai")
                stocks_d = await repo.get_digest(date_str, "stocks")
            ai_text = ai_d.content if ai_d else None
            stocks_text = stocks_d.content if stocks_d else None
            path = exporter.export_markdown(items, ai_text, stocks_text, date_str)

        console.print(f"[green]Exported {len(items)} items to:[/] {path}")

    asyncio.run(_run())


# ── analyze ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--batch", default=500, show_default=True, help="Max items to analyze per run")
@click.option("--topic", default=None, help="Only analyze items for this topic")
def analyze(batch: int, topic: str | None):
    """Run LLM analysis on existing DB items that have no summary yet.

    Useful for backfilling summaries after a large fetch, or after the
    server was restarted before background analysis could complete.
    """
    from news_agent.storage import init_db, get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.pipeline.analyzer import LLMAnalyzer
    from news_agent.config import settings
    from collections import defaultdict

    async def _run():
        await init_db()
        async with get_session() as session:
            repo = NewsRepository(session)
            total_pending = await repo.count_unanalyzed()

        if total_pending == 0:
            console.print("[green]All items already have summaries — nothing to do.[/]")
            return

        console.print(f"[bold]{total_pending}[/] items pending analysis (processing up to {batch})")

        if not (settings.llm_api_key or settings.anthropic_api_key):
            console.print("[red]No LLM API key configured — set LLM_API_KEY or ANTHROPIC_API_KEY in .env[/]")
            return

        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.get_unanalyzed(limit=batch)

        if topic:
            items = [i for i in items if i.topic.lower() == topic.lower()]
            console.print(f"Filtered to {len(items)} items for topic '{topic}'")

        # Group by topic so LLM analysis prompt uses the correct topic label
        by_topic: dict[str, list] = defaultdict(list)
        for item in items:
            by_topic[item.topic].append(item)

        analyzer = LLMAnalyzer()
        total_done = 0

        for t, t_items in by_topic.items():
            with console.status(f"[cyan]Analyzing {len(t_items)} items for topic '{t}'…"):
                analyzed = await analyzer.analyze_batch(t_items, t)
                analyzed_map = {i.id: i for i in analyzed if i.summary}

            if analyzed_map:
                async with get_session() as session:
                    repo = NewsRepository(session)
                    # Write analysis fields directly (upsert won't overwrite analysis fields,
                    # so update rows individually)
                    from sqlalchemy import update
                    from news_agent.models import NewsItemORM
                    for item in analyzed_map.values():
                        await session.execute(
                            update(NewsItemORM)
                            .where(NewsItemORM.id == item.id)
                            .values(
                                summary=item.summary,
                                relevance_score=item.relevance_score,
                                key_entities=item.key_entities,
                                sentiment=item.sentiment,
                                tags=item.tags,
                            )
                        )
            total_done += len(analyzed_map)
            console.print(f"  [green]OK[/] '{t}': {len(analyzed_map)}/{len(t_items)} analyzed")

        remaining = total_pending - total_done
        console.print(
            f"\n[bold green]Done![/] Analyzed {total_done} items. "
            + (f"{remaining} still pending (run again to continue)." if remaining > batch else "")
        )

    asyncio.run(_run())


# ── refetch ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--topics", default="", show_default=True, help="Comma-separated topics (empty = fetch everything)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def refetch(topics: str, yes: bool):
    """Clear the entire database and re-fetch with current filters.

    Useful after changing spam filters or other criteria so old
    unfiltered content is removed and fresh content is fetched.
    """
    from news_agent.storage import init_db, get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.orchestrator import run_fetch_cycle

    if not yes:
        console.print("[bold yellow]This will delete ALL news items and digests from the database.[/]")
        click.confirm("Continue?", abort=True)

    topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    async def _run():
        await init_db()
        async with get_session() as session:
            repo = NewsRepository(session)
            counts = await repo.clear_all()
        console.print(
            f"[red]Deleted {counts['items']} items and {counts['digests']} digests.[/]"
        )
        with console.status("[bold green]Fetching fresh news…"):
            summary = await run_fetch_cycle(topic_list)
        rprint(f"\n[bold green]Done![/] {summary}")

    asyncio.run(_run())


# ── status ────────────────────────────────────────────────────────────────────

@main.command()
def status():
    """Show database stats and per-source status."""
    from news_agent.storage import init_db, get_session
    from news_agent.storage.repository import NewsRepository

    async def _run():
        await init_db()
        async with get_session() as session:
            repo = NewsRepository(session)
            stats = await repo.get_stats()
            states = await repo.get_all_collector_states()

        table = Table(title="Database Stats")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for k, v in stats.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)

        if states:
            state_table = Table(title="Collector Status")
            state_table.add_column("Source")
            state_table.add_column("Enabled")
            state_table.add_column("Last Run")
            state_table.add_column("Items Fetched")
            state_table.add_column("Last Error")
            for s in states:
                state_table.add_row(
                    s.source,
                    "[green]Yes[/]" if s.is_enabled else "[red]No[/]",
                    str(s.last_run)[:19] if s.last_run else "Never",
                    str(s.items_fetched),
                    (s.last_error or "")[:60],
                )
            console.print(state_table)

    asyncio.run(_run())


# ── newsletter ────────────────────────────────────────────────────────────────

@main.command()
@click.option("--topics", default="", help="Comma-separated topics (default: use saved UI topics / NEWSLETTER_TOPICS)")
@click.option("--to", "recipient", default="", help="Recipient email (default: NEWSLETTER_EMAIL_TO)")
@click.option("--hours", default=None, type=int, help="Look back N hours (default: NEWSLETTER_HOURS_LOOKBACK)")
@click.option("--no-audio", is_flag=True, help="Skip the MP3 audio attachment")
@click.option("--no-refresh", is_flag=True, help="Skip the live fetch; send using items already in the DB")
def newsletter(topics: str, recipient: str, hours: int | None, no_audio: bool, no_refresh: bool):
    """Build and email a newsletter right now (manual trigger / test).

    By default this fetches fresh news for every topic and waits for LLM
    analysis before sending. Use --no-refresh to skip the fetch and use only
    items already in the DB (much faster for SMTP testing).
    """
    from news_agent.storage import init_db
    from news_agent.pipeline.newsletter import build_and_send_newsletter

    topic_list = [t.strip() for t in topics.split(",") if t.strip()] or None

    async def _run():
        await init_db()
        label = "Building newsletter…" if no_refresh else "Fetching fresh news, analyzing, and emailing…"
        with console.status(f"[bold cyan]{label}"):
            result = await build_and_send_newsletter(
                topics=topic_list,
                recipient=recipient or None,
                hours=hours,
                include_audio=not no_audio,
                refresh=not no_refresh,
            )
        rprint(f"\n[bold green]Sent![/] {result}")

    asyncio.run(_run())


# ── newsletter-preview ────────────────────────────────────────────────────────

@main.command("newsletter-preview")
@click.option("--topics", default="ai,stocks", show_default=True,
              help="Comma-separated topics to render")
@click.option("--to", "recipient", default="", help="Recipient email (default: NEWSLETTER_EMAIL_TO)")
@click.option("--no-audio", is_flag=True, help="Skip the dummy audio attachment / player")
@click.option("--audio-file", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Use this MP3 instead of the built-in silent clip")
@click.option("--dump-html", type=click.Path(dir_okay=False), default=None,
              help="Also write the rendered HTML body to this path (handy for browser preview)")
def newsletter_preview(
    topics: str,
    recipient: str,
    no_audio: bool,
    audio_file: str | None,
    dump_html: str | None,
):
    """Send a **formatting-only** preview newsletter.

    Skips the fetch pipeline, LLM analysis, and TTS entirely. Renders the
    email with dummy content + a tiny silent MP3 (or your own --audio-file)
    and sends it via SMTP so you can verify the layout, the inline audio
    player, and the attachment behavior without waiting for a real cycle.

    Examples:

    \b
      # Minimal sanity check using defaults from .env
      news-agent newsletter-preview

    \b
      # Override recipient and attach your own MP3
      news-agent newsletter-preview --to me@example.com --audio-file briefing.mp3

    \b
      # No audio at all — just the HTML layout
      news-agent newsletter-preview --no-audio

    \b
      # Also save the HTML for browser-level inspection
      news-agent newsletter-preview --no-audio --dump-html /tmp/preview.html
    """
    from pathlib import Path as _Path
    from news_agent.pipeline.newsletter import (
        build_and_send_preview_newsletter,
        _render_topic_section,
        _render_email_html,
        _dummy_items,
        _dummy_digest,
    )

    topic_list = [t.strip() for t in topics.split(",") if t.strip()] or ["ai", "stocks"]

    # Optional: dump HTML to disk for browser inspection *before* attempting
    # SMTP so you can iterate on layout even without SMTP configured.
    if dump_html:
        sections = [
            _render_topic_section(t, _dummy_digest(t), _dummy_items(t))
            for t in topic_list
        ]
        html_body = _render_email_html(
            date_str=datetime.utcnow().strftime("%B %d, %Y"),
            sections=sections,
            has_audio=False,
            audio_topic_count=0,
        )
        _Path(dump_html).write_text(html_body, encoding="utf-8")
        console.print(f"[green]Wrote HTML preview to[/] {dump_html}")

    async def _run():
        with console.status("[bold cyan]Sending preview newsletter…"):
            result = await build_and_send_preview_newsletter(
                topics=topic_list,
                recipient=recipient or None,
                include_audio=not no_audio,
                audio_file=_Path(audio_file) if audio_file else None,
            )
        rprint(f"\n[bold green]Preview sent![/] {result}")
        if result.get("audio_urls"):
            console.print(
                "\n[dim]Tip: the email's <audio> player points at these "
                "URLs — make sure the app is reachable from the recipient's "
                "mail client:[/dim]"
            )
            for url in result["audio_urls"]:
                console.print(f"  {url}")
        elif not no_audio:
            console.print(
                "\n[yellow]PUBLIC_BASE_URL is not set.[/] The email "
                "contains the MP3 only as a `cid:` attachment, which most "
                "webmail clients (Gmail, Outlook) will show as a download "
                "instead of playing inline. Set PUBLIC_BASE_URL in .env to "
                "enable true inline playback."
            )

    asyncio.run(_run())


# ── sources ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev mode)")
def serve(host: str, port: int, reload: bool):
    """Start the web UI at http://localhost:8000"""
    import uvicorn
    console.print(f"[bold green]Starting web UI at http://{host}:{port}[/]")
    uvicorn.run("news_agent.web.app:app", host=host, port=port, reload=reload)


@main.group()
def sources():
    """Manage news sources."""
    pass


@sources.command("list")
def sources_list():
    """List all configured sources and their status."""
    from news_agent.collectors import ALL_COLLECTORS

    table = Table(title="Configured Sources")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Topics")
    for cls in ALL_COLLECTORS:
        instance = cls()
        enabled = instance.is_enabled()
        table.add_row(
            cls.source_name,
            "[green]Enabled[/]" if enabled else "[red]Disabled[/]",
            ", ".join(instance.topics),
        )
    console.print(table)


@sources.command("enable")
@click.argument("source")
def sources_enable(source: str):
    """Enable a source by name."""
    from news_agent.storage import init_db, get_session
    from news_agent.storage.repository import NewsRepository

    async def _run():
        await init_db()
        async with get_session() as session:
            repo = NewsRepository(session)
            await repo.update_collector_state(source=source, is_enabled=True)
        console.print(f"[green]{source} enabled[/]")

    asyncio.run(_run())


@sources.command("disable")
@click.argument("source")
def sources_disable(source: str):
    """Disable a source by name."""
    from news_agent.storage import init_db, get_session
    from news_agent.storage.repository import NewsRepository

    async def _run():
        await init_db()
        async with get_session() as session:
            repo = NewsRepository(session)
            await repo.update_collector_state(source=source, is_enabled=False)
        console.print(f"[yellow]{source} disabled[/]")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
