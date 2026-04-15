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
