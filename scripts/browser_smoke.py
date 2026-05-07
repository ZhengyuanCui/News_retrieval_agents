from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boot the app against seeded data, verify the homepage renders, and save a screenshot."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root to run against.")
    parser.add_argument("--output", required=True, help="PNG path for the captured screenshot.")
    parser.add_argument("--url-path", default="/", help="Relative path to open in the browser.")
    parser.add_argument("--startup-timeout", type=float, default=20.0, help="Seconds to wait for the app to boot.")
    return parser.parse_args()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen[bytes], deadline_s: float) -> None:
    import httpx

    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if proc.poll() is not None:
            output = b""
            if proc.stdout is not None:
                try:
                    output = proc.stdout.read()
                except Exception:
                    pass
            raise RuntimeError(
                f"server exited during startup (exit={proc.returncode}):\n"
                f"{output.decode('utf-8', 'replace')}"
            )
        try:
            response = httpx.get(f"{base_url}/api/stats", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server did not become healthy within {deadline_s}s")


async def _capture(base_url: str, output_path: Path, url_path: str) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1600})
        try:
            await page.goto(f"{base_url}{url_path}", wait_until="networkidle")
            title = await page.title()
            if "News Digest" not in title:
                raise RuntimeError(f"unexpected page title: {title!r}")
            panel_count = await page.locator("section.panel").count()
            if panel_count < 2:
                raise RuntimeError(f"expected at least 2 panels, found {panel_count}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(output_path), full_page=True)
        finally:
            await browser.close()


async def _seed_items(database_url: str) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from news_agent.models import Base, NewsItem
    from news_agent.storage.repository import NewsRepository

    engine = create_async_engine(
        database_url,
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS news_items_fts "
                "USING fts5(id UNINDEXED, title, content, tokenize='porter unicode61')"
            )
        )

    now = datetime.utcnow()
    items = [
        NewsItem(
            source="openai",
            topic="ai",
            title="OpenAI ships a new reasoning model",
            url="https://example.com/openai-reasoning-model",
            content="OpenAI released a new reasoning model with faster tool use.",
            summary="OpenAI released a new reasoning model with faster tool use.",
            published_at=now - timedelta(hours=2),
            fetched_at=now - timedelta(hours=1),
            relevance_score=9.0,
            raw_score=0.7,
        ),
        NewsItem(
            source="anthropic",
            topic="ai",
            title="Anthropic expands Claude deployment options",
            url="https://example.com/anthropic-deployment-options",
            content="Anthropic added new deployment options for Claude customers.",
            summary="Anthropic added new deployment options for Claude customers.",
            published_at=now - timedelta(hours=4),
            fetched_at=now - timedelta(hours=1),
            relevance_score=8.2,
            raw_score=0.5,
        ),
        NewsItem(
            source="reuters",
            topic="stocks",
            title="NVIDIA shares rise after data-center demand forecast",
            url="https://example.com/nvidia-demand-forecast",
            content="NVIDIA shares rose after updated guidance on data-center demand.",
            summary="NVIDIA shares rose after updated guidance on data-center demand.",
            published_at=now - timedelta(hours=3),
            fetched_at=now - timedelta(hours=1),
            relevance_score=8.5,
            raw_score=0.6,
        ),
        NewsItem(
            source="bloomberg",
            topic="stocks",
            title="Fed officials signal caution on rate cuts",
            url="https://example.com/fed-rate-cuts",
            content="Federal Reserve officials signaled caution on the timing of rate cuts.",
            summary="Federal Reserve officials signaled caution on the timing of rate cuts.",
            published_at=now - timedelta(hours=6),
            fetched_at=now - timedelta(hours=1),
            relevance_score=7.8,
            raw_score=0.4,
        ),
    ]

    async with session_factory() as session:
        repo = NewsRepository(session)
        await repo.upsert_many(items)
        await repo.set_setting("default_topics", "ai|stocks")
        await session.commit()

    await engine.dispose()


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))

    with tempfile.TemporaryDirectory(prefix="browser-smoke-") as work_dir:
        work_path = Path(work_dir)
        db_path = work_path / "news.db"
        audio_dir = work_path / "audio"
        audio_dir.mkdir()
        database_url = f"sqlite+aiosqlite:///{db_path}"

        asyncio.run(_seed_items(database_url))

        env = {
            **os.environ,
            "DATABASE_URL": database_url,
            "NEWS_AGENT_TEST_MODE": "1",
            "NEWSLETTER_AUDIO_DIR": str(audio_dir),
            "TWITTER_ENABLED": "false",
            "REDDIT_ENABLED": "false",
            "YOUTUBE_ENABLED": "false",
            "GITHUB_ENABLED": "false",
            "LLM_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
        }

        port = _free_port()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "news_agent.web.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_for_health(base_url, proc, deadline_s=args.startup_timeout)
            asyncio.run(_capture(base_url, output_path, args.url_path))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
