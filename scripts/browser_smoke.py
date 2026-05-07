from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
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


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    sys.path.insert(0, str(repo_root))
    from tests.e2e.fixtures import seed_items

    with tempfile.TemporaryDirectory(prefix="browser-smoke-") as work_dir:
        work_path = Path(work_dir)
        db_path = work_path / "news.db"
        audio_dir = work_path / "audio"
        audio_dir.mkdir()
        database_url = f"sqlite+aiosqlite:///{db_path}"

        asyncio.run(seed_items(database_url))

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
