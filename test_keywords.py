"""Quick smoke-test: fetch keywords and print top results for manual review."""
from __future__ import annotations
import asyncio
import sys

KEYWORDS = ["Apple", "climate change", "GPT-5", "tariffs", "SpaceX",
            "cybersecurity", "Amazon", "nuclear energy", "Anthropic", "oil prices"]
HOURS = 48


async def main():
    from news_agent.storage import init_db, get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.orchestrator import run_keyword_fetch

    await init_db()

    for keyword in KEYWORDS:
        print(f"\n{'='*70}")
        print(f"  Fetching: {keyword}")
        print('='*70)
        result = await run_keyword_fetch(keyword)
        print(f"  Stored: {result}")

        # Wait for LLM analysis (up to 90s)
        for _ in range(18):
            await asyncio.sleep(5)
            async with get_session() as session:
                repo = NewsRepository(session)
                items = await repo.search(keyword, hours=HOURS)
            analyzed = sum(1 for i in items if i.relevance_score is not None)
            if analyzed >= min(5, len(items)):
                break

        async with get_session() as session:
            repo = NewsRepository(session)
            items = await repo.search(keyword, hours=HOURS)

        if not items:
            print("  [NO RESULTS]")
            continue

        print(f"  {len(items)} items (showing top 15):\n")
        for i, item in enumerate(items[:15], 1):
            rel = f"{item.relevance_score:.1f}" if item.relevance_score is not None else "  - "
            eng = f"{item.raw_score:.2f}"
            src = item.source.ljust(12)
            title = item.title[:75].encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding)
            print(f"  {i:2}. [{rel}/10] eng={eng} src={src} {title}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
