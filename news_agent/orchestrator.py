from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from news_agent.collectors import get_enabled_collectors
from news_agent.config import settings
from news_agent.models import NewsItem
from news_agent.pipeline import Aggregator, ClaudeAnalyzer, Deduplicator
from news_agent.storage import NewsRepository, get_session

logger = logging.getLogger(__name__)


def _is_digest_topic(topic: str, items: list[NewsItem]) -> bool:
    """Allow digests only for semantic topics, not source identifiers."""
    return any(item.topic == topic and item.source != topic for item in items)


async def run_fetch_cycle(topics: list[str] | None = None) -> dict:
    """
    Pipeline: fetch → deduplicate → store immediately → analyze in background.
    Items appear in the UI right after dedup; summaries/scores fill in shortly after.
    Pass topics=[] to fetch from all sources without topic restriction.
    """
    # None → pass None to collectors so they use their own defaults (ai + stocks)
    # []  → pass [] to collectors meaning no topic filter (fetch everything)
    start = datetime.utcnow()

    collectors = get_enabled_collectors(topics=topics)
    if not collectors:
        logger.warning("No enabled collectors found. Check your .env credentials.")
        return {"items_fetched": 0, "items_stored": 0, "duration_seconds": 0}

    # 1. Collect (all sources in parallel)
    aggregator = Aggregator(collectors)
    raw_items = await aggregator.fetch_all()
    logger.info("Aggregated %d raw items from %d collectors", len(raw_items), len(collectors))

    # 2. Deduplicate (sentence-transformers encode is CPU-bound — run in thread pool)
    deduplicator = Deduplicator()
    loop = asyncio.get_event_loop()
    deduped_items = await loop.run_in_executor(None, deduplicator.deduplicate, raw_items)
    unique_count = sum(1 for i in deduped_items if not i.is_duplicate)
    logger.info("After dedup: %d unique / %d total", unique_count, len(deduped_items))

    # 3. Store immediately so items appear in the UI without waiting for Claude
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    fetched_topics = list({
        i.topic
        for i in deduped_items
        if not i.is_duplicate and _is_digest_topic(i.topic, deduped_items)
    })
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many(deduped_items)
        from collections import Counter
        counts = Counter(i.source for i in deduped_items if not i.is_duplicate)
        for source, count in counts.items():
            await repo.update_collector_state(
                source=source,
                last_run=datetime.utcnow(),
                items_fetched=count,
            )
        # Digests are regenerated after analysis — no pre-emptive deletion needed
    logger.info("Items stored — UI will show results now")

    # 4. Claude analysis + digest in background (non-blocking)
    if settings.anthropic_api_key:
        asyncio.create_task(_analyze_and_digest(deduped_items, fetched_topics))
    else:
        logger.warning("ANTHROPIC_API_KEY not set — skipping Claude analysis")

    duration = (datetime.utcnow() - start).total_seconds()
    summary = {
        "items_fetched": len(raw_items),
        "items_stored": len(deduped_items),
        "unique_items": unique_count,
        "duration_seconds": round(duration, 1),
        "sources": list({i.source for i in deduped_items}),
    }
    logger.info("Fetch cycle complete (analysis running in background): %s", summary)
    return summary


async def _analyze_and_digest(deduped_items: list[NewsItem], topics: list[str]) -> None:
    """Run Claude analysis and digest generation after items are already stored."""
    try:
        analyzer = ClaudeAnalyzer()

        # Apply preference boosts
        from news_agent.preference import get_preference_scores, apply_preference_boost
        async with get_session() as session:
            prefs = await get_preference_scores(session)
        if prefs:
            deduped_items = apply_preference_boost(deduped_items, prefs)

        # Analyze per topic
        for topic in topics:
            topic_items = [i for i in deduped_items if i.topic == topic]
            if not topic_items:
                continue
            try:
                analyzed = await analyzer.analyze_batch(topic_items, topic)
                analyzed_map = {i.id: i for i in analyzed}
                deduped_items = [analyzed_map.get(i.id, i) for i in deduped_items]
                # Update stored items with summaries/scores
                async with get_session() as session:
                    repo = NewsRepository(session)
                    await repo.upsert_many([analyzed_map[i] for i in analyzed_map])
                logger.info("Analysis complete for topic '%s'", topic)
            except Exception as e:
                logger.error("Analysis failed for topic '%s': %s", topic, e)

        # Generate digests
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        for topic in topics:
            try:
                if not _is_digest_topic(topic, deduped_items):
                    logger.debug("Skipping digest for source-like topic '%s'", topic)
                    continue
                topic_items = [i for i in deduped_items if i.topic == topic and not i.is_duplicate]
                if topic_items:
                    digest_text = await analyzer.generate_digest(topic_items, topic)
                    async with get_session() as session:
                        repo = NewsRepository(session)
                        await repo.upsert_digest(date_str, topic, digest_text, len(topic_items))
                    logger.info("Digest generated for topic '%s'", topic)
            except Exception as e:
                logger.error("Digest generation failed for topic '%s': %s", topic, e)

    except Exception as e:
        logger.error("Background analysis failed: %s", e)


async def run_keyword_fetch(keyword: str) -> dict:
    """Fetch news from all sources specifically about `keyword` and store with topic=keyword."""
    from news_agent.collectors import get_enabled_collectors

    collectors = get_enabled_collectors(topics=[keyword])
    if not collectors:
        logger.warning("No enabled collectors for keyword fetch")
        return {"items_stored": 0}

    # Run fetch_keyword on all collectors in parallel
    import asyncio as _asyncio
    results = await _asyncio.gather(
        *[c.fetch_keyword(keyword) for c in collectors],
        return_exceptions=True,
    )
    from news_agent.collectors.base import BaseCollector as _BC
    raw_items: list[NewsItem] = []
    for collector, r in zip(collectors, results):
        if isinstance(r, Exception):
            logger.error("Keyword collector error (%s): %s", collector.source_name, r)
        else:
            tagged = _BC.tag_languages(r)
            logger.info("keyword=%r source=%s raw=%d", keyword, collector.source_name, len(tagged))
            raw_items.extend(tagged)

    if not raw_items:
        logger.warning("Keyword fetch for %r: all sources returned 0 items", keyword)
        return {"items_stored": 0}

    logger.info("keyword=%r total raw=%d — starting dedup", keyword, len(raw_items))
    deduplicator = Deduplicator()
    loop = asyncio.get_event_loop()
    deduped = await loop.run_in_executor(None, deduplicator.deduplicate, raw_items)
    unique = [i for i in deduped if not i.is_duplicate]
    logger.info("keyword=%r after dedup: %d unique / %d total", keyword, len(unique), len(deduped))

    # Clear stale analysis so items are re-scored with the correct topic-specific prompt
    for item in deduped:
        item.summary = None
        item.relevance_score = None

    # Store immediately
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many(deduped)

    # Analyze in background
    if settings.anthropic_api_key:
        asyncio.create_task(_analyze_and_digest(deduped, [keyword]))

    logger.info("Keyword fetch for %r stored %d items (%d unique)", keyword, len(deduped), len(unique))
    return {"items_stored": len(unique)}


async def generate_digest(topic: str, hours: int = 24) -> tuple[str, list[NewsItem]]:
    """Generate a Claude digest for a topic from recent DB items."""
    async with get_session() as session:
        repo = NewsRepository(session)
        items = await repo.get_recent(hours=hours, topic=topic)

    if not settings.anthropic_api_key:
        return "ANTHROPIC_API_KEY not set — cannot generate digest.", items

    analyzer = ClaudeAnalyzer()
    digest_text = await analyzer.generate_digest(items, topic)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest(date_str, topic, digest_text, len(items))

    return digest_text, items
