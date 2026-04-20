from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from news_agent.collectors import get_enabled_collectors
from news_agent.config import settings
from news_agent.models import NewsItem
from news_agent.pipeline import Aggregator, LLMAnalyzer, Deduplicator
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
    from news_agent.pipeline.vector_search import invalidate_index, _index
    invalidate_index()
    asyncio.create_task(_index.ensure_fresh())  # pre-warm in background, don't block UI
    logger.info("Items stored — UI will show results now")

    # 4. LLM analysis + digest in background (non-blocking)
    if settings.llm_api_key or settings.anthropic_api_key:
        asyncio.create_task(_analyze_and_digest(deduped_items, fetched_topics))
    else:
        logger.warning("No LLM API key set — skipping analysis (set LLM_API_KEY or ANTHROPIC_API_KEY)")

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
    """Score items with the LLM in the background (summaries, relevance, tags, sentiment).
    Digest generation is handled on-demand by the /api/digest-stream SSE endpoint so it
    doesn't compete with batch analysis for the token-rate-limit budget.
    After scoring the current fetch batch, kicks off a backfill pass for any older
    DB items that still have no summary."""
    try:
        # Brief pause so any in-flight digest stream request gets its LLM call in first.
        await asyncio.sleep(2)
        analyzer = LLMAnalyzer()

        # Apply preference boosts
        from news_agent.preference import get_preference_scores, apply_preference_boost
        async with get_session() as session:
            prefs = await get_preference_scores(session)
        if prefs:
            deduped_items = apply_preference_boost(deduped_items, prefs)

        # Analyze new items with full concurrency — these are highest priority.
        # Each topic's batches run in parallel internally via the semaphore in analyze_batch.
        async def _analyze_topic(topic: str) -> None:
            topic_items = [i for i in deduped_items if i.topic == topic and not i.is_duplicate]
            if not topic_items:
                return
            try:
                analyzed = await analyzer.analyze_batch(topic_items, topic)
                async with get_session() as session:
                    repo = NewsRepository(session)
                    await repo.update_analysis_many(analyzed)
                saved = sum(1 for i in analyzed if i.summary)
                logger.info("Analysis complete for topic '%s': %d/%d items summarized",
                            topic, saved, len(topic_items))
            except Exception as e:
                logger.error("Analysis failed for topic '%s': %s", topic, e)

        await asyncio.gather(*[_analyze_topic(t) for t in topics])

        # After the current batch is done, backfill any older items without summaries
        asyncio.create_task(_backfill_unanalyzed())

    except Exception as e:
        logger.error("Background analysis failed: %s", e)


# Guard to prevent multiple concurrent backfill loops
_backfill_running = False


async def _backfill_unanalyzed(batch_size: int = 200) -> None:
    """Continuously analyze DB items that have no summary, newest-fetched first.

    Runs in the background after each fetch cycle. Items are processed in order
    of `fetched_at DESC` so the most recently retrieved articles get summaries
    before older ones. Uses half the normal analysis concurrency to leave room
    for higher-priority current-batch analysis or digest streams.
    """
    global _backfill_running
    if _backfill_running:
        return
    _backfill_running = True
    try:
        from news_agent.config import settings as _s
        # Use half the configured concurrency so backfill doesn't starve digest/Q&A
        backfill_concurrency = max(1, _s.analysis_concurrency // 2)
        analyzer = LLMAnalyzer()

        while True:
            async with get_session() as session:
                repo = NewsRepository(session)
                items = await repo.get_unanalyzed(limit=batch_size)

            if not items:
                logger.info("Backfill complete — all items have summaries")
                break

            logger.info("Backfill: processing %d items (ordered by fetch recency)", len(items))

            from collections import defaultdict
            by_topic: dict[str, list[NewsItem]] = defaultdict(list)
            for item in items:
                by_topic[item.topic].append(item)

            # Patch the concurrency on the shared analyzer for backfill turns
            original_concurrency = _s.analysis_concurrency
            _s.analysis_concurrency = backfill_concurrency

            async def _backfill_topic(topic: str, topic_items: list[NewsItem]) -> None:
                try:
                    analyzed = await analyzer.analyze_batch(topic_items, topic)
                    async with get_session() as session:
                        repo = NewsRepository(session)
                        await repo.update_analysis_many(analyzed)
                    saved = sum(1 for i in analyzed if i.summary)
                    logger.info("Backfill: '%s' — %d/%d summarized", topic, saved, len(topic_items))
                except Exception as e:
                    logger.error("Backfill failed for topic '%s': %s", topic, e)

            try:
                await asyncio.gather(*[_backfill_topic(t, t_items) for t, t_items in by_topic.items()])
            finally:
                _s.analysis_concurrency = original_concurrency

            if len(items) < batch_size:
                break

            # Brief pause between rounds to avoid sustained API pressure
            await asyncio.sleep(2)

    except Exception as e:
        logger.error("Backfill loop failed: %s", e)
    finally:
        _backfill_running = False


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

    # Load existing DB items for this keyword so we don't re-store near-duplicates
    # that were already fetched in a previous cycle (e.g. same article via new Bing redirect URLs).
    async with get_session() as session:
        repo = NewsRepository(session)
        existing_items = await repo.get_recent(topic=keyword, hours=168, include_duplicates=False, limit=300)
    existing_ids = {i.id for i in existing_items}

    # Dedup against existing + new combined; existing items come first so they are always
    # treated as canonical (never marked duplicate in favour of a newly-fetched copy).
    deduplicator = Deduplicator()
    loop = asyncio.get_event_loop()
    deduped_all = await loop.run_in_executor(None, deduplicator.deduplicate, existing_items + raw_items)

    # Only persist items that are new to the DB (not previously stored).
    # Items already in the DB keep their existing is_duplicate flag untouched.
    deduped = [i for i in deduped_all if i.id not in existing_ids]
    unique = [i for i in deduped if not i.is_duplicate]
    logger.info("keyword=%r after dedup: %d new unique / %d new total (vs %d existing)",
                keyword, len(unique), len(deduped), len(existing_items))

    # Clear stale analysis so items are re-scored with the correct topic-specific prompt
    for item in deduped:
        item.summary = None
        item.relevance_score = None

    # Store immediately
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many(deduped)

    # Analyze in background (also triggers a backfill pass for older unanalyzed items)
    if settings.llm_api_key or settings.anthropic_api_key:
        asyncio.create_task(_analyze_and_digest(deduped, [keyword]))

    from news_agent.pipeline.vector_search import invalidate_index, _index
    invalidate_index()
    asyncio.create_task(_index.ensure_fresh())  # pre-warm in background
    logger.info("Keyword fetch for %r stored %d items (%d unique)", keyword, len(deduped), len(unique))
    return {"items_stored": len(unique)}


async def fetch_and_analyze_topic(keyword: str) -> dict:
    """Fetch news for `keyword` and wait for LLM analysis to complete.

    Unlike `run_keyword_fetch`, this does NOT return until every newly-stored
    item has been through the analyzer (summaries, relevance, tags, sentiment
    populated). Used by the newsletter job so the email always contains
    analyzed content rather than raw placeholders.

    Returns the same summary dict as `run_keyword_fetch`, plus an `analyzed`
    count.
    """
    from news_agent.collectors import get_enabled_collectors

    collectors = get_enabled_collectors(topics=[keyword])
    if not collectors:
        logger.warning("No enabled collectors for fetch-and-analyze of %r", keyword)
        return {"items_stored": 0, "analyzed": 0}

    results = await asyncio.gather(
        *[c.fetch_keyword(keyword) for c in collectors],
        return_exceptions=True,
    )
    from news_agent.collectors.base import BaseCollector as _BC
    raw_items: list[NewsItem] = []
    for collector, r in zip(collectors, results):
        if isinstance(r, Exception):
            logger.error("Collector error (%s): %s", collector.source_name, r)
        else:
            raw_items.extend(_BC.tag_languages(r))

    if not raw_items:
        logger.info("fetch_and_analyze_topic(%r): no raw items", keyword)
        return {"items_stored": 0, "analyzed": 0}

    async with get_session() as session:
        repo = NewsRepository(session)
        existing_items = await repo.get_recent(
            topic=keyword, hours=168, include_duplicates=False, limit=300,
        )
    existing_ids = {i.id for i in existing_items}

    deduplicator = Deduplicator()
    loop = asyncio.get_event_loop()
    deduped_all = await loop.run_in_executor(
        None, deduplicator.deduplicate, existing_items + raw_items,
    )
    deduped = [i for i in deduped_all if i.id not in existing_ids]
    unique = [i for i in deduped if not i.is_duplicate]

    # Clear stale analysis so items are re-scored with the topic-specific prompt
    for item in deduped:
        item.summary = None
        item.relevance_score = None

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many(deduped)

    from news_agent.pipeline.vector_search import invalidate_index
    invalidate_index()

    analyzed_count = 0
    if deduped and (settings.llm_api_key or settings.anthropic_api_key):
        # Synchronous analysis — block until summaries/tags/sentiment are written
        await _analyze_and_digest(deduped, [keyword])
        async with get_session() as session:
            repo = NewsRepository(session)
            analyzed_count = len(
                [i for i in await repo.get_recent(topic=keyword, hours=168, limit=300)
                 if i.summary]
            )

    logger.info(
        "fetch_and_analyze_topic(%r): stored=%d unique=%d analyzed=%d",
        keyword, len(deduped), len(unique), analyzed_count,
    )
    return {
        "items_stored": len(deduped),
        "unique_items": len(unique),
        "analyzed": analyzed_count,
    }


async def fetch_and_analyze_topics(topics: list[str]) -> dict[str, dict]:
    """Run `fetch_and_analyze_topic` for each topic, sequentially.

    Sequential (not parallel) so each topic's analysis batch gets full use of
    the LLM rate-limit budget rather than fighting for tokens with the others.
    """
    results: dict[str, dict] = {}
    for topic in topics:
        try:
            results[topic] = await fetch_and_analyze_topic(topic)
        except Exception as e:
            logger.error("fetch_and_analyze_topic(%r) failed: %s", topic, e, exc_info=True)
            results[topic] = {"items_stored": 0, "analyzed": 0, "error": str(e)}
    return results


async def generate_digest(topic: str, hours: int = 24) -> tuple[str, list[NewsItem]]:
    """Generate a Claude digest for a topic from recent DB items."""
    async with get_session() as session:
        repo = NewsRepository(session)
        items = await repo.get_recent(hours=hours, topic=topic)

    if not (settings.llm_api_key or settings.anthropic_api_key):
        return "No LLM API key configured — cannot generate digest.", items

    analyzer = LLMAnalyzer()
    digest_text = await analyzer.generate_digest(items, topic)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_digest(date_str, topic, digest_text, len(items))

    return digest_text, items
