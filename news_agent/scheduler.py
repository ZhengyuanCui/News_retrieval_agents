from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from news_agent.config import settings

logger = logging.getLogger(__name__)


async def _fetch_job():
    from news_agent.orchestrator import run_fetch_cycle
    logger.info("Scheduled fetch job starting…")
    try:
        # topics=[] → no topic filter: fetch all high-attention content from every source
        summary = await run_fetch_cycle(topics=[])
        logger.info("Scheduled fetch complete: %s", summary)
    except Exception as e:
        logger.error("Scheduled fetch job failed: %s", e, exc_info=True)


async def _digest_job():
    from news_agent.storage import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.orchestrator import generate_digest
    from sqlalchemy import select, distinct
    from news_agent.models import NewsItemORM

    logger.info("Scheduled digest job starting…")
    try:
        # Discover all topics currently in the DB rather than hardcoding
        async with get_session() as session:
            result = await session.execute(select(distinct(NewsItemORM.topic)))
            topics = [row[0] for row in result if row[0]]
            source_result = await session.execute(select(distinct(NewsItemORM.source)))
            source_names = {row[0] for row in source_result if row[0]}

        topics = [topic for topic in topics if topic not in source_names]

        for topic in topics:
            try:
                await generate_digest(topic)
                logger.info("Digest generated for topic '%s'", topic)
            except Exception as e:
                logger.error("Digest failed for topic '%s': %s", topic, e)
    except Exception as e:
        logger.error("Digest job failed: %s", e, exc_info=True)


async def _prune_job():
    from news_agent.storage import get_session
    from news_agent.storage.repository import NewsRepository

    logger.info("Pruning items older than %d days…", settings.retention_days)
    try:
        async with get_session() as session:
            repo = NewsRepository(session)
            deleted = await repo.prune_old_items(settings.retention_days)
        logger.info("Pruned %d old items", deleted)
    except Exception as e:
        logger.error("Prune job failed: %s", e, exc_info=True)


def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    # Fetch every N hours
    scheduler.add_job(
        _fetch_job,
        trigger=IntervalTrigger(hours=settings.schedule_interval_hours),
        id="fetch_job",
        name="Fetch news from all sources",
        replace_existing=True,
    )

    # Daily digest at 08:00
    scheduler.add_job(
        _digest_job,
        trigger=CronTrigger(hour=8, minute=0),
        id="digest_job",
        name="Generate daily digest",
        replace_existing=True,
    )

    # Prune at 03:00
    scheduler.add_job(
        _prune_job,
        trigger=CronTrigger(hour=3, minute=0),
        id="prune_job",
        name="Prune old items",
        replace_existing=True,
    )

    return scheduler


async def run_scheduler():
    """Start the async scheduler and block until interrupted."""
    from news_agent.storage import init_db
    await init_db()

    scheduler = build_scheduler()
    scheduler.start()
    logger.info(
        "Scheduler started. Fetching every %dh, digest at 08:00, prune at 03:00.",
        settings.schedule_interval_hours,
    )

    # Run an initial fetch immediately
    await _fetch_job()

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")
