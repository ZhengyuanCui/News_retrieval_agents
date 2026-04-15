from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from news_agent.collectors.base import BaseCollector
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)


class Aggregator:
    def __init__(self, collectors: list[BaseCollector]) -> None:
        self.collectors = collectors

    async def fetch_all(self) -> list[NewsItem]:
        """Run all collectors concurrently and merge results."""
        tasks = [self._safe_fetch(collector) for collector in self.collectors]
        results = await asyncio.gather(*tasks)

        all_items: list[NewsItem] = []
        for collector, items in zip(self.collectors, results):
            logger.info("%s returned %d items", collector.source_name, len(items))
            all_items.extend(collector.tag_languages(items))

        # Sort by published_at descending
        all_items.sort(key=lambda x: x.published_at, reverse=True)
        return all_items

    async def _safe_fetch(self, collector: BaseCollector) -> list[NewsItem]:
        try:
            return await collector.fetch()
        except Exception as e:
            logger.error("Collector %s failed: %s", collector.source_name, e, exc_info=True)
            return []
