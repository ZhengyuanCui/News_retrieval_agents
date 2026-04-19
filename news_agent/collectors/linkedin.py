from __future__ import annotations

import logging
from datetime import datetime

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

# Public RSS feeds relevant to AI and tech news (supplement for LinkedIn-style content)
DEFAULT_RSS_FEEDS: list[tuple[str, str, str]] = [
    # (url, topic, source_label)
    ("https://www.techmeme.com/feed.xml", "ai", "TechMeme"),
    ("https://feeds.feedburner.com/venturebeat/SZYF", "ai", "VentureBeat AI"),
    ("https://techcrunch.com/category/artificial-intelligence/feed/", "ai", "TechCrunch AI"),
    ("https://www.wired.com/feed/rss", "ai", "Wired"),
    ("https://feeds.bloomberg.com/markets/news.rss", "stocks", "Bloomberg Markets"),
    ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "stocks", "WSJ Markets"),
    ("https://www.ft.com/markets?format=rss", "stocks", "FT Markets"),
    ("https://www.cnbc.com/id/10000664/device/rss/rss.html", "stocks", "CNBC Markets"),
    ("https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", "stocks", "MarketWatch"),
    ("https://news.mit.edu/rss/research", "ai", "MIT Research"),
    ("https://openai.com/news/rss.xml", "ai", "OpenAI News"),
    ("https://www.deepmind.com/blog/rss.xml", "ai", "DeepMind Blog"),
]


class LinkedInCollector(BaseCollector):
    """
    LinkedIn does not expose a public post API.
    This collector uses curated RSS feeds as a high-quality substitute,
    covering similar professional/industry content.
    Falls back gracefully if any feed is unreachable.
    """

    source_name = "linkedin"
    rate_limit_delay = 1.0

    def is_enabled(self) -> bool:
        return settings.linkedin_enabled

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3))
    async def _fetch_feed(self, url: str, topic: str, label: str) -> list[NewsItem]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsAgent/1.0; +https://github.com/newsagent)"}
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content = resp.text

        feed = feedparser.parse(content)
        items = []

        for entry in feed.entries[:15]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))[:2000]
            published = entry.get("published_parsed") or entry.get("updated_parsed")

            if not title or not link:
                continue

            if published:
                published_at = datetime(*published[:6])
            else:
                published_at = datetime.utcnow()

            items.append(
                NewsItem(
                    source="linkedin",
                    topic=topic,
                    title=title,
                    url=link,
                    content=summary or title,
                    published_at=published_at,
                    raw_score=0.4,
                )
            )

        return items

    async def fetch(self) -> list[NewsItem]:
        if not self.is_enabled():
            logger.warning("LinkedInCollector disabled")
            return []

        items: list[NewsItem] = []

        for url, topic, label in DEFAULT_RSS_FEEDS:
            if self.topics and topic not in self.topics:
                continue
            try:
                await self._rate_limit()
                feed_items = await self._fetch_feed(url, topic, label)
                logger.debug("RSS feed '%s' returned %d items", label, len(feed_items))
                items += feed_items
            except Exception as e:
                logger.warning("RSS feed '%s' failed: %s", label, e)

        logger.info("LinkedInCollector (RSS) fetched %d items", len(items))
        return items[: settings.max_items_per_source * 2]
