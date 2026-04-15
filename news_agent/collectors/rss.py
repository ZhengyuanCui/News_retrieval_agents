from __future__ import annotations

import logging
from datetime import datetime, timedelta

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)


import base64 as _base64
import re as _re


def _decode_google_news_url(url: str) -> str:
    """Decode a Google News RSS redirect URL to the real article URL.

    Google News encodes the target URL in base64 inside the path segment
    (e.g. news.google.com/rss/articles/CBMi...).  Decoding the base64 payload
    reveals the real URL without any HTTP request.
    """
    if "news.google.com" not in url:
        return url
    match = _re.search(r"/articles/([A-Za-z0-9_=-]+)", url)
    if not match:
        return url
    encoded = match.group(1)
    # Restore padding
    encoded += "=" * (4 - len(encoded) % 4)
    try:
        data = _base64.urlsafe_b64decode(encoded)
        for prefix in (b"https://", b"http://"):
            idx = data.find(prefix)
            if idx != -1:
                end = data.find(b"\x00", idx)
                decoded_url = data[idx: end if end != -1 else len(data)].decode("utf-8", errors="replace")
                # Sanity check: must look like a real URL with a dot in the host
                if "." in decoded_url[:60]:
                    return decoded_url
    except Exception as e:
        logger.debug("Google News URL decode failed for %s: %s", url[:80], e)
    return url


async def _resolve_url(url: str) -> str:
    """Resolve a Google News redirect URL — try local base64 decode first,
    fall back to an HTTP GET if that doesn't yield a non-Google URL."""
    if "news.google.com" not in url:
        return url
    decoded = _decode_google_news_url(url)
    if decoded != url:
        return decoded
    # Fallback: follow HTTP redirect
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewsAgent/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                final = str(resp.url)
        return url if "news.google.com" in final else final
    except Exception as e:
        logger.debug("URL resolution fallback failed for %s: %s", url[:80], e)
        return url

# (url, topic, source_id, display_label)
DEFAULT_RSS_FEEDS: list[tuple[str, str, str]] = [
    # AI / Tech
    ("https://www.techmeme.com/feed.xml",                               "ai",     "techmeme"),
    ("https://feeds.feedburner.com/venturebeat/SZYF",                   "ai",     "venturebeat"),
    ("https://techcrunch.com/category/artificial-intelligence/feed/",   "ai",     "techcrunch"),
    ("https://www.wired.com/feed/rss",                                   "ai",     "wired"),
    ("https://openai.com/news/rss.xml",                                  "ai",     "openai"),
    ("https://www.deepmind.com/blog/rss.xml",                            "ai",     "deepmind"),
    # Sports
    ("https://www.espn.com/espn/rss/nba/news",                           "basketball", "espn-nba"),
    # Finance / Markets
    ("https://feeds.bloomberg.com/markets/news.rss",                     "stocks", "bloomberg"),
    ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml",                    "stocks", "wsj"),
    ("https://www.ft.com/markets?format=rss",                            "stocks", "ft"),
    ("https://www.cnbc.com/id/10000664/device/rss/rss.html",             "stocks", "cnbc"),
    ("https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines","stocks", "marketwatch"),
]


class RSSCollector(BaseCollector):
    """Aggregates curated RSS feeds from major tech and finance publications."""

    source_name = "rss"
    rate_limit_delay = 1.0

    def is_enabled(self) -> bool:
        return settings.linkedin_enabled  # reuses existing config flag

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(2))
    async def _fetch_feed(self, url: str, topic: str, source_id: str) -> list[NewsItem]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsAgent/1.0)"}
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        items = []
        max_age = datetime.utcnow() - timedelta(days=7)

        for entry in feed.entries[:15]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))[:2000]
            published = entry.get("published_parsed") or entry.get("updated_parsed")

            if not title or not link:
                continue

            published_at = datetime(*published[:6]) if published else datetime.utcnow()

            # Skip articles older than 7 days — they're not news
            if published_at < max_age:
                continue

            try:
                items.append(NewsItem(
                    source=source_id,
                    topic=topic,
                    title=title,
                    url=link,
                    content=summary or title,
                    published_at=published_at,
                    raw_score=0.4,
                ))
            except Exception as e:
                logger.debug("Skipping invalid feed entry from %s: %s", source_id, e)
                continue

        return items

    async def fetch(self) -> list[NewsItem]:
        if not self.is_enabled():
            return []

        items: list[NewsItem] = []

        # When unrestricted, also pull Google News top stories (no query = general trending)
        if not self.topics:
            import asyncio as _asyncio
            general_urls = [
                ("https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en", "news", "google-news"),
                ("https://feeds.bbci.co.uk/news/rss.xml", "news", "bbc"),
            ]
            for url, topic, source_id in general_urls:
                try:
                    feed_items = await self._search_news_rss(url, topic, source_id)
                    items += feed_items
                except Exception as e:
                    logger.warning("General RSS '%s' failed: %s", source_id, e)

        for url, feed_topic, source_id in DEFAULT_RSS_FEEDS:
            if self.topics and feed_topic not in self.topics:
                continue
            try:
                await self._rate_limit()
                # Keep topic semantic (ai/stocks) even during broad fetches.
                # The publication belongs in source, not topic; otherwise digest
                # generation treats source IDs like "techmeme" as digest topics.
                feed_items = await self._fetch_feed(url, feed_topic, source_id)
                logger.debug("RSS '%s' → %d items", source_id, len(feed_items))
                items += feed_items
            except Exception as e:
                logger.warning("RSS '%s' failed: %s", source_id, e)

        logger.info("RSSCollector fetched %d items", len(items))
        return items[: settings.max_items_per_source * 2]

    async def _search_news_rss(self, url: str, topic: str, source_id: str) -> list[NewsItem]:
        """Fetch a search-based RSS feed using feedparser's own HTTP stack.

        feedparser handles redirects, consent pages, and content-negotiation better
        than httpx for services like Google News and Bing News.
        """
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, url)

        items: list[NewsItem] = []
        max_age = datetime.utcnow() - timedelta(days=7)

        for entry in feed.entries[:20]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))[:2000]
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if not title or not link:
                continue
            published_at = datetime(*published[:6]) if published else datetime.utcnow()
            if published_at < max_age:
                continue
            # Decode Google News redirect URLs immediately — no HTTP request needed
            link = _decode_google_news_url(link)
            try:
                items.append(NewsItem(
                    source=source_id,
                    topic=topic,
                    title=title,
                    url=link,
                    content=summary or title,
                    published_at=published_at,
                    raw_score=0.5,
                ))
            except Exception as e:
                logger.debug("Skipping invalid entry from %s: %s", source_id, e)

        # Resolve Google News redirect URLs concurrently
        if any("news.google.com" in i.url for i in items):
            import asyncio as _asyncio
            resolved = await _asyncio.gather(*[_resolve_url(i.url) for i in items])
            for item, real_url in zip(items, resolved):
                item.url = real_url

        logger.debug("News search '%s' → %d items", source_id, len(items))
        return items

    async def fetch_keyword(self, keyword: str) -> list[NewsItem]:
        """Search Google News and Bing News for any keyword."""
        if not self.is_enabled():
            return []
        import asyncio as _asyncio
        from urllib.parse import quote_plus
        q = quote_plus(keyword)
        google_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        bing_url = f"https://www.bing.com/news/search?q={q}&format=rss"
        results = await _asyncio.gather(
            self._search_news_rss(google_url, keyword, "google-news"),
            self._search_news_rss(bing_url, keyword, "bing-news"),
            return_exceptions=True,
        )
        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("News search RSS failed for %r: %s", keyword, r)
            else:
                items.extend(r)
        logger.info("RSSCollector keyword=%r fetched %d items", keyword, len(items))
        return items
