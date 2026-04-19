from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import praw
import praw.exceptions

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

AI_SUBREDDITS = [
    "MachineLearning",
    "artificial",
    "LocalLLaMA",
    "ChatGPT",
    "singularity",
    "deeplearning",
    "openai",
]

STOCK_SUBREDDITS = [
    "investing",
    "stocks",
    "SecurityAnalysis",
    "wallstreetbets",
    "finance",
]


class RedditCollector(BaseCollector):
    source_name = "reddit"
    rate_limit_delay = 1.0

    def is_enabled(self) -> bool:
        cid = settings.reddit_client_id or ""
        return settings.reddit_enabled and bool(cid) and "your_" not in cid

    def _build_client(self) -> praw.Reddit:
        return praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            username=settings.reddit_username,
            user_agent=settings.reddit_user_agent,
        )

    async def fetch(self) -> list[NewsItem]:
        if not self.is_enabled():
            logger.warning("RedditCollector disabled or missing credentials")
            return []

        items: list[NewsItem] = []
        reddit = self._build_client()

        # Build (subreddit_name, topic) pairs.
        # When topics are specified, use curated subreddits for those topics.
        # When unrestricted (topics=[]), fetch r/popular and label each post
        # by its actual subreddit so the DB reflects real topic diversity.
        if self.topics:
            topic_subreddits = []
            for sub in AI_SUBREDDITS:
                if "ai" in self.topics:
                    topic_subreddits.append((sub, "ai"))
            for sub in STOCK_SUBREDDITS:
                if "stocks" in self.topics:
                    topic_subreddits.append((sub, "stocks"))
            fetch_popular = False
        else:
            topic_subreddits = []
            fetch_popular = True

        limit_per_sub = max(10, settings.max_items_per_source // len(topic_subreddits)) if topic_subreddits else 25

        for subreddit_name, topic in topic_subreddits:
            try:
                await self._rate_limit()
                loop = asyncio.get_event_loop()
                posts = await loop.run_in_executor(
                    None, lambda s=subreddit_name: list(reddit.subreddit(s).hot(limit=limit_per_sub))
                )
                max_score = max((p.score for p in posts), default=1)
                for post in posts:
                    if post.is_self and not post.selftext:
                        continue
                    content = post.selftext[:2000] if post.selftext else post.title
                    items.append(NewsItem(
                        source="reddit", topic=topic, title=post.title,
                        url=f"https://www.reddit.com{post.permalink}",
                        content=content,
                        author=str(post.author) if post.author else None,
                        published_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(tzinfo=None),
                        raw_score=self.normalize_score(post.score, 0, max_score),
                    ))
            except praw.exceptions.PRAWException as e:
                logger.error("Reddit error for r/%s: %s", subreddit_name, e)
            except Exception as e:
                logger.error("Unexpected error for r/%s: %s", subreddit_name, e)

        if fetch_popular:
            try:
                await self._rate_limit()
                loop = asyncio.get_event_loop()
                posts = await loop.run_in_executor(
                    None, lambda: list(reddit.subreddit("popular").hot(limit=50))
                )
                max_score = max((p.score for p in posts), default=1)
                for post in posts:
                    if post.is_self and not post.selftext:
                        continue
                    content = post.selftext[:2000] if post.selftext else post.title
                    # Use the actual subreddit as the topic so items are discoverable
                    topic = post.subreddit.display_name.lower()
                    items.append(NewsItem(
                        source="reddit", topic=topic, title=post.title,
                        url=f"https://www.reddit.com{post.permalink}",
                        content=content,
                        author=str(post.author) if post.author else None,
                        published_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(tzinfo=None),
                        raw_score=self.normalize_score(post.score, 0, max_score),
                    ))
            except Exception as e:
                logger.error("Reddit r/popular error: %s", e)

        logger.info("RedditCollector fetched %d items", len(items))
        return items[: settings.max_items_per_source * 2]

    async def fetch_keyword(self, keyword: str) -> list[NewsItem]:
        if not self.is_enabled():
            return []
        items: list[NewsItem] = []
        reddit = self._build_client()
        subreddit_str = "all"
        time_filter = "day"

        try:
            await self._rate_limit()
            loop = asyncio.get_event_loop()
            posts = await loop.run_in_executor(
                None,
                lambda: list(reddit.subreddit(subreddit_str).search(keyword, sort="hot", limit=30, time_filter=time_filter))
            )
            max_score = max((p.score for p in posts), default=1)
            for post in posts:
                if post.is_self and not post.selftext:
                    continue
                content = post.selftext[:2000] if post.selftext else post.title
                items.append(NewsItem(
                    source="reddit", topic=keyword,
                    title=post.title,
                    url=f"https://www.reddit.com{post.permalink}",
                    content=content,
                    author=str(post.author) if post.author else None,
                    published_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(tzinfo=None),
                    raw_score=self.normalize_score(post.score, 0, max_score),
                ))
        except Exception as e:
            logger.error("Reddit keyword fetch error (%r): %s", keyword, e)
        logger.info("RedditCollector keyword=%r fetched %d items", keyword, len(items))
        return items
