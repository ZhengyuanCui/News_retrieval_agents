from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem
from news_agent.spam import is_spam_ml_batch

logger = logging.getLogger(__name__)

# Fast keyword pre-filter — catches unambiguous scam patterns before the ML
# model runs.  Keep this list focused; overly broad terms cause false positives
# on legitimate finance/sports discussion.
_SPAM_PHRASES = [
    "guaranteed profit", "guaranteed return", "guaranteed income",
    "dm me for profit", "dm for signals",
    "100% win rate", "never lose",
    "copy my trades", "mirror trade",
    "get rich quick",
    # influencer / guru promotion patterns — direct
    "go follow", "check out my", "follow the incredible",
    "stock picks always", "stock market guru",
    "making a fortune", "never down",
    "his picks", "her picks",
    # influencer / guru promotion patterns — testimonial style
    "my trading improved", "my returns improved",
    "started following @", "after following @",
    "since following @", "since i started following",
    "his insights", "her insights",
    "his signals", "her signals",
    "align with market", "aligns with market",
]

# Tweets with many cashtags ($NVDA $AMD …) mixed with unrelated content are
# almost always spam dumps.  Legitimate discussion rarely exceeds 3 cashtags.
_CASHTAG_SPAM_THRESHOLD = 4


def _keyword_spam(text: str) -> bool:
    t = text.lower()
    if any(p in t for p in _SPAM_PHRASES):
        return True
    # Count $TICKER references — spam tweets pile them on
    if text.count("$") >= _CASHTAG_SPAM_THRESHOLD:
        return True
    return False


def _batch_spam_filter(tweets: list, engagements: list[int], engagement_floor: float,
                       seen_ids: set, max_age, max_likes: int, keyword: str) -> list[NewsItem]:
    """
    Filter engagement floor + dedup, then batch-classify remaining tweets for spam.
    Returns NewsItem list for tweets that pass all filters.
    """
    from urllib.parse import urlparse

    # Phase 1: cheap filters (no ML)
    candidates = []
    candidate_engagements = []
    for tweet, engagement in zip(tweets, engagements):
        if tweet.id in seen_ids:
            continue
        if engagement < engagement_floor:
            continue
        created = tweet.created_at
        if created and created.tzinfo:
            created = created.replace(tzinfo=None)
        if created and created < max_age:
            continue
        if _keyword_spam(tweet.text):
            logger.debug("Keyword spam: %s", tweet.text[:80])
            continue
        candidates.append((tweet, engagement, created))
        candidate_engagements.append(engagement)

    if not candidates:
        return []

    # Phase 2: batch ML spam classification
    texts = [t.text for t, _, _ in candidates]
    spam_flags = is_spam_ml_batch(texts)

    items = []
    for (tweet, engagement, created), is_spam in zip(candidates, spam_flags):
        if is_spam:
            logger.debug("ML spam: %s", tweet.text[:80])
            continue
        url = f"https://x.com/i/web/status/{tweet.id}"
        source = "x"
        if tweet.entities and tweet.entities.get("urls"):
            expanded = tweet.entities["urls"][0].get("expanded_url", "")
            if expanded and "twitter.com" not in expanded and "x.com" not in expanded:
                url = expanded
                host = urlparse(expanded).hostname or ""
                parts = host.lstrip("www.").split(".")
                source = parts[0] if parts else "x"
        seen_ids.add(tweet.id)
        items.append(NewsItem(
            source=source, topic=keyword,
            title=tweet.text[:280], url=url, content=tweet.text,
            published_at=created or datetime.utcnow(),
            raw_score=_normalize(engagement, 0, max_likes * 3),
        ))
    return items


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.5
    return min(1.0, max(0.0, (value - min_val) / (max_val - min_val)))


class TwitterCollector(BaseCollector):
    source_name = "x"
    rate_limit_delay = 2.0  # be conservative

    def is_enabled(self) -> bool:
        return settings.twitter_enabled and bool(settings.twitter_bearer_token)

    def _build_client(self) -> tweepy.Client:
        return tweepy.Client(bearer_token=settings.twitter_bearer_token, wait_on_rate_limit=True)

    @retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(3))
    def _search_sync(self, client: tweepy.Client, query: str, max_results: int) -> list:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "public_metrics", "author_id", "entities"],
            expansions=["author_id"],
        )
        return response.data or []

    async def _search(self, client: tweepy.Client, query: str, max_results: int) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, client, query, max_results)

    async def fetch(self) -> list[NewsItem]:
        if not self.is_enabled():
            logger.warning("TwitterCollector disabled or missing bearer token")
            return []

        # Without explicit topics, Twitter fetch is a no-op — use fetch_keyword for specific searches
        if not self.topics:
            return []

        client = self._build_client()
        items: list[NewsItem] = []

        queries: list[tuple[str, str]] = [
            (f"{topic} lang:en -is:retweet -is:nullcast", topic) for topic in self.topics
        ]

        max_per_query = max(10, settings.max_items_per_source // max(len(queries), 1))
        max_per_query = min(max_per_query, 100)  # API max is 100

        max_age = datetime.utcnow() - timedelta(days=7)

        for query, topic in queries:
            try:
                await self._rate_limit()
                tweets = await self._search(client, query, max_per_query)

                engagements = [
                    t.public_metrics.get("like_count", 0) + t.public_metrics.get("retweet_count", 0) * 2
                    for t in tweets
                ]
                max_engagement = max(engagements, default=1)
                max_likes = max((t.public_metrics.get("like_count", 0) for t in tweets), default=1)
                # Relative threshold: keep tweets above 5% of batch peak, capped at 50
                # so a single viral outlier doesn't eliminate the entire batch.
                engagement_floor = min(max(20, max_engagement * 0.05), 100)
                seen_ids: set = set()
                batch = _batch_spam_filter(
                    tweets, engagements, engagement_floor,
                    seen_ids, max_age, max_likes, topic,
                )
                items.extend(batch)

            except tweepy.TweepyException as e:
                logger.error("Twitter API error (query=%r): %s — check bearer token and API access tier", query[:50], e)
                raise
            except Exception as e:
                logger.error("Unexpected Twitter error: %s", e)
                raise

        logger.info("TwitterCollector fetched %d items", len(items))
        return items[: settings.max_items_per_source * 2]

    async def fetch_keyword(self, keyword: str) -> list[NewsItem]:
        if not self.is_enabled():
            return []
        client = self._build_client()
        items: list[NewsItem] = []
        max_age = datetime.utcnow() - timedelta(days=7)
        # Two passes:
        # 1. has:links  — article shares; broader spam tolerance since links are required
        # 2. no has:links — organic discussion (game reactions, scores); higher engagement
        #    floor to compensate for the missing link requirement
        queries = [
            (f'{keyword} lang:en -is:retweet -is:nullcast has:links', 0.05),
            (f'{keyword} lang:en -is:retweet -is:nullcast', 0.15),
        ]

        seen_ids: set = set()

        for query, floor_pct in queries:
            try:
                await self._rate_limit()
                tweets = await self._search(client, query, 50)
                metrics_list = [(t.public_metrics or {}) for t in tweets]
                engagements = [
                    m.get("like_count", 0) + m.get("retweet_count", 0) * 2
                    for m in metrics_list
                ]
                max_engagement = max(engagements, default=1)
                max_likes = max((m.get("like_count", 0) for m in metrics_list), default=1)
                engagement_floor = min(max(20, max_engagement * floor_pct), 100)
                batch = _batch_spam_filter(
                    tweets, engagements, engagement_floor,
                    seen_ids, max_age, max_likes, keyword,
                )
                items.extend(batch)
            except Exception as e:
                logger.error("Twitter keyword fetch error (%r): %s", keyword, e, exc_info=True)
        logger.info("TwitterCollector keyword=%r fetched %d items", keyword, len(items))
        return items
