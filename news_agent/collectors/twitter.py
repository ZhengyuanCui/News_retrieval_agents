from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem
from news_agent.spam import is_spam_ml_batch

logger = logging.getLogger(__name__)

# URL shortener hostnames — links through these are not real article URLs
_URL_SHORTENERS = frozenset({
    "tinyurl.com", "bit.ly", "ow.ly", "goo.gl", "buff.ly",
    "dlvr.it", "ift.tt", "rebrand.ly", "short.io", "tiny.cc",
    "t.co", "lnkd.in", "fb.me", "is.gd", "v.gd",
})

# Absolute red-lines: these phrases never appear in legitimate financial content.
_HARD_SPAM_PHRASES = [
    "guaranteed profit", "guaranteed return", "guaranteed income",
    "dm me for signals", "dm for signals", "dm me for profit",
    "100% win rate", "never lose",
    "copy my trades", "mirror my trades",
    "go follow @",
    "stock market guru", "trading guru", "market guru",
]

# Weak shill signals: each phrase alone is fine; two or more together with a
# @mention and a cashtag = influencer shill tweet.
_SHILL_INDICATORS = [
    "his picks", "her picks", "their picks",
    "his calls", "her calls", "their calls",
    "his trades", "her trades", "their trades",
    "his strategies", "her strategies", "their strategies",
    "his stock picks", "her stock picks",
    "always go up", "never go down", "always goes up", "never goes down",
    "making a fortune", "making money every day",
    "incredible", "impressive gains", "crushing it",
    "trade alerts", "trading signals",
    "quietly following", "been following",
]

_CASHTAG_RE = re.compile(r"[$][A-Z]{1,5}\b")  # \$ is invalid in Python 3.14+; use char class
_MENTION_RE = re.compile(r"@\w+")


def _keyword_spam(text: str) -> bool:
    t = text.lower()

    # Hard filters — instant reject
    if any(p in t for p in _HARD_SPAM_PHRASES):
        return True

    # 3+ distinct $TICKER cashtags = keyword stuffing
    if len(set(_CASHTAG_RE.findall(text.upper()))) >= 3:
        return True

    # Influencer shill: @mention + cashtag + 2 or more soft indicators
    has_mention = bool(_MENTION_RE.search(text))
    has_cashtag = bool(_CASHTAG_RE.search(text.upper()))
    if has_mention and has_cashtag:
        indicator_count = sum(1 for p in _SHILL_INDICATORS if p in t)
        if indicator_count >= 2:
            return True

    return False


def _batch_spam_filter(tweets: list, engagements: list[int], engagement_floor: float,
                       seen_ids: set, max_age, max_engagement: int, keyword: str) -> list[NewsItem]:
    """
    Filter engagement floor + dedup, then batch-classify remaining tweets for spam.
    Returns NewsItem list for tweets that pass all filters.
    """
    from urllib.parse import urlparse

    logger.info(
        "spam_filter[%s]: %d tweets, floor=%.1f, max_eng=%d",
        keyword, len(tweets), engagement_floor, max(engagements, default=0),
    )

    # Phase 1: cheap filters (no ML)
    candidates = []
    candidate_engagements = []
    low_eng = dup = old = kw_spam = 0
    for tweet, engagement in zip(tweets, engagements):
        if tweet.id in seen_ids:
            dup += 1
            continue
        if engagement < engagement_floor:
            low_eng += 1
            continue
        created = tweet.created_at
        if created and created.tzinfo:
            created = created.replace(tzinfo=None)
        if created and created < max_age:
            old += 1
            continue
        if _keyword_spam(tweet.text):
            kw_spam += 1
            logger.debug("Keyword spam: %s", tweet.text[:80])
            continue
        candidates.append((tweet, engagement, created))
        candidate_engagements.append(engagement)

    logger.info(
        "spam_filter[%s]: dropped low_eng=%d dup=%d old=%d kw_spam=%d → %d candidates",
        keyword, low_eng, dup, old, kw_spam, len(candidates),
    )

    if not candidates:
        return []

    # Phase 2: batch ML spam classification
    texts = [t.text for t, _, _ in candidates]
    spam_flags = is_spam_ml_batch(texts)
    ml_spam_count = sum(spam_flags)
    if ml_spam_count:
        logger.info("spam_filter[%s]: ML flagged %d/%d as spam", keyword, ml_spam_count, len(candidates))

    items = []
    for (tweet, engagement, created), is_spam in zip(candidates, spam_flags):
        if is_spam:
            logger.debug("ML spam: %s", tweet.text[:80])
            continue
        url = f"https://x.com/i/web/status/{tweet.id}"
        source = "x"
        if tweet.entities and tweet.entities.get("urls"):
            expanded = tweet.entities["urls"][0].get("expanded_url", "")
            host = urlparse(expanded).hostname or "" if expanded else ""
            clean_host = host.lstrip("www.")
            is_shortener = clean_host in _URL_SHORTENERS
            if expanded and not is_shortener and "twitter.com" not in expanded and "x.com" not in expanded:
                url = expanded
        seen_ids.add(tweet.id)
        items.append(NewsItem(
            source=source, topic=keyword,
            title=tweet.text[:280], url=url, content=tweet.text,
            published_at=created or datetime.utcnow(),
            raw_score=_normalize(engagement, 0, max_engagement),
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

    def _search_sync(self, client: tweepy.Client, query: str, max_results: int) -> list:
        """Fetch up to max_results tweets, paginating in batches of 100."""
        per_page = 100  # API hard limit per request
        collected = []
        next_token = None

        while len(collected) < max_results:
            want = min(per_page, max_results - len(collected))
            try:
                response = client.search_recent_tweets(
                    query=query,
                    max_results=want,
                    tweet_fields=["created_at", "public_metrics", "author_id", "entities"],
                    expansions=["author_id"],
                    next_token=next_token,
                )
            except Exception as e:
                logger.warning("Twitter search page failed (collected=%d): %s", len(collected), e)
                break
            page = response.data or []
            collected.extend(page)
            next_token = response.meta.get("next_token") if response.meta else None
            if not next_token or not page:
                break  # no more pages

        return collected

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

        max_per_query = 500

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
                # Fixed floor: just filter zero-engagement noise.
                # Relative floors caused viral outliers to eliminate entire batches.
                engagement_floor = 3
                seen_ids: set = set()
                batch = _batch_spam_filter(
                    tweets, engagements, engagement_floor,
                    seen_ids, max_age, max_engagement, topic,
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
            f'{keyword} lang:en -is:retweet -is:nullcast has:links',
            f'{keyword} lang:en -is:retweet -is:nullcast',
        ]

        seen_ids: set = set()

        for query in queries:
            try:
                await self._rate_limit()
                tweets = await self._search(client, query, 500)
                metrics_list = [(t.public_metrics or {}) for t in tweets]
                engagements = [
                    m.get("like_count", 0) + m.get("retweet_count", 0) * 2
                    for m in metrics_list
                ]
                max_engagement = max(engagements, default=1)
                # Fixed floor: filter zero-engagement noise only.
                # has:links gets a slightly lower floor since link-tweets need a click anyway.
                engagement_floor = 3 if "has:links" in query else 5
                batch = _batch_spam_filter(
                    tweets, engagements, engagement_floor,
                    seen_ids, max_age, max_engagement, keyword,
                )
                items.extend(batch)
            except Exception as e:
                logger.error("Twitter keyword fetch error (%r): %s", keyword, e, exc_info=True)
        logger.info("TwitterCollector keyword=%r fetched %d items", keyword, len(items))
        return items
