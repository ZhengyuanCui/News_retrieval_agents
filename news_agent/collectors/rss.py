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
import html as _html
import re as _re


def _clean_summary(text: str) -> str:
    """Strip HTML tags and entities from RSS summary/description fields.

    Google News RSS wraps summaries in <a href="...">title</a> HTML.
    Store only the human-readable text.
    """
    text = _html.unescape(text or "")
    text = _re.sub(r"<[^>]+>", " ", text)
    return _re.sub(r"\s+", " ", text).strip()




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

# (url, topic, source_id)
DEFAULT_RSS_FEEDS: list[tuple[str, str, str]] = [
    # ── AI News & Analysis ────────────────────────────────────────────────────
    ("https://www.techmeme.com/feed.xml",                                "ai", "techmeme"),
    ("https://feeds.feedburner.com/venturebeat/SZYF",                   "ai", "venturebeat"),
    ("https://techcrunch.com/category/artificial-intelligence/feed/",   "ai", "techcrunch"),
    ("https://www.wired.com/feed/rss",                                   "ai", "wired"),
    ("https://thenextweb.com/feed",                                      "ai", "thenextweb"),
    ("https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss","ai", "ieee-spectrum"),
    ("https://www.technologyreview.com/feed/",                           "ai", "mit-tech-review"),
    ("https://www.theregister.com/headlines.atom",                       "ai", "the-register"),

    # ── Frontier Labs — Research Blogs ────────────────────────────────────────
    ("https://openai.com/news/rss.xml",                                  "ai", "openai"),
    ("https://www.deepmind.com/blog/rss.xml",                            "ai", "deepmind"),
    # Anthropic has no native RSS — use Google News search
    ("https://news.google.com/rss/search?q=Anthropic+AI&hl=en-US&gl=US&ceid=US:en",
                                                                         "ai", "anthropic-news"),
    ("https://ai.googleblog.com/feeds/posts/default",                    "ai", "google-ai-blog"),
    ("https://research.facebook.com/feed/",                              "ai", "meta-research"),
    ("https://machinelearning.apple.com/rss/all_articles.rss",           "ai", "apple-ml"),
    ("https://www.microsoft.com/en-us/research/feed/",                   "ai", "microsoft-research"),
    ("https://aws.amazon.com/blogs/machine-learning/feed/",              "ai", "amazon-science"),
    ("https://huggingface.co/blog/feed.xml",                             "ai", "huggingface"),
    ("https://blogs.nvidia.com/blog/category/deep-learning/feed/",       "ai", "nvidia-ai"),
    ("https://allenai.org/blog?format=rss",                              "ai", "allen-ai"),
    ("https://blog.mistral.ai/rss/",                                     "ai", "mistral"),
    ("https://stability.ai/news?format=rss",                             "ai", "stability-ai"),
    ("https://cohere.com/blog/rss",                                      "ai", "cohere"),

    # ── Robotics & Embodied AI ────────────────────────────────────────────────
    # Physical Intelligence, Figure AI, World Labs have no native RSS — via Google News
    ("https://news.google.com/rss/search?q=Physical+Intelligence+robotics+AI&hl=en-US&gl=US&ceid=US:en",
                                                                         "ai", "physical-intelligence"),
    ("https://news.google.com/rss/search?q=Figure+AI+robot&hl=en-US&gl=US&ceid=US:en",
                                                                         "ai", "figure-ai"),
    ("https://news.google.com/rss/search?q=World+Labs+Fei-Fei+Li+spatial+AI&hl=en-US&gl=US&ceid=US:en",
                                                                         "ai", "world-labs"),
    ("https://news.google.com/rss/search?q=Boston+Dynamics+robot&hl=en-US&gl=US&ceid=US:en",
                                                                         "ai", "boston-dynamics"),
    ("https://waymo.com/blog/rss",                                       "ai", "waymo"),

    # ── Generative Media & Multimodal ─────────────────────────────────────────
    ("https://news.google.com/rss/search?q=Runway+ML+video+generation&hl=en-US&gl=US&ceid=US:en",
                                                                         "ai", "runway-ml"),
    ("https://elevenlabs.io/blog/rss.xml",                               "ai", "elevenlabs"),
    ("https://sakana.ai/blog/index.xml",                                 "ai", "sakana-ai"),

    # ── arXiv — Daily Paper Feeds ────────────────────────────────────────────
    ("https://arxiv.org/rss/cs.AI",                                      "ai", "arxiv-cs-ai"),
    ("https://arxiv.org/rss/cs.LG",                                      "ai", "arxiv-cs-lg"),
    ("https://arxiv.org/rss/cs.CV",                                      "ai", "arxiv-cs-cv"),
    ("https://arxiv.org/rss/cs.RO",                                      "ai", "arxiv-cs-ro"),
    ("https://arxiv.org/rss/cs.CL",                                      "ai", "arxiv-cs-cl"),
    ("https://arxiv.org/rss/stat.ML",                                    "ai", "arxiv-stat-ml"),

    # ── Paper Digests & Newsletters ───────────────────────────────────────────
    ("https://thegradient.pub/rss/",                                     "ai", "the-gradient"),
    ("https://importai.substack.com/feed",                               "ai", "import-ai"),       # Jack Clark
    ("https://www.deeplearning.ai/the-batch/rss/",                       "ai", "the-batch"),        # Andrew Ng
    ("https://lastweekin.ai/feed",                                       "ai", "last-week-in-ai"),
    ("https://paperswithcode.com/newsletter/rss",                        "ai", "papers-with-code"),

    # ── Top AI Researchers — Blogs ────────────────────────────────────────────
    ("https://lilianweng.github.io/index.xml",                           "ai", "lilian-weng"),
    ("https://karpathy.github.io/feed.xml",                              "ai", "karpathy-blog"),
    ("https://www.fast.ai/index.xml",                                    "ai", "fastai"),
    ("https://colah.github.io/rss.xml",                                  "ai", "colah"),
    ("https://blog.samaltman.com/rss",                                   "ai", "sam-altman"),
    ("https://jalammar.github.io/feed.xml",                              "ai", "jay-alammar"),
    ("https://ruder.io/rss/index.rss",                                   "ai", "sebastian-ruder"),
    ("https://huyenchip.com/feed.xml",                                   "ai", "chip-huyen"),
    ("https://simonwillison.net/atom/everything/",                       "ai", "simon-willison"),   # prolific AI/LLM blogger
    ("https://fchollet.substack.com/feed",                               "ai", "chollet"),          # François Chollet
    ("https://bair.berkeley.edu/blog/feed.xml",                          "ai", "bair"),
    ("https://ai.stanford.edu/blog/feed.xml",                            "ai", "stanford-ai"),

    # ── AI Safety & Alignment ─────────────────────────────────────────────────
    ("https://www.alignmentforum.org/feed.xml",                          "ai", "alignment-forum"),
    ("https://www.lesswrong.com/feed.xml",                               "ai", "lesswrong"),
    ("https://intelligence.org/feed/",                                   "ai", "miri"),

    # ── Robotics / Vision (additional) ───────────────────────────────────────
    ("https://spectrum.ieee.org/feeds/topic/robotics.rss",               "ai", "ieee-robotics"),
    ("https://www.csail.mit.edu/news?format=rss",                        "ai", "mit-csail"),

    # ── AI Podcasts — Current Events, Ethics & Future Vision ──────────────────
    # Technical depth + researcher interviews
    ("https://twimlai.com/feed/podcast/",                                "ai", "twiml-ai"),          # This Week in ML & AI — Sam Charrington
    ("https://www.latent.space/feed",                                    "ai", "latent-space"),       # Latent Space — AI engineering deep dives
    ("https://lexfridman.com/feed/podcast/",                             "ai", "lex-fridman-pod"),   # Lex Fridman Podcast
    ("https://changelog.com/practicalai/feed",                           "ai", "practical-ai"),      # Practical AI — weekly applied AI
    ("https://feeds.feedburner.com/nvidia-ai-podcast",                   "ai", "nvidia-ai-pod"),     # NVIDIA AI Podcast
    # Future, ethics & societal impact
    ("https://futureoflife.org/podcast/feed/",                           "ai", "future-of-life"),    # FLI Podcast — existential risk, AI safety
    ("https://80000hours.org/podcast/feed/",                             "ai", "80k-hours"),         # 80,000 Hours — AI safety careers & ethics
    ("https://cognitiverevolution.ai/feed/",                             "ai", "cognitive-rev"),     # The Cognitive Revolution — Nathan Labenz
    ("https://www.nopriorsai.com/feed",                                  "ai", "no-priors"),         # No Priors — Sarah Guo & Elad Gil (VCs)
    ("https://www.eye-on.ai/podcast-audio/feed",                         "ai", "eye-on-ai"),         # Eye on AI — Craig Smith
    # Humane tech & ethics focus
    ("https://your-undivided-attention.simplecast.com/episodes/rss",     "ai", "undivided-attention"), # Tristan Harris / Center for Humane Tech
    # Neuroscience × AI
    ("https://braininspired.co/feed/podcast/",                           "ai", "brain-inspired"),    # Brain Inspired — Paul Middlebrooks

    # ── Sports ────────────────────────────────────────────────────────────────
    ("https://www.espn.com/espn/rss/nba/news",                           "basketball", "espn-nba"),

    # ── Finance / Markets ─────────────────────────────────────────────────────
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
        return True  # RSS needs no credentials

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
            summary = _clean_summary(entry.get("summary", entry.get("description", ""))[:2000])
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

        import asyncio as _asyncio

        tasks: list[tuple[str, str, str, bool]] = []  # (url, topic, source_id, use_search_method)

        # When unrestricted, also pull Google News top stories (no query = general trending)
        if not self.topics:
            tasks += [
                ("https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en", "news", "google-news", True),
                ("https://feeds.bbci.co.uk/news/rss.xml", "news", "bbc", True),
            ]

        for url, feed_topic, source_id in DEFAULT_RSS_FEEDS:
            if self.topics and feed_topic not in self.topics:
                continue
            tasks.append((url, feed_topic, source_id, False))

        async def _fetch_one(url: str, topic: str, source_id: str, use_search: bool) -> list[NewsItem]:
            try:
                if use_search:
                    return await self._search_news_rss(url, topic, source_id)
                else:
                    return await self._fetch_feed(url, topic, source_id)
            except Exception as e:
                logger.warning("RSS '%s' failed: %s", source_id, e)
                return []

        results = await _asyncio.gather(*[_fetch_one(u, t, s, m) for u, t, s, m in tasks])
        items: list[NewsItem] = []
        for source_items in results:
            # Cap per feed so no single high-volume source (e.g. arXiv) crowds out others
            items += source_items[:8]

        logger.info("RSSCollector fetched %d items across %d feeds", len(items), len(tasks))
        return items

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
            summary = _clean_summary(entry.get("summary", entry.get("description", ""))[:2000])
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
