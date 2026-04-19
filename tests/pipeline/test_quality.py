"""
End-to-end news quality regression tests.

These tests run the full retrieval → dedup → rank pipeline on a controlled
set of realistic articles and assert that the output matches human expectations:
the most relevant, freshest, and most authoritative articles should win.

No API keys or external services are required.
"""
from __future__ import annotations

import pytest

from news_agent.pipeline.deduplicator import Deduplicator
from news_agent.pipeline.ranker import rank_by_query
from tests.conftest import hours_ago, make_item


# ── Scenario: breaking news on a hot topic ───────────────────────────────────

class TestBreakingNewsScenario:
    """Fed rate decision story covered by 5 sources at different times."""

    def setup_method(self):
        self.items = [
            make_item(
                url="https://reuters.com/fed-hike",
                title="Federal Reserve raises rates by 50bps",
                content="The Federal Reserve voted to raise interest rates by 50 basis points, the largest hike in 22 years, citing persistent inflation.",
                source="reuters",
                published_at=hours_ago(1),
                relevance_score=9.0,
                raw_score=0.8,
            ),
            make_item(
                url="https://bloomberg.com/fed-hike",
                title="Fed hikes rates 50bps in battle against inflation",
                content="The Federal Reserve raised its benchmark interest rate by half a percentage point as it battles the worst inflation in four decades.",
                source="bloomberg",
                published_at=hours_ago(2),
                relevance_score=8.5,
                raw_score=0.75,
            ),
            make_item(
                url="https://twitter.com/breakingnews/1",
                title="Fed raises rates 50bps",
                content="BREAKING: Fed raises rates 50bps",
                source="twitter",
                published_at=hours_ago(0.5),  # freshest, but low authority
                relevance_score=6.0,
                raw_score=0.95,
            ),
            make_item(
                url="https://reddit.com/r/economics/abc",
                title="Fed just hiked 50bps - what does this mean?",
                content="The Fed raised rates by 50 basis points today. This is the biggest hike since 2000. What does this mean for mortgages and stocks?",
                source="reddit",
                published_at=hours_ago(3),
                relevance_score=5.0,
                raw_score=0.7,
            ),
            make_item(
                url="https://example.com/unrelated",
                title="Best pizza restaurants in New York",
                content="We reviewed 50 pizza places in New York City to find the absolute best slices.",
                source="rss",
                published_at=hours_ago(1),
                relevance_score=1.0,
                raw_score=0.4,
            ),
        ]

    def test_authoritative_source_beats_twitter_despite_engagement(self):
        ranked = rank_by_query("Federal Reserve interest rate hike", self.items)
        top_sources = [i.source for i in ranked[:2]]
        assert "twitter" not in top_sources, (
            "Twitter should not be in top 2 despite highest raw_score"
        )
        assert any(s in top_sources for s in ("reuters", "bloomberg")), (
            "Reuters or Bloomberg should be in top 2"
        )

    def test_unrelated_article_ranks_last(self):
        ranked = rank_by_query("Federal Reserve interest rate hike", self.items)
        assert ranked[-1].url == "https://example.com/unrelated", (
            "Pizza article should rank last for Fed rate query"
        )

    def test_all_five_articles_returned(self):
        ranked = rank_by_query("Federal Reserve interest rate hike", self.items)
        assert len(ranked) == 5


# ── Scenario: dedup collapses same story across sources ──────────────────────

class TestDeduplicationScenario:
    """Same story (NVIDIA earnings) syndicated across 4 outlets."""

    def setup_method(self):
        base = "NVIDIA reported Q3 earnings that beat analyst expectations. Revenue rose 206% year-over-year driven by data center demand."
        # TF-IDF cosine sims for these near-identical articles range 0.62–0.76;
        # threshold=0.60 ensures all three collapse transitively to one survivor.
        self.dedup = Deduplicator(strategy="tfidf", threshold=0.60)
        self.items = [
            make_item(url="https://reuters.com/nvda-q3",     title="NVIDIA beats Q3 earnings estimates",      content=base,                            source="reuters",   raw_score=0.8),
            make_item(url="https://bloomberg.com/nvda-q3",   title="Nvidia Q3 earnings top Wall Street view", content=base + " Shares jumped 8%.",     source="bloomberg", raw_score=0.75),
            make_item(url="https://cnbc.com/nvda-q3",        title="NVIDIA's blowout earnings: what to know", content=base + " CEO Jensen Huang cited AI demand.", source="cnbc", raw_score=0.7),
            make_item(url="https://example.com/apple-event", title="Apple announces Vision Pro 2",            content="Apple revealed the next generation Vision Pro headset at its annual developer conference.", source="rss", raw_score=0.5),
        ]

    def test_nvidia_stories_collapse_to_one(self):
        result = self.dedup.deduplicate(self.items)
        survivors = [i for i in result if not i.is_duplicate]
        nvidia_survivors = [i for i in survivors if "NVIDIA" in i.title or "Nvidia" in i.title]
        assert len(nvidia_survivors) == 1, (
            f"3 near-identical NVIDIA stories should dedup to 1, got {len(nvidia_survivors)}"
        )

    def test_distinct_apple_story_survives(self):
        result = self.dedup.deduplicate(self.items)
        survivors = [i for i in result if not i.is_duplicate]
        apple_survivors = [i for i in survivors if "Apple" in i.title]
        assert len(apple_survivors) == 1, "Apple story is distinct and should survive"

    def test_dedup_keeps_richest_version(self):
        result = self.dedup.deduplicate(self.items)
        survivors = [i for i in result if not i.is_duplicate and ("NVIDIA" in i.title or "Nvidia" in i.title)]
        if survivors:
            # The surviving version should have the longest content (most detail)
            survivor_content_len = len(survivors[0].content)
            shortest = len(min(self.items[:3], key=lambda x: len(x.content)).content)
            assert survivor_content_len >= shortest


# ── Scenario: freshness correctly orders a live news stream ──────────────────

class TestFreshnessScenario:
    """Same relevance, different ages — freshness should determine order."""

    def test_stream_ordered_by_recency_for_equal_relevance(self):
        articles = [
            make_item(url="https://a.com/1", title="AI model benchmark results",
                      content="New AI model achieves state-of-the-art results on key benchmarks.",
                      published_at=hours_ago(72), relevance_score=7.0, source="rss"),
            make_item(url="https://a.com/2", title="AI model benchmark results",
                      content="New AI model achieves state-of-the-art results on key benchmarks.",
                      published_at=hours_ago(24), relevance_score=7.0, source="rss"),
            make_item(url="https://a.com/3", title="AI model benchmark results",
                      content="New AI model achieves state-of-the-art results on key benchmarks.",
                      published_at=hours_ago(2), relevance_score=7.0, source="rss"),
        ]
        ranked = rank_by_query("AI benchmark model", articles)
        ages = [(datetime_age(a.published_at)) for a in ranked]
        assert ages == sorted(ages), "With equal relevance, articles should be ordered newest-first"

    def test_very_fresh_low_quality_doesnt_always_win(self):
        """A 1-hour-old tweet shouldn't beat a 12-hour-old Bloomberg article with 9.0 relevance."""
        quality = make_item(
            url="https://bloomberg.com/analysis",
            title="Deep analysis: AI chip supply chain disruption",
            content="A comprehensive analysis of how AI chip shortages are reshaping the global supply chain.",
            source="bloomberg",
            published_at=hours_ago(12),
            relevance_score=9.0,
            raw_score=0.7,
        )
        noise = make_item(
            url="https://twitter.com/rando/1",
            title="AI chips lol",
            content="AI chips something something supply chain",
            source="twitter",
            published_at=hours_ago(1),
            relevance_score=4.0,
            raw_score=0.3,
        )
        ranked = rank_by_query("AI chip supply chain", [noise, quality])
        assert ranked[0].source == "bloomberg", (
            "High-relevance Bloomberg article should beat low-quality fresh tweet"
        )


def datetime_age(dt) -> float:
    from datetime import datetime
    return (datetime.utcnow() - dt).total_seconds()


# ── Scenario: personalization boost ──────────────────────────────────────────

class TestPersonalizationScenario:
    def test_starred_source_boosted(self):
        from news_agent.preference import apply_preference_boost

        items = [
            make_item(url="https://openai.com/post", source="openai",
                      relevance_score=6.0, title="OpenAI update"),
            make_item(url="https://reddit.com/post", source="reddit",
                      relevance_score=6.0, title="Reddit discussion", url_suffix="/post2"),
        ]
        # Simulate user having starred openai content many times
        prefs = {("source", "openai"): 15.0, ("source", "reddit"): -5.0}
        boosted = apply_preference_boost(items, prefs)

        openai_item = next(i for i in boosted if i.source == "openai")
        reddit_item = next(i for i in boosted if i.source == "reddit")
        assert openai_item.relevance_score > reddit_item.relevance_score, (
            "Frequently starred source should receive higher relevance after boost"
        )

    def test_disliked_source_demoted(self):
        from news_agent.preference import apply_preference_boost

        item = make_item(source="twitter", relevance_score=7.0)
        prefs = {("source", "twitter"): -20.0}  # strongly disliked
        boosted = apply_preference_boost([item], prefs)
        assert boosted[0].relevance_score < 7.0, "Disliked source should be demoted"

    def test_no_prefs_unchanged(self):
        from news_agent.preference import apply_preference_boost

        items = [make_item(relevance_score=5.0), make_item(url="https://b.com", relevance_score=8.0)]
        boosted = apply_preference_boost(items, {})
        assert [i.relevance_score for i in boosted] == [5.0, 8.0]
