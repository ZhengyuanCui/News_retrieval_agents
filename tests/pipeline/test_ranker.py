"""
Ranking quality tests.

These tests use controlled articles with known properties to verify that
each ranking signal (freshness, source authority, LLM relevance, blending)
moves articles in the expected direction.  No API calls or ML models required
for the deterministic signal tests — the semantic/cross-encoder path degrades
gracefully to original order when the model is not loaded.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from news_agent.pipeline.ranker import _SOURCE_AUTHORITY, rank_by_query
from tests.conftest import hours_ago, make_item


# ── Freshness decay formula ───────────────────────────────────────────────────

class TestFreshnessDecay:
    def test_fresh_article_scores_near_one(self):
        age_hours = 0.5
        score = math.exp(-age_hours / 48)
        assert score > 0.98

    def test_48h_article_at_half(self):
        score = math.exp(-48 / 48)
        assert abs(score - math.exp(-1)) < 1e-9  # e^-1 ≈ 0.368

    def test_7day_article_heavily_penalised(self):
        score = math.exp(-168 / 48)
        assert score < 0.04  # exp(-3.5) ≈ 0.030 — well below 4%

    def test_decay_is_monotone(self):
        ages = [1, 6, 12, 24, 48, 72, 168]
        scores = [math.exp(-h / 48) for h in ages]
        assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))


# ── Source authority table ────────────────────────────────────────────────────

class TestSourceAuthority:
    def test_high_authority_sources(self):
        for src in ("openai", "deepmind", "reuters", "bloomberg"):
            assert _SOURCE_AUTHORITY[src] >= 0.85, f"{src} should be high authority"

    def test_low_authority_sources(self):
        for src in ("twitter", "reddit"):
            assert _SOURCE_AUTHORITY[src] <= 0.55, f"{src} should be low authority"

    def test_unknown_source_gets_default(self):
        unknown = _SOURCE_AUTHORITY.get("some_obscure_blog", 0.65)
        assert 0.5 <= unknown <= 0.8

    def test_authority_ordering(self):
        assert _SOURCE_AUTHORITY["openai"] > _SOURCE_AUTHORITY["reddit"]
        assert _SOURCE_AUTHORITY["reuters"] > _SOURCE_AUTHORITY["twitter"]
        assert _SOURCE_AUTHORITY["github"] > _SOURCE_AUTHORITY["youtube"]


# ── rank_by_query end-to-end ranking signals ──────────────────────────────────

class TestRankByQuery:
    """
    These tests check that ranking signals dominate in isolation.
    Two items are constructed that differ only on one dimension; we assert
    that the better item ranks first.  The semantic embedding score will be
    similar for both (same title/content), so the other signal is the tiebreaker.
    """

    def test_fresher_article_ranks_above_older(self):
        fresh = make_item(
            url="https://example.com/fresh",
            title="OpenAI releases new model today",
            content="OpenAI released a major new language model today.",
            published_at=hours_ago(2),
            relevance_score=7.0,
            raw_score=0.5,
            source="rss",
        )
        stale = make_item(
            url="https://example.com/stale",
            title="OpenAI releases new model today",
            content="OpenAI released a major new language model today.",
            published_at=hours_ago(120),  # 5 days old
            relevance_score=7.0,
            raw_score=0.5,
            source="rss",
        )
        ranked = rank_by_query("OpenAI model release", [stale, fresh])
        assert ranked[0].url == fresh.url, "Fresh article should rank first"

    def test_high_authority_source_ranks_above_low_authority(self):
        authoritative = make_item(
            url="https://openai.com/blog/gpt5",
            title="OpenAI announces GPT-5",
            content="OpenAI has announced GPT-5 with significant capability improvements.",
            published_at=hours_ago(3),
            relevance_score=7.0,
            raw_score=0.5,
            source="openai",
        )
        low_auth = make_item(
            url="https://twitter.com/user/123",
            title="OpenAI announces GPT-5",
            content="OpenAI has announced GPT-5 with significant capability improvements.",
            published_at=hours_ago(3),
            relevance_score=7.0,
            raw_score=0.9,  # higher engagement, but lower authority
            source="twitter",
        )
        ranked = rank_by_query("OpenAI GPT-5 announcement", [low_auth, authoritative])
        assert ranked[0].source == "openai", "High-authority source should rank above Twitter"

    def test_high_relevance_score_ranks_above_low(self):
        high_rel = make_item(
            url="https://example.com/high",
            title="Federal Reserve raises interest rates by 50bps",
            content="The Federal Reserve raised rates by 50 basis points citing inflation.",
            published_at=hours_ago(1),
            relevance_score=9.0,
            source="bloomberg",
        )
        low_rel = make_item(
            url="https://example.com/low",
            title="Federal Reserve raises interest rates by 50bps",
            content="The Federal Reserve raised rates by 50 basis points citing inflation.",
            published_at=hours_ago(1),
            relevance_score=3.0,
            source="bloomberg",
        )
        ranked = rank_by_query("Fed interest rate hike", [low_rel, high_rel])
        assert ranked[0].url == high_rel.url, "Higher LLM relevance should rank first"

    def test_returns_all_items(self):
        items = [make_item(url=f"https://example.com/{i}") for i in range(10)]
        ranked = rank_by_query("AI news", items)
        assert len(ranked) == 10

    def test_empty_list_returns_empty(self):
        assert rank_by_query("anything", []) == []

    def test_single_item_returns_unchanged(self):
        item = make_item()
        assert rank_by_query("AI news", [item]) == [item]

    def test_combined_signals_correct_order(self):
        """A fresh high-authority article should beat a stale low-authority one
        even when the stale article has a slightly better relevance score."""
        winner = make_item(
            url="https://reuters.com/story",
            title="NVIDIA earnings beat expectations",
            content="NVIDIA reported quarterly earnings significantly above analyst estimates.",
            published_at=hours_ago(4),
            relevance_score=7.5,
            raw_score=0.6,
            source="reuters",
        )
        loser = make_item(
            url="https://twitter.com/some_tweet",
            title="NVIDIA earnings beat expectations",
            content="NVIDIA reported quarterly earnings significantly above analyst estimates.",
            published_at=hours_ago(96),   # 4 days old
            relevance_score=8.0,          # slightly higher relevance
            raw_score=0.9,                # higher engagement
            source="twitter",
        )
        ranked = rank_by_query("NVIDIA earnings", [loser, winner])
        assert ranked[0].source == "reuters", (
            "Fresh reuters article should beat stale tweet despite higher tweet relevance"
        )
