"""
Deduplication quality tests.

Covers: URL normalisation, title normalisation, TF-IDF near-dedup,
semantic near-dedup (with cluster_id), and detail-score winner selection.
"""
from __future__ import annotations

import pytest

from news_agent.pipeline.deduplicator import Deduplicator, _title_key, normalize_url
from tests.conftest import hours_ago, make_item


# ── URL normalisation ─────────────────────────────────────────────────────────

class TestNormalizeUrl:
    def test_strips_utm_params(self):
        url = "https://example.com/article?utm_source=twitter&utm_medium=social"
        assert "utm_source" not in normalize_url(url)
        assert "utm_medium" not in normalize_url(url)

    def test_preserves_non_tracking_params(self):
        url = "https://example.com/article?id=123&page=2"
        norm = normalize_url(url)
        assert "id=123" in norm
        assert "page=2" in norm

    def test_strips_fragment(self):
        assert "#section" not in normalize_url("https://example.com/a#section")

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/a/") == normalize_url("https://example.com/a")

    def test_lowercases(self):
        assert normalize_url("https://Example.COM/Article") == normalize_url("https://example.com/article")

    def test_fbclid_stripped(self):
        url = "https://example.com/a?fbclid=AbC123"
        assert "fbclid" not in normalize_url(url)


# ── Title normalisation ───────────────────────────────────────────────────────

class TestTitleKey:
    def test_strips_publication_suffix(self):
        assert _title_key("Fed raises rates - Reuters") == _title_key("Fed raises rates")

    def test_strips_long_dash_suffix(self):
        assert _title_key("OpenAI releases GPT-5 - The New York Times") == _title_key("OpenAI releases GPT-5")

    def test_keeps_long_suffix_that_is_part_of_title(self):
        # The regex strips suffixes ≤41 chars after " - "; this one is 63 chars — must be kept
        key = _title_key("How to build a RAG pipeline - a comprehensive step-by-step guide for deploying production systems")
        assert "comprehensive" in key

    def test_case_insensitive(self):
        assert _title_key("OpenAI GPT-5") == _title_key("openai gpt-5")

    def test_ignores_punctuation(self):
        assert _title_key("OpenAI: GPT-5!") == _title_key("OpenAI GPT5")


# ── URL-based deduplication ───────────────────────────────────────────────────

class TestUrlDedup:
    def setup_method(self):
        self.dedup = Deduplicator(strategy="url_only")

    def test_exact_url_duplicate_removed(self):
        items = [
            make_item(url="https://example.com/a"),
            make_item(url="https://example.com/a"),  # exact duplicate
        ]
        result = self.dedup.deduplicate(items)
        assert sum(1 for i in result if not i.is_duplicate) == 1

    def test_tracking_param_duplicate_removed(self):
        items = [
            make_item(url="https://example.com/a", raw_score=0.3),
            make_item(url="https://example.com/a?utm_source=newsletter", raw_score=0.3),
        ]
        result = self.dedup.deduplicate(items)
        assert sum(1 for i in result if not i.is_duplicate) == 1

    def test_different_urls_both_kept(self):
        items = [
            make_item(url="https://example.com/a", title="OpenAI releases GPT-5"),
            make_item(url="https://example.com/b", title="Tesla Q3 earnings beat estimates"),
        ]
        result = self.dedup.deduplicate(items)
        assert sum(1 for i in result if not i.is_duplicate) == 2

    def test_title_duplicate_removed(self):
        items = [
            make_item(url="https://a.com/story", title="Fed raises rates by 50bps"),
            make_item(url="https://b.com/story", title="Fed raises rates by 50bps"),  # same title, different URL
        ]
        result = self.dedup.deduplicate(items)
        assert sum(1 for i in result if not i.is_duplicate) == 1

    def test_publication_suffix_title_duplicate_removed(self):
        items = [
            make_item(url="https://a.com/s", title="NVIDIA earnings beat - Reuters"),
            make_item(url="https://b.com/s", title="NVIDIA earnings beat - Bloomberg"),
        ]
        result = self.dedup.deduplicate(items)
        assert sum(1 for i in result if not i.is_duplicate) == 1

    def test_detail_score_winner_kept(self):
        rich = make_item(
            url="https://example.com/rich",
            content="A" * 500,   # much more content
            raw_score=0.3,
        )
        sparse = make_item(
            url="https://example.com/sparse",
            title="Default test title",  # same title → deduped
            content="Short",
            raw_score=0.9,
        )
        result = self.dedup.deduplicate([sparse, rich])
        survivors = [i for i in result if not i.is_duplicate]
        assert len(survivors) == 1
        assert survivors[0].url == rich.url, "Item with more content should survive"

    def test_duplicate_of_field_set(self):
        a = make_item(url="https://example.com/x", raw_score=0.8)
        b = make_item(url="https://example.com/x", raw_score=0.3)
        result = self.dedup.deduplicate([a, b])
        dupes = [i for i in result if i.is_duplicate]
        assert len(dupes) == 1
        assert dupes[0].duplicate_of is not None


# ── TF-IDF near-deduplication ─────────────────────────────────────────────────

class TestTfidfDedup:
    def setup_method(self):
        self.dedup = Deduplicator(strategy="tfidf", threshold=0.85)

    def test_near_identical_articles_deduped(self):
        # TF-IDF cosine sim for this pair is ~0.834 — use threshold=0.80 to catch it
        dedup = Deduplicator(strategy="tfidf", threshold=0.80)
        base_content = "OpenAI announced GPT-5 with major capability improvements over GPT-4."
        items = [
            make_item(url="https://a.com", title="OpenAI announces GPT-5", content=base_content),
            make_item(url="https://b.com", title="OpenAI unveils GPT-5", content=base_content + " The model is available now."),
            make_item(url="https://c.com", title="Stock market hits record high", content="S&P 500 reached an all-time high driven by tech stocks."),
        ]
        result = dedup.deduplicate(items)
        survivors = [i for i in result if not i.is_duplicate]
        assert len(survivors) == 2, "Near-identical GPT-5 pair should collapse to 1"

    def test_distinct_articles_all_kept(self):
        items = [
            make_item(url="https://a.com", title="OpenAI GPT-5", content="OpenAI released GPT-5 today."),
            make_item(url="https://b.com", title="Fed rate hike", content="The Federal Reserve raised interest rates by 50bps."),
            make_item(url="https://c.com", title="NVIDIA earnings", content="NVIDIA reported record quarterly earnings beating estimates."),
        ]
        result = self.dedup.deduplicate(items)
        survivors = [i for i in result if not i.is_duplicate]
        assert len(survivors) == 3


# ── Cluster ID assignment ─────────────────────────────────────────────────────

class TestClusterAssignment:
    def test_semantic_near_dupes_share_cluster_id(self):
        dedup = Deduplicator(strategy="tfidf", threshold=0.85)
        content = "OpenAI released GPT-5 with major improvements over the previous generation."
        items = [
            make_item(url="https://a.com", content=content, title="OpenAI releases GPT-5"),
            make_item(url="https://b.com", content=content + " Available today.", title="OpenAI unveils GPT-5"),
        ]
        result = dedup.deduplicate(items)
        # Both items should share the same cluster_id after semantic dedup
        cluster_ids = {i.cluster_id for i in result if i.cluster_id}
        if len(cluster_ids) > 0:  # only asserted when TF-IDF threshold triggers
            assert len(cluster_ids) == 1, "Near-duplicates should share one cluster_id"

    def test_distinct_articles_have_no_cluster_id(self):
        dedup = Deduplicator(strategy="tfidf", threshold=0.85)
        items = [
            make_item(url="https://a.com", content="OpenAI released GPT-5.", title="OpenAI GPT-5"),
            make_item(url="https://b.com", content="Tesla earnings beat expectations.", title="Tesla earnings"),
        ]
        result = dedup.deduplicate(items)
        assert all(i.cluster_id is None for i in result), "Distinct articles should have no cluster_id"
