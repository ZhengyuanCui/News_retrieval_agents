"""
Search quality tests: BM25 exact matching, RRF merging, and hybrid search.

These tests run against a real in-memory SQLite database (no mocking) so they
verify the full stack from query → FTS5 → vector index → RRF → ranked results.
All tests are self-contained: they insert their own data and clean up after.
"""
from __future__ import annotations

import pytest
from datetime import datetime

from news_agent.storage.repository import _fts_escape, _rrf_merge
from tests.conftest import hours_ago, make_item


# ── _fts_escape ───────────────────────────────────────────────────────────────

class TestFtsEscape:
    def test_single_word(self):
        assert _fts_escape("nvidia") == '"nvidia"'

    def test_multi_word(self):
        result = _fts_escape("fed rate hike")
        assert result == '"fed" "rate" "hike"'

    def test_strips_fts5_operators(self):
        result = _fts_escape("nvidia OR AMD -intel")
        # All tokens are quoted so FTS5 boolean operators are neutralised.
        # "OR" becomes the literal '"OR"', not an unquoted operator.
        # The hyphen in "-intel" is stripped; "intel" is quoted as a normal term.
        assert result == '"nvidia" "OR" "AMD" "intel"'

    def test_empty_string(self):
        assert _fts_escape("") == '""'

    def test_ticker_symbol(self):
        assert _fts_escape("NVDA") == '"NVDA"'

    def test_hyphenated_term(self):
        result = _fts_escape("GPT-5")
        assert '"GPT"' in result
        assert '"5"' in result


# ── _rrf_merge ────────────────────────────────────────────────────────────────

class TestRrfMerge:
    def test_id_in_both_lists_ranks_above_id_in_one(self):
        bm25 = ["a", "b", "c"]
        vec  = ["b", "d", "e"]
        merged = _rrf_merge([bm25, vec])
        # "b" appears in both → should be top
        assert merged[0] == "b"

    def test_top_ranked_in_both_lists_wins(self):
        merged = _rrf_merge([["x", "y", "z"], ["x", "y", "z"]])
        assert merged[0] == "x"

    def test_all_ids_present(self):
        bm25 = ["a", "b"]
        vec  = ["c", "d"]
        merged = _rrf_merge([bm25, vec])
        assert set(merged) == {"a", "b", "c", "d"}

    def test_empty_lists_ignored(self):
        merged = _rrf_merge([[], ["a", "b"]])
        assert merged == ["a", "b"]

    def test_both_empty(self):
        assert _rrf_merge([[], []]) == []

    def test_single_list_preserves_order(self):
        lst = ["a", "b", "c", "d"]
        assert _rrf_merge([lst]) == lst

    def test_rrf_constant_k_smooths_rank_differences(self):
        # With k=60, rank-0 and rank-1 scores are close: 1/61 vs 1/62
        # An ID at rank-0 in both lists should beat one at rank-0 in only one
        merged = _rrf_merge([["shared", "only_bm25"], ["shared", "only_vec"]])
        assert merged[0] == "shared"


# ── DB integration: BM25 + hybrid search ─────────────────────────────────────

@pytest.mark.asyncio
async def test_bm25_finds_exact_ticker():
    """BM25 should return an article containing the ticker 'NVDA' for that query."""
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    await init_db()
    item = make_item(
        source="bloomberg",
        url="https://bloomberg.com/nvda-earnings",
        title="NVDA surges after earnings beat",
        content="NVDA reported earnings above analyst expectations. The stock rose 8%.",
        published_at=hours_ago(2),
    )
    decoy = make_item(
        url="https://example.com/unrelated",
        title="Apple announces new iPhone model",
        content="Apple unveiled the new iPhone with improved camera performance.",
        published_at=hours_ago(2),
    )

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        await repo.upsert(decoy)
        ids = await repo.bm25_search("NVDA", limit=10)

    assert item.id in ids, "NVDA article should appear in BM25 results for 'NVDA' query"


@pytest.mark.asyncio
async def test_bm25_finds_multi_word_phrase():
    """BM25 should surface articles containing all query words."""
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    await init_db()
    target = make_item(
        url="https://reuters.com/fed-rate",
        title="Federal Reserve raises interest rates by 50bps",
        content="The Federal Reserve voted to raise interest rates by 50 basis points.",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(target)
        ids = await repo.bm25_search("Federal Reserve interest rates", limit=20)

    assert target.id in ids


@pytest.mark.asyncio
async def test_upsert_syncs_to_fts():
    """After upserting an article, BM25 should find it immediately."""
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    await init_db()
    item = make_item(
        url="https://techcrunch.com/openai-o3",
        title="OpenAI releases o3 reasoning model",
        content="OpenAI today announced o3, a frontier reasoning model.",
        published_at=hours_ago(1),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        ids = await repo.bm25_search("o3 reasoning model", limit=10)

    assert item.id in ids, "Freshly upserted article must be findable via BM25"


@pytest.mark.asyncio
async def test_search_returns_relevant_over_irrelevant():
    """The hybrid search() should rank a directly relevant article above an
    unrelated one when querying by a specific topic keyword."""
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    await init_db()
    relevant = make_item(
        url="https://example.com/claude-sonnet",
        title="Anthropic launches Claude Sonnet with improved reasoning",
        content="Anthropic released Claude Sonnet, its latest AI model with enhanced reasoning and coding capabilities.",
        topic="anthropic",
        published_at=hours_ago(2),
        relevance_score=8.0,
    )
    irrelevant = make_item(
        url="https://example.com/recipe",
        title="Best pasta recipes of 2025",
        content="Here are the top pasta recipes trending this year.",
        topic="food",
        published_at=hours_ago(1),
        relevance_score=8.0,
    )

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(relevant)
        await repo.upsert(irrelevant)
        results = await repo.search("Anthropic Claude model", hours=24, limit=20)

    result_ids = [r.id for r in results]
    assert relevant.id in result_ids, "Relevant article should appear in search results"
    if irrelevant.id in result_ids:
        rel_rank = result_ids.index(relevant.id)
        irr_rank = result_ids.index(irrelevant.id)
        assert rel_rank < irr_rank, "Relevant article should rank above irrelevant one"


@pytest.mark.asyncio
async def test_search_respects_hours_window():
    """Articles older than the requested hours window should not appear in results."""
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    await init_db()
    recent = make_item(
        url="https://example.com/recent-ai",
        title="Recent AI developments in 2025",
        content="Recent breakthroughs in AI have accelerated dramatically.",
        topic="ai",
        published_at=hours_ago(6),
    )
    old = make_item(
        url="https://example.com/old-ai",
        title="Recent AI developments in 2025",  # same title to ensure same topic match
        content="Recent breakthroughs in AI have accelerated dramatically.",
        topic="ai",
        published_at=hours_ago(200),  # well outside any reasonable window
    )

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(recent)
        await repo.upsert(old)
        results = await repo.search("AI developments", hours=24, limit=20)

    result_ids = [r.id for r in results]
    assert recent.id in result_ids, "Recent article should be in 24h window"
    assert old.id not in result_ids, "200h-old article should be outside 24h window"


# ── hybrid_alpha knob (T1-C) ──────────────────────────────────────────────────

class TestRrfMergeWeights:
    def test_weights_none_matches_legacy(self):
        """No weights argument must reproduce the pre-T1-C behaviour exactly."""
        a = ["x", "y", "z"]
        b = ["y", "z", "w"]
        assert _rrf_merge([a, b]) == _rrf_merge([a, b], weights=None)

    def test_equal_weights_match_none(self):
        """Equal non-zero weights are a constant factor on all scores → same order."""
        a = ["x", "y", "z"]
        b = ["y", "w", "x"]
        assert _rrf_merge([a, b], weights=[1.0, 1.0]) == _rrf_merge([a, b])
        assert _rrf_merge([a, b], weights=[0.5, 0.5]) == _rrf_merge([a, b])

    def test_alpha_one_is_bm25_only(self):
        """weights=[1, 0] must yield BM25's order, ignoring the vector list."""
        bm25 = ["bm1", "bm2", "bm3"]
        vec  = ["vec1", "vec2", "vec3"]
        merged = _rrf_merge([bm25, vec], weights=[1.0, 0.0])
        assert merged == bm25

    def test_alpha_zero_is_semantic_only(self):
        """weights=[0, 1] must yield the semantic order, ignoring BM25."""
        bm25 = ["bm1", "bm2", "bm3"]
        vec  = ["vec1", "vec2", "vec3"]
        merged = _rrf_merge([bm25, vec], weights=[0.0, 1.0])
        assert merged == vec

    def test_weights_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _rrf_merge([["a"], ["b"]], weights=[1.0])
        with pytest.raises(ValueError):
            _rrf_merge([["a"], ["b"]], weights=[1.0, 0.5, 0.3])

    def test_skewed_weights_promote_one_list(self):
        """The top-ranked id of a heavily-weighted list must outrank the top-ranked
        id of a lightly-weighted list when they are disjoint.

        (We deliberately don't assert merged[0] here because a "shared" id at
        rank 1 of both lists can still edge out the weighted leader; that is
        the expected fused-ranking behaviour. The property this test pins down
        is a strict comparison between the two list leaders.)
        """
        bm25 = ["heavy_top", "bm_filler_1", "bm_filler_2"]
        vec  = ["vec_top", "vec_filler_1", "vec_filler_2"]
        merged = _rrf_merge([bm25, vec], weights=[0.9, 0.1])
        assert merged.index("heavy_top") < merged.index("vec_top")
        merged_flipped = _rrf_merge([bm25, vec], weights=[0.1, 0.9])
        assert merged_flipped.index("vec_top") < merged_flipped.index("heavy_top")


@pytest.mark.asyncio
async def test_search_alpha_out_of_range_is_clamped(monkeypatch):
    """search(hybrid_alpha=...) must clamp to [0, 1] silently, not raise."""
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    await init_db()
    item = make_item(
        url="https://example.com/clamp-test",
        title="NVDA surges on strong guidance",
        content="NVDA posted a blowout quarter and raised guidance.",
        published_at=hours_ago(2),
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        # Out-of-range values must not raise and must still return sane results.
        r_hi = await repo.search("NVDA", hours=24, limit=10, hybrid_alpha=5.0)
        r_lo = await repo.search("NVDA", hours=24, limit=10, hybrid_alpha=-3.0)
        r_one = await repo.search("NVDA", hours=24, limit=10, hybrid_alpha=1.0)
        r_zero = await repo.search("NVDA", hours=24, limit=10, hybrid_alpha=0.0)

    # Clamped: 5.0 → 1.0, -3.0 → 0.0, so the result sets should match the
    # already-clamped boundary calls on the same index state.
    assert [i.id for i in r_hi] == [i.id for i in r_one]
    assert [i.id for i in r_lo] == [i.id for i in r_zero]


@pytest.mark.asyncio
async def test_search_alpha_none_uses_default(monkeypatch):
    """hybrid_alpha=None must fall back to settings.default_hybrid_alpha and
    produce the same ranking as passing that value explicitly."""
    from news_agent.config import settings
    from news_agent.storage.database import get_session, init_db
    from news_agent.storage.repository import NewsRepository

    monkeypatch.setattr(settings, "default_hybrid_alpha", 0.5)

    await init_db()
    items = [
        make_item(
            url=f"https://example.com/alpha-none-{i}",
            title=f"Claude Sonnet update {i}",
            content="Anthropic released another Claude Sonnet revision with tweaks.",
            published_at=hours_ago(i + 1),
        )
        for i in range(3)
    ]
    async with get_session() as session:
        repo = NewsRepository(session)
        for it in items:
            await repo.upsert(it)
        r_default = await repo.search("Claude Sonnet", hours=24, limit=10)
        r_explicit = await repo.search(
            "Claude Sonnet", hours=24, limit=10, hybrid_alpha=0.5
        )

    assert [i.id for i in r_default] == [i.id for i in r_explicit]
