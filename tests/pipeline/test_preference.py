"""
Tests for preference.py: interaction weights, preference recomputation,
apply_preference_boost, and _sigmoid_boost.
"""
from __future__ import annotations

import math
import pytest

from news_agent.preference import (
    WEIGHTS,
    _interaction_weight,
    _sigmoid_boost,
    apply_preference_boost,
    get_preference_scores,
    record_interaction,
    recompute_preferences,
)
from tests.conftest import make_item


# ── WEIGHTS ───────────────────────────────────────────────────────────────────

class TestWeights:
    def test_upvote_is_positive(self):
        assert WEIGHTS["upvote"] > 0

    def test_unupvote_is_negative(self):
        assert WEIGHTS["unupvote"] < 0

    def test_downvote_is_negative(self):
        assert WEIGHTS["downvote"] < 0

    def test_undownvote_is_positive(self):
        assert WEIGHTS["undownvote"] > 0

    def test_downvote_stronger_than_unupvote(self):
        # Downvote is a stronger negative signal than cancelling an upvote
        assert abs(WEIGHTS["downvote"]) > abs(WEIGHTS["unupvote"])

    def test_click_is_small_positive(self):
        assert 0 < WEIGHTS["click"] < WEIGHTS["upvote"]

    def test_read_long_stronger_than_read_short(self):
        assert WEIGHTS["read_long"] > WEIGHTS["read_medium"] > WEIGHTS["read_short"] > 0


# ── _interaction_weight ───────────────────────────────────────────────────────

class TestInteractionWeight:
    def _make_interaction(self, action, read_seconds=None):
        from types import SimpleNamespace
        return SimpleNamespace(action=action, read_seconds=read_seconds)

    def test_upvote(self):
        assert _interaction_weight(self._make_interaction("upvote")) == WEIGHTS["upvote"]

    def test_unupvote(self):
        assert _interaction_weight(self._make_interaction("unupvote")) == WEIGHTS["unupvote"]

    def test_downvote(self):
        assert _interaction_weight(self._make_interaction("downvote")) == WEIGHTS["downvote"]

    def test_undownvote(self):
        assert _interaction_weight(self._make_interaction("undownvote")) == WEIGHTS["undownvote"]

    def test_click(self):
        assert _interaction_weight(self._make_interaction("click")) == WEIGHTS["click"]

    def test_read_short(self):
        w = _interaction_weight(self._make_interaction("read", read_seconds=30))
        assert w == WEIGHTS["read_short"]

    def test_read_medium(self):
        w = _interaction_weight(self._make_interaction("read", read_seconds=90))
        assert w == WEIGHTS["read_medium"]

    def test_read_long(self):
        w = _interaction_weight(self._make_interaction("read", read_seconds=300))
        assert w == WEIGHTS["read_long"]

    def test_read_too_short_returns_zero(self):
        w = _interaction_weight(self._make_interaction("read", read_seconds=5))
        assert w == 0.0

    def test_read_no_seconds_returns_zero(self):
        w = _interaction_weight(self._make_interaction("read", read_seconds=None))
        assert w == 0.0

    def test_unknown_action_returns_zero(self):
        w = _interaction_weight(self._make_interaction("unknown_action"))
        assert w == 0.0


# ── _sigmoid_boost ────────────────────────────────────────────────────────────

class TestSigmoidBoost:
    def test_zero_score_gives_zero_boost(self):
        assert _sigmoid_boost(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_positive_score_gives_positive_boost(self):
        assert _sigmoid_boost(5.0) > 0

    def test_negative_score_gives_negative_boost(self):
        assert _sigmoid_boost(-5.0) < 0

    def test_boost_bounded_at_most_one(self):
        assert abs(_sigmoid_boost(1000.0)) <= 1.0

    def test_boost_is_antisymmetric(self):
        # f(x) = -f(-x)
        assert _sigmoid_boost(3.0) == pytest.approx(-_sigmoid_boost(-3.0), abs=1e-9)

    def test_larger_scale_amplifies_boost(self):
        assert abs(_sigmoid_boost(2.0, scale=1.0)) > abs(_sigmoid_boost(2.0, scale=0.1))


# ── apply_preference_boost ────────────────────────────────────────────────────

class TestApplyPreferenceBoost:
    def test_empty_prefs_returns_items_unchanged(self):
        items = [make_item(relevance_score=5.0), make_item(url="https://x.com/b", relevance_score=7.0)]
        result = apply_preference_boost(items, prefs={})
        assert result is items  # same list object

    def test_positive_tag_pref_raises_relevance(self):
        item = make_item(
            url="https://example.com/ai-item",
            tags=["ai", "openai"],
            relevance_score=5.0,
        )
        prefs = {("tag", "ai"): 10.0, ("tag", "openai"): 10.0}
        result = apply_preference_boost([item], prefs)
        assert result[0].relevance_score > 5.0

    def test_negative_source_pref_lowers_relevance(self):
        item = make_item(
            url="https://example.com/reddit-item",
            source="reddit",
            relevance_score=7.0,
        )
        prefs = {("source", "reddit"): -20.0}
        result = apply_preference_boost([item], prefs)
        assert result[0].relevance_score < 7.0

    def test_relevance_score_clamped_at_ten(self):
        item = make_item(url="https://example.com/clamp-high", tags=["ai"], relevance_score=9.5)
        prefs = {("tag", "ai"): 1000.0}
        result = apply_preference_boost([item], prefs)
        assert result[0].relevance_score <= 10.0

    def test_relevance_score_clamped_at_zero(self):
        item = make_item(url="https://example.com/clamp-low", tags=["spam"], relevance_score=0.5)
        prefs = {("tag", "spam"): -1000.0}
        result = apply_preference_boost([item], prefs)
        assert result[0].relevance_score >= 0.0

    def test_none_relevance_score_not_modified(self):
        item = make_item(url="https://example.com/no-score", tags=["ai"], relevance_score=None)
        prefs = {("tag", "ai"): 10.0}
        result = apply_preference_boost([item], prefs)
        assert result[0].relevance_score is None


# ── record_interaction + recompute_preferences (DB integration) ───────────────

@pytest.mark.asyncio
async def test_record_interaction_persists_to_db():
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository
    from news_agent.models import UserInteractionORM
    from sqlalchemy import select

    item = make_item(url="https://example.com/pref-record-test")
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        await record_interaction(session, item.id, "upvote")

    async with get_session() as session:
        result = await session.execute(
            select(UserInteractionORM).where(
                UserInteractionORM.item_id == item.id,
                UserInteractionORM.action == "upvote",
            )
        )
        assert result.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_recompute_preferences_upvote_boosts_source():
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(
        url="https://example.com/pref-source-test",
        source="bloomberg",
        tags=["finance"],
        key_entities=["Goldman Sachs"],
        relevance_score=7.0,
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        item.tags = ["finance"]
        item.key_entities = ["Goldman Sachs"]
        await repo.update_analysis_many([item])
        await record_interaction(session, item.id, "upvote")
        await recompute_preferences(session)
        prefs = await get_preference_scores(session)

    assert prefs.get(("source", "bloomberg"), 0.0) > 0
    assert prefs.get(("tag", "finance"), 0.0) > 0
    assert prefs.get(("entity", "goldman sachs"), 0.0) > 0


@pytest.mark.asyncio
async def test_recompute_preferences_downvote_demotes_source():
    from news_agent.storage.database import get_session
    from news_agent.storage.repository import NewsRepository

    item = make_item(
        url="https://example.com/pref-demote-test",
        source="reddit",
        tags=["crypto"],
        relevance_score=6.0,
    )
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert(item)
        item.tags = ["crypto"]
        await repo.update_analysis_many([item])
        await record_interaction(session, item.id, "downvote")
        await recompute_preferences(session)
        prefs = await get_preference_scores(session)

    assert prefs.get(("source", "reddit"), 0.0) < 0
    assert prefs.get(("tag", "crypto"), 0.0) < 0


@pytest.mark.asyncio
async def test_recompute_preferences_no_interactions_is_noop():
    from news_agent.storage.database import get_session

    async with get_session() as session:
        await recompute_preferences(session)
        prefs = await get_preference_scores(session)
    assert prefs == {}
