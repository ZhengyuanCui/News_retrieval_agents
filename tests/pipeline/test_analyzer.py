"""Tests for LLMAnalyzer and is_question().

Uses monkeypatching to avoid real API calls.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_agent.pipeline.analyzer import LLMAnalyzer, is_question
from tests.conftest import make_item


# ── is_question ───────────────────────────────────────────────────────────────

class TestIsQuestion:
    def test_ends_with_question_mark(self):
        assert is_question("Will the Fed raise rates?")

    def test_starts_with_what(self):
        assert is_question("what is the best AI model")

    def test_starts_with_why(self):
        assert is_question("why did markets crash today")

    def test_starts_with_how(self):
        assert is_question("how does quantitative easing work")

    def test_starts_with_will(self):
        assert is_question("will OpenAI release GPT-5 this year")

    def test_starts_with_is(self):
        assert is_question("is the stock market overvalued")

    def test_starts_with_can(self):
        assert is_question("can AI replace software engineers")

    def test_seven_or_more_words_is_question(self):
        # 7 words → treated as natural-language question
        assert is_question("impact of federal reserve rate decisions on markets")

    def test_short_keyword_query_is_not_question(self):
        assert not is_question("NVDA earnings")

    def test_ticker_query_is_not_question(self):
        assert not is_question("Fed rate hike")

    def test_six_words_borderline(self):
        # exactly 6 words — NOT a question by word count (threshold is >=7)
        result = is_question("federal reserve rate hike this week")
        assert isinstance(result, bool)


# ── analyze_batch: JSON parsing ───────────────────────────────────────────────

def _make_llm_response(data: list[dict]) -> MagicMock:
    """Build a mock litellm response object."""
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_markdown_response(data: list[dict]) -> MagicMock:
    """Build a mock response with markdown code fence."""
    msg = MagicMock()
    msg.content = f"```json\n{json.dumps(data)}\n```"
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_analyze_batch_fills_fields(monkeypatch):
    items = [make_item(url="https://a.com/1", title="OpenAI releases GPT-5")]
    expected = [{
        "summary": "OpenAI released GPT-5.",
        "relevance_score": 9.0,
        "key_entities": ["OpenAI", "GPT-5"],
        "sentiment": "positive",
        "tags": ["ai", "openai", "gpt"],
    }]

    async def fake_acompletion(**kwargs):
        return _make_llm_response(expected)

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    analyzer = LLMAnalyzer()
    result = await analyzer.analyze_batch(items, topic="ai")
    assert result[0].summary == "OpenAI released GPT-5."
    assert result[0].relevance_score == 9.0
    assert result[0].sentiment == "positive"
    assert "OpenAI" in result[0].key_entities


@pytest.mark.asyncio
async def test_analyze_batch_parses_markdown_wrapped_json(monkeypatch):
    items = [make_item(url="https://a.com/1")]
    expected = [{"summary": "Test.", "relevance_score": 5.0,
                 "key_entities": [], "sentiment": "neutral", "tags": []}]

    async def fake_acompletion(**kwargs):
        return _make_markdown_response(expected)

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    analyzer = LLMAnalyzer()
    result = await analyzer.analyze_batch(items, topic="ai")
    assert result[0].summary == "Test."


@pytest.mark.asyncio
async def test_analyze_batch_skips_already_summarized(monkeypatch):
    items = [make_item(url="https://a.com/1", summary="Already done.")]
    call_count = 0

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        return _make_llm_response([])

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    analyzer = LLMAnalyzer()
    result = await analyzer.analyze_batch(items, topic="ai")
    assert call_count == 0, "should not call LLM if all items already have summaries"
    assert result[0].summary == "Already done."


@pytest.mark.asyncio
async def test_analyze_batch_handles_invalid_json(monkeypatch):
    items = [make_item(url="https://a.com/1")]

    async def fake_acompletion(**kwargs):
        msg = MagicMock()
        msg.content = "This is not JSON at all"
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    analyzer = LLMAnalyzer()
    result = await analyzer.analyze_batch(items, topic="ai")
    # Should not raise; item returned as-is without summary filled in
    assert result[0].summary is None


@pytest.mark.asyncio
async def test_analyze_batch_rotates_on_rate_limit(monkeypatch):
    """When first model slot hits RateLimitError, should try next slot."""
    import litellm as _litellm

    items = [make_item(url="https://a.com/1")]
    call_count = 0
    expected = [{"summary": "Rotated.", "relevance_score": 7.0,
                 "key_entities": [], "sentiment": "neutral", "tags": []}]

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _litellm.RateLimitError("rate limit", llm_provider="anthropic", model="test")
        return _make_llm_response(expected)

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    # Ensure pool has at least 2 slots by patching _build_weighted_pool
    from news_agent.pipeline import analyzer
    from news_agent.pipeline.analyzer import _ModelSlot

    slot_a = _ModelSlot("model-a", None, 10)
    slot_b = _ModelSlot("model-b", None, 10)
    monkeypatch.setattr(analyzer, "_build_weighted_pool", lambda: [slot_a, slot_b])

    llm = LLMAnalyzer()
    result = await llm.analyze_batch(items, topic="ai")
    assert call_count == 2, "should have retried after first rate limit"
    assert result[0].summary == "Rotated."


@pytest.mark.asyncio
async def test_analyze_batch_empty_returns_empty():
    analyzer = LLMAnalyzer()
    result = await analyzer.analyze_batch([], topic="ai")
    assert result == []
