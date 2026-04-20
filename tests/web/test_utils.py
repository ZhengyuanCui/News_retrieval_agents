"""
Unit tests for web/app.py utility functions:
  _parse_digest, _parse_languages, _strip_html, _bold_md
"""
from __future__ import annotations

import pytest

from news_agent.web.app import _parse_digest, _parse_languages, _strip_html, _bold_md


# ── _parse_digest ─────────────────────────────────────────────────────────────

class TestParseDigest:
    def test_none_returns_none(self):
        assert _parse_digest(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_digest("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_digest("   \n  ") is None

    def test_headline_only(self):
        result = _parse_digest("OpenAI releases GPT-5")
        assert result is not None
        assert result["headline"] == "OpenAI releases GPT-5"
        assert result["bullets"] == []

    def test_headline_and_bullets(self):
        raw = "Markets rally on Fed pivot\nStocks up 2%\nBonds also rise\nDollar weakens"
        result = _parse_digest(raw)
        assert result["headline"] == "Markets rally on Fed pivot"
        assert len(result["bullets"]) == 3
        assert "Stocks up 2%" in result["bullets"]

    def test_legacy_pipe_format(self):
        raw = "Top AI story|||GPT-5 released|||Claude 4 competes"
        result = _parse_digest(raw)
        assert result["headline"] == "Top AI story"
        assert "GPT-5 released" in result["bullets"]
        assert "Claude 4 competes" in result["bullets"]

    def test_legacy_format_empty_bullets_filtered(self):
        raw = "Headline|||"
        result = _parse_digest(raw)
        assert result["headline"] == "Headline"
        assert result["bullets"] == []

    def test_blank_lines_ignored(self):
        raw = "Big headline\n\nBullet one\n\nBullet two"
        result = _parse_digest(raw)
        assert result["headline"] == "Big headline"
        assert len(result["bullets"]) == 2


# ── _parse_languages ──────────────────────────────────────────────────────────

class TestParseLanguages:
    def test_empty_string_returns_none(self):
        assert _parse_languages("") is None

    def test_none_returns_none(self):
        assert _parse_languages(None) is None

    def test_all_returns_none(self):
        assert _parse_languages("all") is None
        assert _parse_languages("ALL") is None

    def test_single_language(self):
        result = _parse_languages("en")
        assert result == ["en"]

    def test_multiple_languages(self):
        result = _parse_languages("en,zh,fr")
        assert set(result) == {"en", "zh", "fr"}

    def test_strips_whitespace(self):
        result = _parse_languages("en , zh , de")
        assert "en" in result
        assert "zh" in result
        assert "de" in result

    def test_empty_tokens_filtered(self):
        result = _parse_languages("en,,zh")
        assert "" not in result
        assert len(result) == 2


# ── _strip_html ───────────────────────────────────────────────────────────────

class TestStripHtml:
    def test_plain_text_unchanged(self):
        assert _strip_html("Hello world") == "Hello world"

    def test_strips_anchor_tags(self):
        result = _strip_html('<a href="https://example.com">Click here</a>')
        assert "<a" not in result
        assert "Click here" in result

    def test_unescapes_entities(self):
        result = _strip_html("&amp; &lt;tag&gt;")
        assert "&amp;" not in result
        assert "& <tag>" in result or "&" in result

    def test_collapses_whitespace(self):
        result = _strip_html("word1  <br/>  word2")
        assert "  " not in result

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_none_coerced_to_empty(self):
        assert _strip_html(None) == ""

    def test_nested_tags(self):
        html = "<div><p><strong>Important</strong> text</p></div>"
        result = _strip_html(html)
        assert "Important" in result
        assert "<" not in result


# ── _bold_md ──────────────────────────────────────────────────────────────────

class TestBoldMd:
    def test_converts_bold_markdown(self):
        from markupsafe import Markup
        result = _bold_md("This is **important** text")
        assert "<strong>important</strong>" in str(result)
        assert "**" not in str(result)

    def test_no_markdown_passes_through(self):
        result = _bold_md("Plain text here")
        assert "Plain text here" in str(result)

    def test_multiple_bold_spans(self):
        result = _bold_md("**a** and **b**")
        assert str(result).count("<strong>") == 2

    def test_xss_escaped(self):
        result = _bold_md('<script>alert("xss")</script>')
        assert "<script>" not in str(result)

    def test_returns_markup_safe(self):
        from markupsafe import Markup
        result = _bold_md("text")
        assert isinstance(result, Markup)
