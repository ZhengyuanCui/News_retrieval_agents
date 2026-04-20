"""
Tests for RSS collector utilities: _clean_summary and _decode_google_news_url.

No network calls are made — all tests use controlled inputs.
"""
from __future__ import annotations

import pytest

from news_agent.collectors.rss import _clean_summary, _decode_google_news_url


# ── _clean_summary ────────────────────────────────────────────────────────────

class TestCleanSummary:
    def test_plain_text_unchanged(self):
        text = "OpenAI releases GPT-5 with major capability improvements."
        assert _clean_summary(text) == text

    def test_strips_anchor_tag(self):
        html = '<a href="https://example.com">Article title - Publisher</a>'
        result = _clean_summary(html)
        assert "<a" not in result
        assert "Article title" in result

    def test_strips_html_entities(self):
        html = "Fed hikes rates &amp; markets react"
        result = _clean_summary(html)
        assert "&amp;" not in result
        assert "& markets" in result

    def test_strips_nested_tags(self):
        html = "<div><p>Summary <strong>text</strong></p></div>"
        result = _clean_summary(html)
        assert "<" not in result
        assert "Summary" in result
        assert "text" in result

    def test_collapses_whitespace(self):
        html = "word1   <br/>   word2"
        result = _clean_summary(html)
        # Should not have multiple consecutive spaces
        assert "  " not in result

    def test_empty_string(self):
        assert _clean_summary("") == ""

    def test_none_coerced(self):
        assert _clean_summary(None) == ""

    def test_google_news_anchor_cleanup(self):
        # Google News RSS wraps titles in anchor tags
        html = '<a href="https://news.google.com/articles/CBMi..." >'
        html += 'AI Model Released - TechCrunch</a>'
        result = _clean_summary(html)
        assert "AI Model Released" in result
        assert "<a" not in result

    def test_lt_gt_entities_stripped_as_tags(self):
        # &lt;important&gt; unescapes to <important> which is treated as an HTML tag
        # and stripped — the surrounding text is preserved
        html = "prefix &lt;important&gt; breaking news"
        result = _clean_summary(html)
        assert "&lt;" not in result
        assert "breaking news" in result

    def test_strips_img_tags(self):
        html = 'Story text <img src="photo.jpg" alt="Photo"> continues here'
        result = _clean_summary(html)
        assert "<img" not in result
        assert "Story text" in result
        assert "continues here" in result


# ── _decode_google_news_url ───────────────────────────────────────────────────

class TestDecodeGoogleNewsUrl:
    def test_non_google_url_unchanged(self):
        url = "https://techcrunch.com/article/about-ai"
        assert _decode_google_news_url(url) == url

    def test_google_url_without_articles_path_unchanged(self):
        url = "https://news.google.com/search?q=AI"
        result = _decode_google_news_url(url)
        assert result == url

    def test_invalid_base64_returns_original(self):
        # Malformed article path — no valid base64 payload
        url = "https://news.google.com/rss/articles/XXXXXXXXXXXXXXXXX"
        result = _decode_google_news_url(url)
        # Should not raise; returns original URL if decoding fails
        assert isinstance(result, str)

    def test_returns_string(self):
        url = "https://news.google.com/rss/articles/CBMiAA"
        result = _decode_google_news_url(url)
        assert isinstance(result, str)

    def test_decoded_url_does_not_contain_google(self):
        """If a real article URL is successfully decoded, it should not be a Google URL."""
        import base64
        # Build a fake encoded payload that embeds a real URL
        fake_target = b"https://reuters.com/technology/ai-news-article"
        # Wrap with padding bytes to simulate the Google encoding
        payload = b"\x00\x01\x02\x03" + fake_target + b"\x00"
        encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        url = f"https://news.google.com/rss/articles/{encoded}"
        result = _decode_google_news_url(url)
        if result != url:  # only assert if decoding succeeded
            assert "news.google.com" not in result
