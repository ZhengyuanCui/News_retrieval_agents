"""Tests for lang.detect_language()."""
from __future__ import annotations

import pytest

from news_agent.lang import detect_language, SUPPORTED_LANGUAGES


class TestDetectLanguage:
    def test_english_text(self):
        assert detect_language("The Federal Reserve raised interest rates today amid inflation concerns.") == "en"

    def test_short_text_defaults_to_english(self):
        # Less than 10 chars → always returns "en"
        assert detect_language("Hi") == "en"
        assert detect_language("abc") == "en"

    def test_empty_string_defaults_to_english(self):
        assert detect_language("") == "en"

    def test_whitespace_only_defaults_to_english(self):
        assert detect_language("   ") == "en"

    def test_exception_defaults_to_english(self, monkeypatch):
        """If langdetect raises, should fall back to 'en'."""
        langdetect = pytest.importorskip("langdetect")
        monkeypatch.setattr(langdetect, "detect", lambda t: (_ for _ in ()).throw(Exception("fail")))
        result = detect_language("This is a long enough English text to trigger detection.")
        assert result == "en"

    def test_normalises_zh_cn_to_zh(self, monkeypatch):
        langdetect = pytest.importorskip("langdetect")
        monkeypatch.setattr(langdetect, "detect", lambda t: "zh-cn")
        result = detect_language("这是一个关于人工智能的新闻。")
        assert result == "zh"

    def test_normalises_zh_tw_to_zh(self, monkeypatch):
        langdetect = pytest.importorskip("langdetect")
        monkeypatch.setattr(langdetect, "detect", lambda t: "zh-tw")
        result = detect_language("這是一個關於人工智能的新聞。")
        assert result == "zh"

    def test_uses_only_first_500_chars(self, monkeypatch):
        """detect_language should pass at most 500 chars to langdetect."""
        langdetect = pytest.importorskip("langdetect")
        received = []
        def spy_detect(text):
            received.append(len(text))
            return "en"
        monkeypatch.setattr(langdetect, "detect", spy_detect)
        long_text = "a" * 1000
        detect_language(long_text)
        assert received[0] <= 500


class TestSupportedLanguages:
    def test_english_in_supported(self):
        assert "en" in SUPPORTED_LANGUAGES.values()

    def test_chinese_in_supported(self):
        assert "zh" in SUPPORTED_LANGUAGES.values()

    def test_supported_contains_ten_languages(self):
        assert len(SUPPORTED_LANGUAGES) == 10
