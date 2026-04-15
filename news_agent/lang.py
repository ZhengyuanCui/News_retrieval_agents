"""Lightweight language detection for news items."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ISO 639-1 codes we care about (display name → code)
SUPPORTED_LANGUAGES: dict[str, str] = {
    "English":    "en",
    "Chinese":    "zh",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Japanese":   "ja",
    "Portuguese": "pt",
    "Arabic":     "ar",
    "Korean":     "ko",
    "Hindi":      "hi",
}

# langdetect returns zh-cn / zh-tw — normalise both to "zh"
_NORMALISE = {"zh-cn": "zh", "zh-tw": "zh"}


def detect_language(text: str) -> str:
    """Return ISO 639-1 language code for *text*, defaulting to 'en' on failure."""
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        from langdetect import detect, LangDetectException
        code = detect(text[:500])
        return _NORMALISE.get(code, code)
    except Exception:
        return "en"
