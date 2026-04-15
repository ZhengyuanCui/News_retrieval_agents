from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_MODEL = "mrm8488/bert-tiny-finetuned-sms-spam-detection"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load the spam classifier once and cache it for the process lifetime."""
    try:
        from transformers import pipeline
        logger.info("Loading spam classifier model '%s'…", _MODEL)
        clf = pipeline("text-classification", model=_MODEL, truncation=True, max_length=512)
        logger.info("Spam classifier ready.")
        return clf
    except Exception as e:
        logger.warning("Could not load spam classifier (%s) — falling back to keyword filter only.", e)
        return None


def is_spam_ml(text: str, threshold: float = 0.85) -> bool:
    """
    Return True if the ML model considers the text spam with confidence >= threshold.
    Falls back to False (don't block) if the model is unavailable.
    """
    clf = _get_pipeline()
    if clf is None:
        return False
    try:
        result = clf(text[:512])[0]
        label = result["label"].upper()
        score = result["score"]
        if label == "SPAM" and score >= threshold:
            logger.debug("ML spam (%.2f): %s", score, text[:80])
            return True
        return False
    except Exception as e:
        logger.debug("Spam classifier error: %s", e)
        return False
