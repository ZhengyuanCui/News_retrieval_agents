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


def warmup() -> None:
    """Pre-load the model so the first real request doesn't pay the loading cost."""
    _get_pipeline()


def is_spam_ml(text: str, threshold: float = 0.85) -> bool:
    """Single-item check. Prefer is_spam_ml_batch() for multiple texts."""
    results = is_spam_ml_batch([text], threshold=threshold)
    return results[0]


def is_spam_ml_batch(texts: list[str], threshold: float = 0.80) -> list[bool]:
    """
    Classify a batch of texts in one forward pass — much faster than one at a time.
    Returns a list of booleans (True = spam) in the same order as input.
    Falls back to all-False if the model is unavailable.
    """
    if not texts:
        return []
    clf = _get_pipeline()
    if clf is None:
        return [False] * len(texts)
    try:
        results = clf([t[:512] for t in texts], batch_size=32)
        flags = []
        for text, r in zip(texts, results):
            is_spam = r["label"].upper() == "SPAM" and r["score"] >= threshold
            if is_spam:
                logger.debug("ML spam (%.2f): %s", r["score"], text[:80])
            flags.append(is_spam)
        return flags
    except Exception as e:
        logger.debug("Spam classifier batch error: %s", e)
        return [False] * len(texts)
