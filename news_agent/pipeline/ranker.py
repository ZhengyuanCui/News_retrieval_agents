from __future__ import annotations

import logging
import math
from datetime import datetime

from news_agent.models import NewsItem
from news_agent.pipeline.embeddings import get_model

logger = logging.getLogger(__name__)

# Editorial authority weights per source label (default 0.65 for unknowns)
_SOURCE_AUTHORITY: dict[str, float] = {
    "openai": 1.0, "deepmind": 1.0, "mit": 0.9, "github": 0.9,
    "bloomberg": 0.9, "reuters": 0.9, "wsj": 0.85, "ft": 0.85,
    "bbc": 0.82, "techcrunch": 0.80, "wired": 0.75,
    "cnbc": 0.75, "marketwatch": 0.70, "ventureBeat": 0.75,
    "youtube": 0.60, "linkedin": 0.60, "rss": 0.65,
    "reddit": 0.50, "twitter": 0.40,
}

_cross_encoder = None  # lazy-loaded; False = tried and unavailable


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder loaded for top-20 re-ranking")
        except Exception as e:
            logger.debug("Cross-encoder unavailable (%s) — skipping", e)
            _cross_encoder = False
    return _cross_encoder if _cross_encoder is not False else None


def rank_by_query(keyword: str, items: list[NewsItem]) -> list[NewsItem]:
    """Re-rank items using a blend of semantic similarity, LLM score, freshness, and source authority.

    Blend weights:
      40% cosine similarity (bi-encoder)
      30% LLM relevance score (normalized to [0,1]; 0.5 if unscored)
      10% raw engagement score
      10% freshness decay (exponential, half-life 48 h)
      10% source authority

    Top-20 results are further re-ranked with a cross-encoder when available.
    Falls back to original order if sentence-transformers is unavailable.
    CPU-bound — call via run_in_executor.
    """
    if not items:
        return items

    model = get_model()
    if model is None:
        return items

    now = datetime.utcnow()

    try:
        import numpy as np

        article_texts = [
            f"{item.title} {(item.summary or item.content)[:300]}"
            for item in items
        ]
        embeddings = model.encode(
            [keyword] + article_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_emb = embeddings[0]
        article_embs = embeddings[1:]
        semantic_scores = np.dot(article_embs, query_emb)

        scored: list[tuple[NewsItem, float]] = []
        for item, sem_score in zip(items, semantic_scores):
            llm_norm = (item.relevance_score / 10.0) if item.relevance_score is not None else 0.5
            age_hours = max(0.0, (now - item.published_at).total_seconds() / 3600)
            freshness = math.exp(-age_hours / 48)  # half-life 48 h
            authority = _SOURCE_AUTHORITY.get(item.source.lower(), 0.65)
            blended = (
                0.40 * float(sem_score)
                + 0.30 * llm_norm
                + 0.10 * item.raw_score
                + 0.10 * freshness
                + 0.10 * authority
            )
            scored.append((item, blended))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [item for item, _ in scored]

        # Cross-encoder re-ranking on top-20
        ce = _get_cross_encoder()
        if ce is not None and len(ranked) > 1:
            top_k = min(20, len(ranked))
            top, rest = ranked[:top_k], ranked[top_k:]
            pairs = [
                (keyword, f"{it.title} {(it.summary or it.content)[:200]}")
                for it in top
            ]
            ce_scores = ce.predict(pairs, show_progress_bar=False)
            top = [it for it, _ in sorted(zip(top, ce_scores), key=lambda x: x[1], reverse=True)]
            ranked = top + rest

        logger.debug(
            "rank_by_query(%r): %d items; top=%r",
            keyword, len(ranked),
            ranked[0].title[:60] if ranked else "",
        )
        return ranked

    except Exception as e:
        logger.error("Semantic ranking failed for query %r: %s", keyword, e)
        return items
