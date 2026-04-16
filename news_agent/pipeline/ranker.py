from __future__ import annotations

import logging

from news_agent.models import NewsItem
from news_agent.pipeline.embeddings import get_model

logger = logging.getLogger(__name__)


def rank_by_query(keyword: str, items: list[NewsItem]) -> list[NewsItem]:
    """Re-rank items using a blend of semantic similarity and existing scores.

    Blend weights:
      - 50% cosine similarity between the query embedding and article embedding
      - 40% LLM relevance score (normalized to [0, 1]; defaults to 0.5 if not yet scored)
      - 10% raw engagement score

    Falls back to original order if sentence-transformers is unavailable.
    This function is CPU-bound and should be called via run_in_executor.
    """
    if not items:
        return items

    model = get_model()
    if model is None:
        return items

    try:
        import numpy as np

        article_texts = [
            f"{item.title} {(item.summary or item.content)[:300]}"
            for item in items
        ]
        all_texts = [keyword] + article_texts
        embeddings = model.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        query_emb = embeddings[0]
        article_embs = embeddings[1:]
        # Cosine similarity: normalized embeddings → dot product
        semantic_scores = np.dot(article_embs, query_emb)

        scored: list[tuple[NewsItem, float]] = []
        for item, sem_score in zip(items, semantic_scores):
            llm_norm = (item.relevance_score / 10.0) if item.relevance_score is not None else 0.5
            blended = 0.5 * float(sem_score) + 0.4 * llm_norm + 0.1 * item.raw_score
            scored.append((item, blended))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "rank_by_query(%r): ranked %d items; top=%r (%.3f)",
            keyword, len(scored),
            scored[0][0].title[:60] if scored else "",
            scored[0][1] if scored else 0.0,
        )
        return [item for item, _ in scored]

    except Exception as e:
        logger.error("Semantic ranking failed for query %r: %s", keyword, e)
        return items
