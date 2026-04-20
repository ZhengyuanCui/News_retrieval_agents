from __future__ import annotations

import logging
import math
from datetime import datetime

from news_agent.models import NewsItem
from news_agent.pipeline.embeddings import get_model

logger = logging.getLogger(__name__)

# Editorial authority weights per source label.  Higher = boost, lower = penalty.
# The table is organized into tiers that match the tone of the result ordering
# we want: frontier-lab research blogs and primary research at the top;
# researcher personal blogs and academic labs next; reputable trade press in
# the middle; syndicated wire/aggregator feeds weighted *down* because they
# produce copies of stories that are already covered by the primary sources.
#
# Unknown sources get `_DEFAULT_AUTHORITY`.  Source strings are matched
# case-insensitively against `NewsItem.source`.
_DEFAULT_AUTHORITY = 0.55

_SOURCE_AUTHORITY: dict[str, float] = {
    # ── Tier S: Frontier AI labs — primary research / announcement blogs ──
    # These publish the actual research/product announcements that every other
    # outlet then summarizes.  We want these near the top of any AI query.
    "openai":             1.00,
    "anthropic":          1.00,
    "deepmind":           1.00,
    "google-ai-blog":     0.98,
    "meta-research":      0.98,
    "apple-ml":           0.96,
    "microsoft-research": 0.96,
    "amazon-science":     0.93,
    "nvidia-ai":          0.93,
    "huggingface":        0.92,
    "mistral":            0.92,
    "cohere":             0.90,
    "allen-ai":           0.92,
    "stability-ai":       0.88,
    "sakana-ai":          0.88,
    "elevenlabs":         0.85,
    "waymo":              0.88,

    # Anthropic / robotics / media labs without native RSS — delivered via a
    # Google News search wrapper, but the content is still frontier-lab news.
    "anthropic-news":     0.92,
    "physical-intelligence": 0.90,
    "figure-ai":          0.88,
    "world-labs":         0.88,
    "boston-dynamics":    0.88,
    "runway-ml":          0.85,

    # ── Tier A: Primary research — arXiv, academic labs ─────────────────────
    "arxiv-cs-ai":        0.95,
    "arxiv-cs-lg":        0.95,
    "arxiv-cs-cv":        0.93,
    "arxiv-cs-ro":        0.93,
    "arxiv-cs-cl":        0.93,
    "arxiv-stat-ml":      0.93,
    "bair":               0.92,  # Berkeley AI Research
    "stanford-ai":        0.92,
    "mit-csail":          0.92,
    "mit":                0.90,
    "ieee-spectrum":      0.78,
    "ieee-robotics":      0.80,
    "mit-tech-review":    0.82,

    # ── Tier A: Top researcher personal blogs / substacks ───────────────────
    "lilian-weng":        0.95,
    "karpathy-blog":      0.95,
    "karpathy-new":       0.95,
    "colah":              0.95,
    "chollet":            0.92,
    "jay-alammar":        0.90,
    "sebastian-ruder":    0.90,
    "chip-huyen":         0.90,
    "simon-willison":     0.88,
    "gwern":              0.90,
    "neel-nanda":         0.92,
    "yoshua-bengio":      0.95,
    "paul-christiano":    0.92,
    "steinhardt":         0.90,
    "sam-altman":         0.85,
    "dario-amodei":       0.95,
    "fastai":             0.85,

    # ── Tier B: Curated AI newsletters & paper digests ──────────────────────
    "the-batch":          0.85,  # Andrew Ng
    "import-ai":          0.85,  # Jack Clark
    "last-week-in-ai":    0.82,
    "the-gradient":       0.82,
    "papers-with-code":   0.85,
    "alignment-forum":    0.85,
    "lesswrong":          0.72,
    "miri":               0.80,

    # ── Tier B: High-signal AI podcasts (researcher interviews) ─────────────
    "latent-space":       0.82,
    "twiml-ai":           0.80,
    "lex-fridman-pod":    0.78,
    "cognitive-rev":      0.78,
    "practical-ai":       0.75,
    "no-priors":          0.72,
    "nvidia-ai-pod":      0.75,
    "eye-on-ai":          0.72,
    "brain-inspired":     0.72,
    "future-of-life":     0.75,
    "80k-hours":          0.72,
    "undivided-attention": 0.70,

    # ── Tier C: Reputable business / financial wires ────────────────────────
    # These are trusted but produce many incremental-update stories; keep
    # them in the middle of the pack so frontier-lab primary sources win.
    "bloomberg":          0.75,
    "reuters":            0.75,
    "wsj":                0.72,
    "ft":                 0.72,
    "cnbc":               0.65,
    "marketwatch":        0.60,

    # ── Tier C: Tech-trade press ────────────────────────────────────────────
    "techcrunch":         0.60,
    "venturebeat":        0.55,
    "wired":              0.55,
    "thenextweb":         0.50,
    "the-register":       0.50,
    "techmeme":           0.60,   # meta-aggregator but curated

    # ── Tier C: General news aggregators / wires ────────────────────────────
    # Down-weighted on purpose — "steady news articles" that mostly repackage
    # what primary sources already published.
    "bbc":                0.50,
    "bbc-news":           0.50,

    # ── Tier D: Syndicated search aggregators — down-weighted ──────────────
    # Google News and Bing News are firehoses of duplicated coverage; we
    # prefer to pick up the original publisher directly when possible.
    "google-news":        0.25,
    "bing-news":          0.25,
    "news":               0.30,   # generic "news" topic from unrestricted top-stories feed

    # ── Platform sources ────────────────────────────────────────────────────
    "github":             0.85,   # repo releases / PRs — primary dev activity
    "youtube":            0.55,
    "reddit":             0.40,
    "x":                  0.35,
    "twitter":            0.35,
    "rss":                0.55,   # generic fallback label
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
      35% cosine similarity (bi-encoder)
      25% LLM relevance score (normalized to [0,1]; 0.5 if unscored)
      20% source authority   — frontier-lab blogs / arXiv up, news aggregators down
      10% freshness decay (exponential, half-life 48 h)
      10% raw engagement score

    The authority weight is deliberately high so that frontier-lab primary
    sources (OpenAI, DeepMind, Anthropic, arXiv, ...) outrank the many
    aggregator copies of the same story (Google News, Bing News, wires).

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
            authority = _SOURCE_AUTHORITY.get(item.source.lower(), _DEFAULT_AUTHORITY)
            blended = (
                0.35 * float(sem_score)
                + 0.25 * llm_norm
                + 0.20 * authority
                + 0.10 * freshness
                + 0.10 * item.raw_score
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
