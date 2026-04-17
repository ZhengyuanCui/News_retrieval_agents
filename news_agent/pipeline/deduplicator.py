from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

def _title_key(title: str) -> str:
    """Normalize a title to a compact dedup key.

    Strips trailing ' - Publication Name' suffixes that news aggregators
    (Google News, Bing News) append, so 'Article Title - Reuters' and
    'Article Title' hash to the same key.
    """
    clean = title.strip()
    # Strip trailing " - Source" (up to ~40 chars after the dash) — publication names
    # are short; a long suffix is part of the actual title and should be kept.
    clean = re.sub(r"\s+-\s+\w[^-]{0,40}$", "", clean)
    return re.sub(r"[^a-z0-9]", "", clean.lower())[:80]

from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

# URL query params that are pure tracking noise
TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "referrer",
    "_hsenc", "_hsmi", "hsCtaTracking",
})


def normalize_url(url: str) -> str:
    """Strip tracking params and normalize URL for dedup comparison."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=False)
        clean_params = {k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS}
        clean_query = urlencode(clean_params, doseq=True)
        return urlunparse(parsed._replace(query=clean_query, fragment="")).rstrip("/").lower()
    except Exception:
        return url.lower()


class Deduplicator:
    def __init__(self, strategy: str | None = None, threshold: float | None = None) -> None:
        self.strategy = strategy or settings.dedup_strategy
        self.threshold = threshold or settings.dedup_similarity_threshold
        self._model = None  # lazy-loaded

    def _load_semantic_model(self):
        if self._model is None:
            from news_agent.pipeline.embeddings import get_model
            self._model = get_model()
            if self._model is None:
                logger.warning("sentence-transformers not installed; falling back to tfidf dedup")
                self.strategy = "tfidf"
        return self._model

    def deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Remove duplicates in two passes:
        1. URL-based exact dedup (fast)
        2. Semantic/TF-IDF similarity dedup (quality)
        Returns the deduplicated list; duplicate items have is_duplicate=True.
        """
        items = self._url_dedup(items)
        if self.strategy == "semantic":
            items = self._semantic_dedup(items)
        elif self.strategy == "tfidf":
            items = self._tfidf_dedup(items)
        return items

    @staticmethod
    def _detail_score(item: NewsItem) -> float:
        """Score how detailed an item is — prefer rich content over high engagement."""
        content_len = len(item.content or "")
        has_summary = 1.0 if item.summary else 0.0
        # Weight content length heavily; engagement as tiebreaker
        return content_len * 0.001 + has_summary * 2.0 + item.raw_score * 0.5

    def _url_dedup(self, items: list[NewsItem]) -> list[NewsItem]:
        seen_urls: dict[str, str] = {}    # normalized_url -> item_id
        seen_titles: dict[str, str] = {}  # normalized_title -> item_id
        result = []
        for item in items:
            norm_url = normalize_url(item.url)
            norm_title = _title_key(item.title)

            # Check URL match first, then title match
            existing_id = seen_urls.get(norm_url) or (seen_titles.get(norm_title) if norm_title else None)

            if existing_id:
                existing_idx = next((i for i, x in enumerate(result) if x.id == existing_id), None)
                if existing_idx is not None:
                    existing = result[existing_idx]
                    # Keep the more detailed version
                    if self._detail_score(item) > self._detail_score(existing):
                        existing.is_duplicate = True
                        existing.duplicate_of = item.id
                        seen_urls[norm_url] = item.id
                        if norm_title:
                            seen_titles[norm_title] = item.id
                        result[existing_idx] = item
                    else:
                        item.is_duplicate = True
                        item.duplicate_of = existing_id
                        result.append(item)
                else:
                    item.is_duplicate = True
                    item.duplicate_of = existing_id
                    result.append(item)
            else:
                seen_urls[norm_url] = item.id
                if norm_title:
                    seen_titles[norm_title] = item.id
                result.append(item)
        dupes = sum(1 for i in result if i.is_duplicate)
        if dupes:
            logger.debug("URL/title dedup removed %d duplicates", dupes)
        return result

    def _semantic_dedup(self, items: list[NewsItem]) -> list[NewsItem]:
        import numpy as np

        model = self._load_semantic_model()
        if model is None:
            return items

        non_dupes = [i for i in items if not i.is_duplicate]
        if len(non_dupes) < 2:
            return items

        texts = [f"{item.title} {item.content[:200]}" for item in non_dupes]
        try:
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            # Cosine similarity matrix (embeddings are normalized, so dot product = cosine sim)
            sim_matrix = np.dot(embeddings, embeddings.T)

            marked_duplicate = set()
            for i in range(len(non_dupes)):
                if i in marked_duplicate:
                    continue
                for j in range(i + 1, len(non_dupes)):
                    if j in marked_duplicate:
                        continue
                    if sim_matrix[i, j] >= self.threshold:
                        # Keep the more detailed version
                        if self._detail_score(non_dupes[i]) >= self._detail_score(non_dupes[j]):
                            non_dupes[j].is_duplicate = True
                            non_dupes[j].duplicate_of = non_dupes[i].id
                            marked_duplicate.add(j)
                        else:
                            non_dupes[i].is_duplicate = True
                            non_dupes[i].duplicate_of = non_dupes[j].id
                            marked_duplicate.add(i)
                            break

            if marked_duplicate:
                logger.debug("Semantic dedup removed %d near-duplicates", len(marked_duplicate))
        except Exception as e:
            logger.error("Semantic dedup failed: %s", e)

        return items

    def _tfidf_dedup(self, items: list[NewsItem]) -> list[NewsItem]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        non_dupes = [i for i in items if not i.is_duplicate]
        if len(non_dupes) < 2:
            return items

        texts = [f"{item.title} {item.content[:200]}" for item in non_dupes]
        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(matrix)

            marked_duplicate = set()
            for i in range(len(non_dupes)):
                if i in marked_duplicate:
                    continue
                for j in range(i + 1, len(non_dupes)):
                    if j in marked_duplicate:
                        continue
                    if sim_matrix[i, j] >= self.threshold:
                        if self._detail_score(non_dupes[i]) >= self._detail_score(non_dupes[j]):
                            non_dupes[j].is_duplicate = True
                            non_dupes[j].duplicate_of = non_dupes[i].id
                            marked_duplicate.add(j)
                        else:
                            non_dupes[i].is_duplicate = True
                            non_dupes[i].duplicate_of = non_dupes[j].id
                            marked_duplicate.add(i)
                            break

            if marked_duplicate:
                logger.debug("TF-IDF dedup removed %d near-duplicates", len(marked_duplicate))
        except Exception as e:
            logger.error("TF-IDF dedup failed: %s", e)

        return items
