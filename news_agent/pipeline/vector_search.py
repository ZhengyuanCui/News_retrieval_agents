"""
In-memory vector index for semantic search over stored articles.

The index is rebuilt lazily on first use and refreshed when stale (TTL=10 min).
When few new articles exist since the last build, an incremental append is used
instead of a full rebuild — saving the cost of re-encoding thousands of articles.
A disk cache (data/vector_index.npz) is written after every rebuild so that a
server restart within the TTL window skips the expensive encode step entirely.

Uses the same all-MiniLM-L6-v2 model as the rest of the pipeline so the model
is already warm when search is called.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from news_agent.pipeline.embeddings import get_model

logger = logging.getLogger(__name__)

_TTL_SECONDS = 600  # rebuild index if older than 10 minutes
_CACHE_DIR = Path("data")
_CACHE_NPZ = _CACHE_DIR / "vector_index.npz"
_CACHE_META = _CACHE_DIR / "vector_index.json"

_EXPANSION_PROMPT = """\
Generate 4 alternative phrasings of the following search query that cover different \
ways someone might express the same intent. Return only the alternatives, one per line, \
no numbering, no bullets, no extra text.
Query: {query}"""


class _VectorIndex:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (N, 384), L2-normalised
        self._built_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_stale(self) -> bool:
        return time.monotonic() - self._built_at > _TTL_SECONDS

    # ── disk cache helpers ────────────────────────────────────────────────────

    def _cache_is_fresh(self) -> bool:
        try:
            if not _CACHE_NPZ.exists() or not _CACHE_META.exists():
                return False
            meta = json.loads(_CACHE_META.read_text())
            return time.time() - meta["saved_at"] < _TTL_SECONDS
        except Exception:
            return False

    def _load_cache(self) -> bool:
        try:
            meta = json.loads(_CACHE_META.read_text())
            data = np.load(str(_CACHE_NPZ))
            self._ids = meta["ids"]
            self._embeddings = data["embeddings"]
            age = time.time() - meta["saved_at"]
            self._built_at = time.monotonic() - age
            logger.info("Vector index loaded from disk: %d articles (age %.0fs)", len(self._ids), age)
            return True
        except Exception as e:
            logger.warning("Failed to load vector index cache: %s", e)
            return False

    def _save_cache(self) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(_CACHE_NPZ), embeddings=self._embeddings)
            _CACHE_META.write_text(json.dumps({"ids": self._ids, "saved_at": time.time()}))
            logger.debug("Vector index cache saved (%d articles)", len(self._ids))
        except Exception as e:
            logger.warning("Failed to save vector index cache: %s", e)

    def _delete_cache(self) -> None:
        for path in (_CACHE_NPZ, _CACHE_META):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    # ── core logic ────────────────────────────────────────────────────────────

    async def ensure_fresh(self) -> None:
        """Load from disk cache or rebuild from DB if stale. Thread-safe."""
        if not self._is_stale():
            return
        async with self._lock:
            if not self._is_stale():
                return
            if self._cache_is_fresh() and self._load_cache():
                return
            # Incremental append when we have an existing in-memory index
            if self._embeddings is not None and await self._incremental_update():
                return
            await self._rebuild()

    async def _incremental_update(self) -> bool:
        """Encode only articles not yet in the index and append them.

        Returns True on success, False to signal the caller to do a full rebuild.
        Large batches (>200 new articles) always trigger a full rebuild — it's
        faster than re-normalising and appending a giant matrix.
        """
        from news_agent.storage.database import get_session
        from news_agent.models import NewsItemORM
        from sqlalchemy import select

        existing = set(self._ids)
        cutoff_7d = datetime.utcnow() - timedelta(days=7)

        async with get_session() as session:
            result = await session.execute(
                select(NewsItemORM.id, NewsItemORM.title, NewsItemORM.content)
                .where(
                    NewsItemORM.published_at >= cutoff_7d,
                    NewsItemORM.is_duplicate == False,  # noqa: E712
                    NewsItemORM.id.notin_(existing),
                )
            )
            new_rows = result.all()

        if not new_rows:
            self._built_at = time.monotonic()
            return True

        if len(new_rows) > 200:
            return False  # trigger full rebuild

        model = get_model()
        if model is None:
            return False

        ids = [r[0] for r in new_rows]
        texts = [f"{r[1]} {(r[2] or '')[:200]}" for r in new_rows]
        loop = asyncio.get_event_loop()
        new_embs: np.ndarray = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False),
        )
        self._ids = self._ids + ids
        self._embeddings = np.vstack([self._embeddings, new_embs])
        self._built_at = time.monotonic()
        logger.info("Vector index incremental: +%d articles (total %d)", len(ids), len(self._ids))
        return True

    async def _rebuild(self) -> None:
        from news_agent.storage.database import get_session
        from news_agent.models import NewsItemORM
        from sqlalchemy import select

        cutoff = datetime.utcnow() - timedelta(days=7)
        async with get_session() as session:
            result = await session.execute(
                select(NewsItemORM.id, NewsItemORM.title, NewsItemORM.content)
                .where(
                    NewsItemORM.published_at >= cutoff,
                    NewsItemORM.is_duplicate == False,  # noqa: E712
                )
            )
            rows = result.all()

        if not rows:
            self._ids = []
            self._embeddings = None
            self._built_at = time.monotonic()
            return

        ids = [r[0] for r in rows]
        texts = [f"{r[1]} {(r[2] or '')[:200]}" for r in rows]
        model = get_model()
        loop = asyncio.get_event_loop()
        embeddings: np.ndarray = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False),
        )
        self._ids = ids
        self._embeddings = embeddings
        self._built_at = time.monotonic()
        logger.info("Vector index rebuilt: %d articles, %.1f MB", len(ids), embeddings.nbytes / 1024 ** 2)
        loop.run_in_executor(None, self._save_cache)

    def _search_sync(self, query: str, top_k: int, q_emb: np.ndarray | None = None) -> list[str]:
        if self._embeddings is None or not self._ids:
            return []
        model = get_model()
        if q_emb is None:
            q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores: np.ndarray = self._embeddings @ q_emb
        top_idx = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self._ids[i] for i in top_idx]

    def invalidate(self) -> None:
        """Mark the index stale and delete the disk cache.
        Call after bulk ingest so the next search triggers a rebuild.
        """
        self._built_at = 0.0
        self._delete_cache()


_index = _VectorIndex()


async def _expand_query_embedding(query: str) -> np.ndarray | None:
    """Ask the analysis LLM for alternative phrasings, return their averaged embedding.

    Improves recall for natural-language and entity-heavy queries by covering
    synonym and paraphrase space (e.g. 'GPT vs Claude' → also encodes
    'Comparing OpenAI and Anthropic models').
    Times out after 4 s so a slow LLM never stalls search.
    """
    model = get_model()
    if model is None:
        return None
    try:
        import litellm
        from news_agent.config import settings

        m = settings.analysis_model.lower()
        if m.startswith("anthropic/"):
            api_key = settings.anthropic_api_key or settings.llm_api_key or None
        elif m.startswith("gemini/") or m.startswith("google/"):
            api_key = settings.gemini_api_key or None
        elif m.startswith("openai/"):
            api_key = settings.openai_api_key or None
        else:
            api_key = settings.llm_api_key or None

        response = await asyncio.wait_for(
            litellm.acompletion(
                model=settings.analysis_model,
                max_tokens=120,
                messages=[{"role": "user", "content": _EXPANSION_PROMPT.format(query=query)}],
                **({"api_key": api_key} if api_key else {}),
            ),
            timeout=4.0,
        )
        raw = response.choices[0].message.content.strip()
        phrasings = [p.strip() for p in raw.splitlines() if len(p.strip()) > 3][:4]
    except Exception as e:
        logger.debug("Query expansion skipped (%s)", e)
        phrasings = []

    all_queries = [query] + phrasings
    loop = asyncio.get_event_loop()
    embs: np.ndarray = await loop.run_in_executor(
        None,
        lambda: model.encode(all_queries, normalize_embeddings=True, show_progress_bar=False),
    )
    avg = embs.mean(axis=0)
    norm = float(np.linalg.norm(avg))
    return (avg / norm) if norm > 0 else avg


async def semantic_search(query: str, top_k: int = 60, expand: bool = False) -> list[str]:
    """Return up to top_k article IDs ranked by semantic similarity to query.

    When expand=True, uses LLM query expansion — expansion and freshness check
    run concurrently so the added latency is ≤ the LLM round-trip (capped at 4 s).
    Returns IDs in descending similarity order.
    """
    if expand and get_model() is not None:
        results = await asyncio.gather(
            _index.ensure_fresh(),
            _expand_query_embedding(query),
            return_exceptions=True,
        )
        q_emb = results[1] if isinstance(results[1], np.ndarray) else None
    else:
        await _index.ensure_fresh()
        q_emb = None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _index._search_sync(query, top_k, q_emb))


def invalidate_index() -> None:
    """Mark the index stale and remove the disk cache.
    Call after bulk ingest so the next search rebuilds from the latest DB state.
    """
    _index.invalidate()
