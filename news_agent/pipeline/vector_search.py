"""
In-memory vector index for semantic search over stored articles.

The index is rebuilt lazily on first use and refreshed when stale (TTL=10 min).
A disk cache (data/vector_index.npz) is written after every rebuild so that a
server restart within the TTL window skips the expensive encode step entirely.

Uses the same all-MiniLM-L6-v2 model as the rest of the pipeline so the model
is already warm when search is called.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from news_agent.pipeline.embeddings import get_model

logger = logging.getLogger(__name__)

_TTL_SECONDS = 600  # rebuild index if older than 10 minutes
_CACHE_DIR = Path("data")
_CACHE_NPZ = _CACHE_DIR / "vector_index.npz"   # embeddings array
_CACHE_META = _CACHE_DIR / "vector_index.json"  # ids + timestamp


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
        """Return True if a valid on-disk cache exists and is within TTL."""
        try:
            if not _CACHE_NPZ.exists() or not _CACHE_META.exists():
                return False
            meta = json.loads(_CACHE_META.read_text())
            age = time.time() - meta["saved_at"]
            return age < _TTL_SECONDS
        except Exception:
            return False

    def _load_cache(self) -> bool:
        """Load embeddings and IDs from disk. Returns True on success."""
        try:
            meta = json.loads(_CACHE_META.read_text())
            data = np.load(str(_CACHE_NPZ))
            self._ids = meta["ids"]
            self._embeddings = data["embeddings"]
            # Set _built_at so it expires at the right wall-clock time
            age = time.time() - meta["saved_at"]
            self._built_at = time.monotonic() - age
            logger.info(
                "Vector index loaded from disk cache: %d articles (age %.0fs)",
                len(self._ids), age,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load vector index cache: %s", e)
            return False

    def _save_cache(self) -> None:
        """Persist embeddings and IDs to disk."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(_CACHE_NPZ), embeddings=self._embeddings)
            _CACHE_META.write_text(json.dumps({"ids": self._ids, "saved_at": time.time()}))
            logger.debug("Vector index cache saved (%d articles)", len(self._ids))
        except Exception as e:
            logger.warning("Failed to save vector index cache: %s", e)

    def _delete_cache(self) -> None:
        """Remove the on-disk cache so the next ensure_fresh triggers a full rebuild."""
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
            if not self._is_stale():   # double-check after acquiring lock
                return
            # Try disk cache first — avoids DB query + encode on server restart
            if self._cache_is_fresh() and self._load_cache():
                return
            await self._rebuild()

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
            lambda: model.encode(
                texts,
                batch_size=256,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )

        self._ids = ids
        self._embeddings = embeddings
        self._built_at = time.monotonic()
        logger.info("Vector index rebuilt: %d articles, %.1f MB", len(ids), embeddings.nbytes / 1024 ** 2)

        # Persist to disk so the next server restart within TTL skips this step
        loop.run_in_executor(None, self._save_cache)

    def _search_sync(self, query: str, top_k: int) -> list[str]:
        if self._embeddings is None or not self._ids:
            return []
        model = get_model()
        q_emb: np.ndarray = model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores: np.ndarray = self._embeddings @ q_emb          # (N,)
        top_idx = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort descending
        return [self._ids[i] for i in top_idx]

    def invalidate(self) -> None:
        """Mark the index as stale and delete the disk cache.

        Call after bulk ingest so the next search triggers a full rebuild
        rather than loading a now-stale cache.
        """
        self._built_at = 0.0
        self._delete_cache()


_index = _VectorIndex()


async def semantic_search(query: str, top_k: int = 60) -> list[str]:
    """Return up to top_k article IDs ranked by semantic similarity to query.

    On first call after a server restart: loads from disk cache if fresh,
    otherwise rebuilds from DB (~5-10s with warm model).
    Returns IDs in descending similarity order.
    """
    await _index.ensure_fresh()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _index._search_sync, query, top_k)


def invalidate_index() -> None:
    """Mark the index stale and remove the disk cache.
    Call after bulk ingest so the next search rebuilds from the latest DB state.
    """
    _index.invalidate()
