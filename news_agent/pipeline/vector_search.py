"""
In-memory vector index for semantic search over stored articles.

The index is rebuilt lazily on first use and refreshed when stale (TTL=10 min).
Uses the same all-MiniLM-L6-v2 model as the rest of the pipeline so the model
is already warm when search is called.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta

import numpy as np

from news_agent.pipeline.embeddings import get_model

logger = logging.getLogger(__name__)

_TTL_SECONDS = 600  # rebuild index if older than 10 minutes


class _VectorIndex:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (N, 384), L2-normalised
        self._built_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_stale(self) -> bool:
        return time.monotonic() - self._built_at > _TTL_SECONDS

    async def ensure_fresh(self) -> None:
        """Rebuild the index if it is stale or empty. Thread-safe via asyncio.Lock."""
        if not self._is_stale():
            return
        async with self._lock:
            if not self._is_stale():   # double-check after acquiring lock
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
        logger.info("Vector index built: %d articles, %.1f MB", len(ids), embeddings.nbytes / 1024 ** 2)

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
        """Mark the index as stale so the next search triggers a rebuild."""
        self._built_at = 0.0


_index = _VectorIndex()


async def semantic_search(query: str, top_k: int = 60) -> list[str]:
    """Return up to top_k article IDs ranked by semantic similarity to query.

    The index is rebuilt lazily on the first call and refreshed every 10 minutes.
    Returns IDs in descending similarity order.
    """
    await _index.ensure_fresh()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _index._search_sync, query, top_k)


def invalidate_index() -> None:
    """Mark the index stale. Call after bulk ingest so the next search rebuilds."""
    _index.invalidate()
