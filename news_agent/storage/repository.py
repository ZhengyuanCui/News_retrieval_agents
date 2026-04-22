from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta

from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from news_agent.config import settings
from news_agent.models import (
    CollectorStateORM,
    DigestORM,
    DismissedItemORM,
    NewsItem,
    NewsItemORM,
    UserSettingORM,
)

logger = logging.getLogger(__name__)


def _fts_escape(query: str) -> str:
    """Convert a search query to a safe FTS5 match expression (AND of quoted tokens)."""
    words = re.findall(r"\w+", query)
    return " ".join(f'"{w}"' for w in words) if words else '""'


def _rrf_merge(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[str]:
    """Reciprocal Rank Fusion: merge ranked ID lists, rewarding IDs that rank
    highly in multiple lists. k=60 is the standard RRF constant.

    weights: optional per-list multiplier applied to each list's RRF score.
    None → all lists weighted equally (legacy behaviour). Passing e.g.
    [1.0, 0.0] reduces to the first list only. Any scaled pair (c, c) is
    equivalent to None up to a constant factor and produces the same order.
    Must have the same length as ranked_lists if provided.
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length {len(weights)} != ranked_lists length {len(ranked_lists)}"
        )
    scores: dict[str, float] = {}
    for i, ranked in enumerate(ranked_lists):
        w = 1.0 if weights is None else float(weights[i])
        if w == 0.0:
            continue  # skip entirely to avoid adding zero-weighted ids into dict
        for rank, id_ in enumerate(ranked):
            scores[id_] = scores.get(id_, 0.0) + w * (1.0 / (k + rank + 1))
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class NewsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, item: NewsItem) -> None:
        """Insert or replace a NewsItem (idempotent by item.id)."""
        stmt = sqlite_insert(NewsItemORM).values(
            id=item.id,
            source=item.source,
            topic=item.topic,
            title=item.title,
            url=item.url,
            content=item.content,
            author=item.author,
            published_at=item.published_at,
            raw_score=item.raw_score,
            fetched_at=item.fetched_at,
            summary=item.summary,
            tags=item.tags,
            sentiment=item.sentiment,
            relevance_score=item.relevance_score,
            key_entities=item.key_entities,
            is_duplicate=item.is_duplicate,
            duplicate_of=item.duplicate_of,
            language=item.language,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                # Fetch-owned fields — updated every time the item is re-collected
                "source": stmt.excluded.source,
                "topic": stmt.excluded.topic,
                "raw_score": stmt.excluded.raw_score,
                "is_duplicate": stmt.excluded.is_duplicate,
                "duplicate_of": stmt.excluded.duplicate_of,
                # Analysis-owned fields (summary, relevance_score, sentiment, tags,
                # key_entities) are intentionally excluded here — they are only written
                # by LLMAnalyzer, never overwritten by a plain re-fetch.
                # is_starred is also excluded — preserve user stars.
            },
        )
        await self.session.execute(stmt)
        # Keep FTS5 index in sync (DELETE + INSERT because FTS5 has no ON CONFLICT)
        await self.session.execute(
            text("DELETE FROM news_items_fts WHERE id = :id"), {"id": item.id}
        )
        await self.session.execute(
            text("INSERT INTO news_items_fts(id, title, content) VALUES (:id, :title, :content)"),
            {"id": item.id, "title": item.title, "content": item.content or ""},
        )

    async def upsert_many(self, items: list[NewsItem]) -> None:
        for item in items:
            await self.upsert(item)

    async def update_url(self, item_id: str, url: str) -> None:
        """Persist a resolved URL for an existing item (e.g. Google News redirect → real URL)."""
        from sqlalchemy import update
        await self.session.execute(
            update(NewsItemORM).where(NewsItemORM.id == item_id).values(url=url)
        )
        await self.session.commit()

    async def get_by_id(self, item_id: str) -> NewsItem | None:
        result = await self.session.get(NewsItemORM, item_id)
        return result.to_pydantic() if result else None

    async def exists(self, item_id: str) -> bool:
        result = await self.session.get(NewsItemORM, item_id)
        return result is not None

    # ── Dismissal / tombstones ────────────────────────────────────────────────

    async def dismiss(self, item_id: str, reason: str | None = "downvote") -> None:
        """Insert a tombstone so the item is hidden from get_recent/search.
        Idempotent — dismissing an already-dismissed item is a no-op."""
        stmt = sqlite_insert(DismissedItemORM).values(
            item_id=item_id,
            dismissed_at=datetime.utcnow(),
            reason=reason,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["item_id"])
        await self.session.execute(stmt)

    async def undismiss(self, item_id: str) -> None:
        """Remove the tombstone so the item is visible again.
        Idempotent — no-op if not dismissed."""
        await self.session.execute(
            delete(DismissedItemORM).where(DismissedItemORM.item_id == item_id)
        )

    async def is_dismissed(self, item_id: str) -> bool:
        result = await self.session.get(DismissedItemORM, item_id)
        return result is not None

    def _dismissed_subquery(self):
        """Scalar subquery of tombstoned item_ids, for NOT IN filters.
        Returns None when the feature is flagged off so the filter is skipped."""
        if not settings.dismiss_on_downvote:
            return None
        return select(DismissedItemORM.item_id).scalar_subquery()

    async def get_recent(
        self,
        hours: float = 24,
        topic: str | None = None,
        source: str | None = None,
        include_duplicates: bool = False,
        limit: int = 200,
        languages: list[str] | None = None,
    ) -> list[NewsItem]:
        since = datetime.utcnow() - timedelta(hours=hours)
        # Filter by published_at so old articles re-appearing in feeds are excluded.
        # Cap at 7 days regardless of the hours param to avoid surfacing stale news.
        max_age = datetime.utcnow() - timedelta(days=7)
        cutoff = max(since, max_age)
        q = select(NewsItemORM).where(NewsItemORM.published_at >= cutoff)
        if topic:
            q = q.where(NewsItemORM.topic.ilike(topic))
        if source:
            q = q.where(NewsItemORM.source == source)
        if not include_duplicates:
            q = q.where(NewsItemORM.is_duplicate == False)  # noqa: E712
        if languages:
            q = q.where(NewsItemORM.language.in_(languages))
        dismissed = self._dismissed_subquery()
        if dismissed is not None:
            q = q.where(NewsItemORM.id.notin_(dismissed))
        q = q.order_by(
            # Most recently fetched batch first — newly retrieved items appear at top
            # before analysis fills in their relevance scores.
            NewsItemORM.fetched_at.desc(),
            NewsItemORM.relevance_score.desc().nullslast(),
            NewsItemORM.raw_score.desc(),
            NewsItemORM.published_at.desc(),
        )
        q = q.limit(limit)
        result = await self.session.execute(q)
        return [row.to_pydantic() for row in result.scalars()]

    async def bm25_search(self, query: str, limit: int = 60) -> list[str]:
        """BM25 keyword search via SQLite FTS5. Returns article IDs in relevance order."""
        try:
            result = await self.session.execute(
                text(
                    "SELECT id FROM news_items_fts "
                    "WHERE news_items_fts MATCH :q "
                    "ORDER BY bm25(news_items_fts) "
                    "LIMIT :lim"
                ),
                {"q": _fts_escape(query), "lim": limit},
            )
            return [row[0] for row in result]
        except Exception as e:
            logger.warning("BM25 search failed for %r: %s", query, e)
            return []

    async def search(
        self,
        query: str,
        hours: float = 24,
        limit: int = 60,
        languages: list[str] | None = None,
        min_relevance: float = 4.0,
        hybrid_alpha: float | None = None,
    ) -> list[NewsItem]:
        """Content-based search using BM25 + semantic (RRF merge).

        Topic labels are ignored — any article relevant to the query is returned
        regardless of how it was filed. This means 'machine learning' and 'AI'
        surface the same articles, and a stocks article about AI chip demand shows
        up in both AI and NVDA searches.

        Items scored below min_relevance by the LLM are excluded; un-analyzed
        items (NULL relevance_score) are always included.

        hybrid_alpha in [0, 1] controls the BM25 vs semantic blend:
          1.0 → BM25-only (tickers, proper nouns)
          0.5 → balanced (default; equivalent to legacy unweighted RRF)
          0.0 → semantic-only (paraphrase, concepts)
        None → use settings.default_hybrid_alpha. Values outside [0, 1] are
        clamped silently.
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        max_age = datetime.utcnow() - timedelta(days=7)
        cutoff = max(since, max_age)

        from news_agent.pipeline.analyzer import is_question
        from news_agent.pipeline.vector_search import semantic_search

        base_where = [
            NewsItemORM.published_at >= cutoff,
            NewsItemORM.is_duplicate == False,  # noqa: E712
            (NewsItemORM.relevance_score == None) | (NewsItemORM.relevance_score >= min_relevance),  # noqa: E711
        ]
        if languages:
            base_where.append(NewsItemORM.language.in_(languages))
        dismissed = self._dismissed_subquery()
        if dismissed is not None:
            base_where.append(NewsItemORM.id.notin_(dismissed))

        # Hybrid BM25 + semantic with RRF merge. BM25 catches exact keyword/ticker
        # matches; semantic catches paraphrases and related concepts.
        expand = is_question(query) or len(query.split()) >= 3
        bm25_ids, vector_ids = await asyncio.gather(
            self.bm25_search(query, limit=limit * 2),
            semantic_search(query, top_k=limit * 3, expand=expand),
        )
        if hybrid_alpha is None:
            hybrid_alpha = settings.default_hybrid_alpha
        alpha = max(0.0, min(1.0, float(hybrid_alpha)))
        candidate_ids = _rrf_merge(
            [bm25_ids, vector_ids], weights=[alpha, 1.0 - alpha]
        )

        rows = []
        if candidate_ids:
            result = await self.session.execute(
                select(NewsItemORM).where(*base_where, NewsItemORM.id.in_(candidate_ids))
            )
            rows = result.scalars().all()

        # Sort by RRF rank (content relevance), then recency as tiebreaker
        id_rank = {id_: rank for rank, id_ in enumerate(candidate_ids)}
        rows.sort(key=lambda r: (id_rank.get(r.id, 9999), -r.published_at.timestamp()))

        # Deduplicate by title
        seen_titles: set[str] = set()
        deduped = []
        for r in rows:
            title_key = r.title.lower().strip()[:80]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                deduped.append(r)
        return [r.to_pydantic() for r in deduped[:limit]]

    async def mark_duplicate(self, item_id: str, duplicate_of: str) -> None:
        await self.session.execute(
            update(NewsItemORM)
            .where(NewsItemORM.id == item_id)
            .values(is_duplicate=True, duplicate_of=duplicate_of)
        )

    async def clear_all(self) -> dict[str, int]:
        """Delete every row from news_items and digests. Returns row counts deleted."""
        items_result = await self.session.execute(delete(NewsItemORM))
        digests_result = await self.session.execute(delete(DigestORM))
        await self.session.commit()
        return {
            "items": items_result.rowcount,
            "digests": digests_result.rowcount,
        }

    async def prune_old_items(self, retention_days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        result = await self.session.execute(
            delete(NewsItemORM).where(NewsItemORM.fetched_at < cutoff)
        )
        return result.rowcount

    async def upsert_digest(self, date: str, topic: str, content: str, item_count: int) -> None:
        topic = topic.lower()
        stmt = sqlite_insert(DigestORM).values(
            id=f"{date}_{topic}",
            date=date,
            topic=topic,
            content=content,
            item_count=item_count,
            generated_at=datetime.utcnow(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={"content": stmt.excluded.content, "item_count": stmt.excluded.item_count, "generated_at": stmt.excluded.generated_at},
        )
        await self.session.execute(stmt)

    async def get_digest(self, date: str, topic: str) -> DigestORM | None:
        """Return today's digest, or the most recent one if today's hasn't been generated yet."""
        topic = topic.lower()
        exact = await self.session.get(DigestORM, f"{date}_{topic}")
        if exact:
            return exact
        # Fall back to most recent digest for this topic across any date
        result = await self.session.execute(
            select(DigestORM)
            .where(DigestORM.topic == topic)
            .order_by(DigestORM.generated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def delete_digest(self, date: str, topic: str) -> None:
        obj = await self.session.get(DigestORM, f"{date}_{topic}")
        if obj:
            await self.session.delete(obj)

    # ── User settings (server-side key/value store) ───────────────────────────

    async def get_setting(self, key: str, default: str = "") -> str:
        obj = await self.session.get(UserSettingORM, key)
        return obj.value if obj else default

    async def set_setting(self, key: str, value: str) -> None:
        existing = await self.session.get(UserSettingORM, key)
        if existing is None:
            self.session.add(UserSettingORM(key=key, value=value, updated_at=datetime.utcnow()))
        else:
            existing.value = value
            existing.updated_at = datetime.utcnow()

    # ── Collector state ───────────────────────────────────────────────────────

    async def update_collector_state(
        self,
        source: str,
        last_run: datetime | None = None,
        last_error: str | None = None,
        items_fetched: int = 0,
        is_enabled: bool | None = None,
        state: dict | None = None,
    ) -> None:
        existing = await self.session.get(CollectorStateORM, source)
        if existing is None:
            existing = CollectorStateORM(source=source)
            self.session.add(existing)
        if last_run is not None:
            existing.last_run = last_run
        if last_error is not None:
            existing.last_error = last_error
        if items_fetched:
            existing.items_fetched = (existing.items_fetched or 0) + items_fetched
        if is_enabled is not None:
            existing.is_enabled = is_enabled
        if state is not None:
            existing.state = state

    async def get_collector_state(self, source: str) -> CollectorStateORM | None:
        return await self.session.get(CollectorStateORM, source)

    async def get_all_collector_states(self) -> list[CollectorStateORM]:
        result = await self.session.execute(select(CollectorStateORM))
        return list(result.scalars())

    async def update_analysis_many(self, items: list[NewsItem]) -> None:
        """Persist LLM analysis fields for existing items.

        Uses direct UPDATE instead of upsert so analysis results are never
        silently dropped by the on_conflict_do_update that guards fetch-owned fields.
        Only writes items where analyze_batch actually filled in a summary.
        """
        from sqlalchemy import update
        for item in items:
            if item.summary is not None:
                await self.session.execute(
                    update(NewsItemORM)
                    .where(NewsItemORM.id == item.id)
                    .values(
                        summary=item.summary,
                        relevance_score=item.relevance_score,
                        key_entities=item.key_entities,
                        sentiment=item.sentiment,
                        tags=item.tags,
                    )
                )

    async def get_unanalyzed(self, limit: int = 500) -> list[NewsItem]:
        """Return items that have no LLM-generated summary yet, most recently fetched first.

        Ordering by fetched_at ensures the backfill worker prioritizes newly retrieved
        items over old ones that were never analyzed.
        """
        q = (
            select(NewsItemORM)
            .where(
                NewsItemORM.summary == None,  # noqa: E711
                NewsItemORM.is_duplicate == False,  # noqa: E712
            )
            .order_by(NewsItemORM.fetched_at.desc(), NewsItemORM.published_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(q)
        return [row.to_pydantic() for row in result.scalars()]

    async def count_unanalyzed(self) -> int:
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.count()).select_from(NewsItemORM).where(
                NewsItemORM.summary == None,  # noqa: E711
                NewsItemORM.is_duplicate == False,  # noqa: E712
            )
        )
        return result.scalar() or 0

    async def get_stats(self) -> dict:
        from sqlalchemy import func
        total_q = await self.session.execute(select(func.count()).select_from(NewsItemORM))
        topic_q = await self.session.execute(
            select(NewsItemORM.topic, func.count().label("cnt"))
            .group_by(NewsItemORM.topic)
            .order_by(func.count().desc())
        )
        by_topic = {row[0]: row[1] for row in topic_q if row[0]}
        return {
            "total_items": total_q.scalar(),
            "by_topic": by_topic,
        }
