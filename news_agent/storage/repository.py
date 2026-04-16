from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from news_agent.models import CollectorStateORM, DigestORM, NewsItem, NewsItemORM


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
            is_starred=item.is_starred,
            language=item.language,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                # Fetch-owned fields — updated every time the item is re-collected
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
        q = q.order_by(
            NewsItemORM.relevance_score.desc().nullslast(),
            NewsItemORM.raw_score.desc(),
            NewsItemORM.published_at.desc(),
        )
        q = q.limit(limit)
        result = await self.session.execute(q)
        return [row.to_pydantic() for row in result.scalars()]

    async def search(
        self,
        query: str,
        hours: float = 24,
        limit: int = 60,
        strict: bool = False,
        languages: list[str] | None = None,
        min_relevance: float = 4.0,
    ) -> list[NewsItem]:
        """Search for items matching query.

        Priority: items fetched specifically for this topic (topic==query) are returned first.
        If none exist yet, falls back to content search (OR match) as a placeholder while
        a keyword fetch is in progress.

        Items that have been analyzed and scored below min_relevance are excluded.
        Un-analyzed items (NULL relevance_score) are always included.
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        max_age = datetime.utcnow() - timedelta(days=7)
        cutoff = max(since, max_age)
        base_where = [
            NewsItemORM.published_at >= cutoff,
            NewsItemORM.is_duplicate == False,  # noqa: E712
            # Keep un-analyzed items (NULL) but drop confirmed low-relevance ones
            (NewsItemORM.relevance_score == None) | (NewsItemORM.relevance_score >= min_relevance),  # noqa: E711
        ]
        if languages:
            base_where.append(NewsItemORM.language.in_(languages))

        # 1. Try topic-exact match first (items fetched specifically for this keyword)
        topic_q = (
            select(NewsItemORM)
            .where(*base_where, NewsItemORM.topic.ilike(query))
            .order_by(
                NewsItemORM.relevance_score.desc().nullslast(),
                NewsItemORM.raw_score.desc(),
                NewsItemORM.published_at.desc(),
            )
            .limit(limit)
        )
        result = await self.session.execute(topic_q)
        rows = result.scalars().all()
        if rows:
            return [r.to_pydantic() for r in rows]

        # No topic-specific items yet — return empty so the spinner stays visible
        # while the background keyword fetch runs. A content-search fallback
        # surfaces unrelated items (AI, stocks) that happen to mention the keyword,
        # producing misleading results with wrong topic tags.
        return []

    async def mark_duplicate(self, item_id: str, duplicate_of: str) -> None:
        await self.session.execute(
            update(NewsItemORM)
            .where(NewsItemORM.id == item_id)
            .values(is_duplicate=True, duplicate_of=duplicate_of)
        )

    async def set_starred(self, item_id: str, starred: bool) -> None:
        await self.session.execute(
            update(NewsItemORM).where(NewsItemORM.id == item_id).values(is_starred=starred)
        )

    async def get_starred_ids(self) -> set[str]:
        result = await self.session.execute(
            select(NewsItemORM.id).where(NewsItemORM.is_starred == True)  # noqa: E712
        )
        return set(result.scalars())

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

    # ── Collector state ───────────────────────────────────────────────────────

    async def update_collector_state(
        self,
        source: str,
        last_run: datetime | None = None,
        last_error: str | None = None,
        items_fetched: int = 0,
        is_enabled: bool | None = None,
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

    async def get_all_collector_states(self) -> list[CollectorStateORM]:
        result = await self.session.execute(select(CollectorStateORM))
        return list(result.scalars())

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
