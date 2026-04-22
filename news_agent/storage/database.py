from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import sqlalchemy
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from news_agent.config import settings
from news_agent.models import Base

# Ensure data/ directory exists
os.makedirs("data", exist_ok=True)

_is_sqlite = "sqlite" in settings.database_url

engine = create_async_engine(
    settings.database_url,
    echo=False,
    # timeout= sets the busy-wait on the sqlite3 connection (seconds)
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:
    # WAL mode lets readers and the writer coexist without blocking each other.
    # busy_timeout makes writers retry for up to 10 s instead of failing immediately.
    # This event fires for every raw sqlite3 connection created by the pool.
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def _migrate_drop_is_starred(conn) -> None:
    """Drop the is_starred column that was removed from the ORM (SQLite 3.35+)."""
    result = await conn.execute(text("PRAGMA table_info(news_items)"))
    cols = {row[1] for row in result.fetchall()}
    if "is_starred" in cols:
        await conn.execute(text("ALTER TABLE news_items DROP COLUMN is_starred"))


async def _migrate_collector_state(conn) -> None:
    """Add the JSON state column and backfill YouTube legacy fields."""
    result = await conn.execute(text(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='collector_state'"
    ))
    if result.scalar_one_or_none() is None:
        return

    result = await conn.execute(text("PRAGMA table_info(collector_state)"))
    cols = {row[1] for row in result.fetchall()}
    if "state" not in cols:
        await conn.execute(text("ALTER TABLE collector_state ADD COLUMN state JSON DEFAULT '{}'"))

    rows = await conn.execute(text(
        """
        SELECT source, youtube_quota_used, youtube_quota_reset_date
        FROM collector_state
        WHERE (state IS NULL OR state = '' OR state = '{}')
          AND (COALESCE(youtube_quota_used, 0) != 0 OR youtube_quota_reset_date IS NOT NULL)
        """
    ))
    for row in rows.fetchall():
        payload = {
            "youtube_quota_used": row.youtube_quota_used or 0,
            "youtube_quota_reset_date": row.youtube_quota_reset_date,
        }
        await conn.execute(
            text("UPDATE collector_state SET state = :state WHERE source = :source"),
            {"source": row.source, "state": json.dumps(payload)},
        )


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        if _is_sqlite:
            await _migrate_drop_is_starred(conn)
        await conn.run_sync(Base.metadata.create_all)
        if _is_sqlite:
            await _migrate_collector_state(conn)
            await conn.execute(text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS news_items_fts "
                "USING fts5(id UNINDEXED, title, content, tokenize='porter unicode61')"
            ))
            # Backfill any rows that pre-date the FTS table
            await conn.execute(text(
                "INSERT INTO news_items_fts(id, title, content) "
                "SELECT id, title, coalesce(content,'') FROM news_items "
                "WHERE id NOT IN (SELECT id FROM news_items_fts)"
            ))


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
