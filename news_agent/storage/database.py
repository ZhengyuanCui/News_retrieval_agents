from __future__ import annotations

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


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if _is_sqlite:
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
