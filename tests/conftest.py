"""Shared fixtures for the test suite."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from news_agent.models import NewsItem


def make_item(**kwargs) -> NewsItem:
    """Factory for NewsItem with sensible defaults."""
    defaults = dict(
        source="rss",
        topic="ai",
        title="Default test title",
        url="https://example.com/article",
        content="Some content about artificial intelligence and machine learning.",
        published_at=datetime.utcnow(),
        raw_score=0.5,
        relevance_score=5.0,
    )
    defaults.update(kwargs)
    return NewsItem(**defaults)


def hours_ago(n: float) -> datetime:
    return datetime.utcnow() - timedelta(hours=n)


@pytest_asyncio.fixture(autouse=True)
async def isolated_db(tmp_path, monkeypatch):
    """
    Each test gets a fresh in-memory SQLite database.

    Patches news_agent.storage.database and all modules that imported from it
    so no test writes to the production news.db.
    """
    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from news_agent.models import Base
    import news_agent.storage.database as db_module
    import news_agent.storage as storage_pkg

    db_path = tmp_path / "test.db"
    test_url = f"sqlite+aiosqlite:///{db_path}"
    test_engine = create_async_engine(
        test_url,
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 10},
    )

    @event.listens_for(test_engine.sync_engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    TestSessionLocal = async_sessionmaker(
        test_engine, expire_on_commit=False, class_=AsyncSession
    )

    # Create schema
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS news_items_fts "
            "USING fts5(id UNINDEXED, title, content, tokenize='porter unicode61')"
        ))

    @asynccontextmanager
    async def test_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def test_init_db() -> None:
        pass  # already done above

    # Patch db module globals
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "AsyncSessionLocal", TestSessionLocal)
    monkeypatch.setattr(db_module, "get_session", test_get_session)
    monkeypatch.setattr(db_module, "init_db", test_init_db)

    # Patch storage package re-exports
    monkeypatch.setattr(storage_pkg, "get_session", test_get_session)
    monkeypatch.setattr(storage_pkg, "init_db", test_init_db)

    # Patch any module that imported get_session/init_db by name
    for mod_name in (
        "news_agent.web.app",
        "news_agent.pipeline.vector_search",
        "news_agent.orchestrator",
        "news_agent.scheduler",
    ):
        import importlib, sys
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "get_session"):
                monkeypatch.setattr(mod, "get_session", test_get_session)
            if hasattr(mod, "init_db"):
                monkeypatch.setattr(mod, "init_db", test_init_db)

    # Reset the vector index so tests don't see each other's data
    try:
        from news_agent.pipeline.vector_search import _VectorIndex
        import news_agent.pipeline.vector_search as vs_module
        fresh_index = _VectorIndex()
        monkeypatch.setattr(vs_module, "_index", fresh_index)
    except Exception:
        pass

    yield

    await test_engine.dispose()
