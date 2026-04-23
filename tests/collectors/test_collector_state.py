from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from news_agent.collectors.base import BaseCollector
from news_agent.models import CollectorStateORM, NewsItem
import news_agent.storage as storage_pkg
import news_agent.storage.database as db_module
import news_agent.storage.repository as repository_module


class _FakeState(BaseModel):
    cursor: str = ""
    seen: int = 0


class _StatefulCollector(BaseCollector):
    source_name = "stateful"
    StateSchema = _FakeState

    async def fetch(self) -> list[NewsItem]:
        return []


@pytest.fixture(autouse=True)
def isolated_db():
    """Override the global DB fixture for this module."""
    yield


class _FakeCollectorStateRepo:
    def __init__(self, _session, store: dict[str, CollectorStateORM | None]) -> None:
        self._store = store

    async def update_collector_state(
        self,
        source: str,
        last_run: datetime | None = None,
        last_error: str | None = None,
        items_fetched: int = 0,
        is_enabled: bool | None = None,
        state: dict | None = None,
    ) -> None:
        existing = self._store.get(source)
        if existing is None:
            existing = CollectorStateORM(source=source)
            self._store[source] = existing
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
        return self._store.get(source)


@pytest.fixture
def collector_repo_boundary(monkeypatch):
    store: dict[str, CollectorStateORM | None] = {}

    @asynccontextmanager
    async def fake_get_session():
        yield object()

    def fake_repo_factory(session):
        return _FakeCollectorStateRepo(session, store)

    monkeypatch.setattr(storage_pkg, "get_session", fake_get_session)
    monkeypatch.setattr(repository_module, "NewsRepository", fake_repo_factory)
    return store


@pytest.mark.asyncio
async def test_state_schema_roundtrips(collector_repo_boundary):
    collector = _StatefulCollector(topics=[])
    saved = await collector.save_state({"cursor": "abc123", "seen": 7})
    loaded = await collector.load_state()

    assert isinstance(saved, _FakeState)
    assert isinstance(loaded, _FakeState)
    assert loaded.cursor == "abc123"
    assert loaded.seen == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", [None, {}])
async def test_state_default_when_missing_or_empty(collector_repo_boundary, persisted):
    if persisted is not None:
        collector_repo_boundary["stateful"] = CollectorStateORM(source="stateful", state=persisted)

    collector = _StatefulCollector(topics=[])
    loaded = await collector.load_state()

    assert isinstance(loaded, _FakeState)
    assert loaded.cursor == ""
    assert loaded.seen == 0


@pytest.mark.asyncio
async def test_state_validation_rejects_garbage(collector_repo_boundary):
    collector_repo_boundary["stateful"] = CollectorStateORM(
        source="stateful",
        state={"cursor": ["bad"], "seen": "oops"},
    )

    collector = _StatefulCollector(topics=[])
    with pytest.raises(ValidationError):
        await collector.load_state()


class _FakeResult:
    def __init__(self, *, scalar=None, rows=None) -> None:
        self._scalar = scalar
        self._rows = rows or []

    def scalar_one_or_none(self):
        return self._scalar

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, *, table_exists: bool, has_state_column: bool, rows) -> None:
        self.table_exists = table_exists
        self.has_state_column = has_state_column
        self.rows = rows
        self.updated_payloads: dict[str, str] = {}
        self.added_state_column = False

    async def execute(self, statement, params=None):
        sql = str(statement)
        if "sqlite_master" in sql:
            return _FakeResult(scalar="collector_state" if self.table_exists else None)
        if "PRAGMA table_info(collector_state)" in sql:
            cols = [("source",), ("last_run",), ("youtube_quota_used",)]
            if self.has_state_column:
                cols.append(("state",))
            return _FakeResult(rows=[(0, name) for (name,) in cols])
        if "ALTER TABLE collector_state ADD COLUMN state" in sql:
            self.added_state_column = True
            self.has_state_column = True
            return _FakeResult()
        if "SELECT source, youtube_quota_used, youtube_quota_reset_date" in sql:
            return _FakeResult(rows=self.rows)
        if "UPDATE collector_state SET state = :state WHERE source = :source" in sql:
            assert params is not None
            self.updated_payloads[params["source"]] = params["state"]
            return _FakeResult()
        raise AssertionError(f"Unexpected SQL: {sql}")


@pytest.mark.asyncio
async def test_youtube_backfill_transforms_legacy_columns_to_json_state():
    row = SimpleNamespace(
        source="youtube",
        youtube_quota_used=42,
        youtube_quota_reset_date="2026-04-30",
    )
    conn = _FakeConn(table_exists=True, has_state_column=True, rows=[row])

    await db_module._migrate_collector_state(conn)

    assert conn.updated_payloads == {
        "youtube": json.dumps(
            {
                "youtube_quota_used": 42,
                "youtube_quota_reset_date": "2026-04-30",
            }
        )
    }


@pytest.mark.asyncio
async def test_youtube_backfill_adds_state_column_before_transforming():
    row = SimpleNamespace(
        source="youtube",
        youtube_quota_used=7,
        youtube_quota_reset_date=None,
    )
    conn = _FakeConn(table_exists=True, has_state_column=False, rows=[row])

    await db_module._migrate_collector_state(conn)

    assert conn.added_state_column is True
    assert conn.updated_payloads == {
        "youtube": json.dumps(
            {
                "youtube_quota_used": 7,
                "youtube_quota_reset_date": None,
            }
        )
    }
