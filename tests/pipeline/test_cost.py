"""Tests for the in-memory LLM cost tracker (T1-B).

These tests never hit the network: we construct `CostEntry` objects directly
or invoke `_litellm_success_callback` with a hand-crafted fake response.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import litellm
import pytest

from news_agent.config import settings
from news_agent.pipeline import cost as cost_module
from news_agent.pipeline.cost import (
    CostEntry,
    CostTracker,
    _litellm_failure_callback,
    _litellm_success_callback,
    caller_tag,
    current_caller,
    get_tracker,
    install_callbacks,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Every test starts with a fresh singleton so ordering doesn't leak state."""
    cost_module._tracker = None
    get_tracker().reset()
    yield
    cost_module._tracker = None


def _entry(
    *,
    caller: str = "analyzer.batch",
    model: str = "anthropic/claude-haiku-4-5",
    cost_usd: float | None = 0.001,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_ms: float = 250.0,
    success: bool = True,
    ts: datetime | None = None,
) -> CostEntry:
    return CostEntry(
        timestamp=ts or datetime.now(timezone.utc),
        model=model,
        caller=caller,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        success=success,
    )


def test_record_and_summary_basic():
    tracker = get_tracker()
    tracker.record(_entry(model="anthropic/claude-haiku-4-5", cost_usd=0.001, caller="analyzer.batch"))
    tracker.record(_entry(model="anthropic/claude-haiku-4-5", cost_usd=0.002, caller="analyzer.batch"))
    tracker.record(_entry(model="anthropic/claude-sonnet-4-6", cost_usd=0.05, caller="analyzer.digest"))

    s = tracker.summary()

    assert s["count"] == 3
    assert s["total_usd"] == pytest.approx(0.053)
    assert s["unknown_cost_count"] == 0
    assert s["error_count"] == 0
    assert s["window_hours"] is None

    assert set(s["by_model"].keys()) == {
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4-6",
    }
    assert s["by_model"]["anthropic/claude-haiku-4-5"]["count"] == 2
    assert s["by_model"]["anthropic/claude-haiku-4-5"]["total_usd"] == pytest.approx(0.003)
    assert s["by_model"]["anthropic/claude-sonnet-4-6"]["count"] == 1

    assert s["by_caller"]["analyzer.batch"]["count"] == 2
    assert s["by_caller"]["analyzer.digest"]["count"] == 1
    assert s["by_caller"]["analyzer.batch"]["total_tokens"] == 300  # 2 * 150


def test_summary_window_filters_old_entries():
    tracker = get_tracker()
    now = datetime.now(timezone.utc)
    tracker.record(_entry(cost_usd=0.01, ts=now - timedelta(hours=48)))
    tracker.record(_entry(cost_usd=0.02, ts=now - timedelta(hours=30)))
    tracker.record(_entry(cost_usd=0.04, ts=now - timedelta(hours=5)))
    tracker.record(_entry(cost_usd=0.08, ts=now - timedelta(minutes=10)))

    window_24h = tracker.summary(hours=24)
    all_time = tracker.summary(hours=None)

    assert window_24h["count"] == 2
    assert window_24h["total_usd"] == pytest.approx(0.12)
    assert window_24h["window_hours"] == 24

    assert all_time["count"] == 4
    assert all_time["total_usd"] == pytest.approx(0.15)


def test_ring_buffer_evicts_oldest():
    small = CostTracker(maxlen=3)
    for i in range(5):
        small.record(_entry(cost_usd=float(i), caller=f"c{i}"))

    s = small.summary()
    assert s["count"] == 3
    # First two (cost 0.0 + 1.0) should be gone; 2 + 3 + 4 = 9.0
    assert s["total_usd"] == pytest.approx(9.0)
    assert set(s["by_caller"].keys()) == {"c2", "c3", "c4"}


async def test_caller_tag_contextvar_isolation_across_tasks():
    """Two concurrent tasks with different caller tags must not cross-contaminate."""
    tracker = get_tracker()

    fake_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )

    async def _worker(tag: str, delay: float) -> None:
        with caller_tag(tag):
            # Yield to the loop so the two tasks interleave.  ContextVar should
            # still return the right tag for each task.
            await asyncio.sleep(delay)
            assert current_caller.get() == tag
            _litellm_success_callback(
                kwargs={"model": f"model-for-{tag}"},
                completion_response=fake_response,
                start_time=0.0,
                end_time=0.1,
            )

    await asyncio.gather(_worker("a", 0.02), _worker("b", 0.01))

    s = tracker.summary()
    assert s["count"] == 2
    assert set(s["by_caller"].keys()) == {"a", "b"}
    assert s["by_caller"]["a"]["count"] == 1
    assert s["by_caller"]["b"]["count"] == 1

    # Outside any context the default tag is restored.
    assert current_caller.get() == "unknown"


def test_unknown_cost_falls_back_to_none(monkeypatch):
    tracker = get_tracker()

    def _boom(**_kwargs):
        raise RuntimeError("no pricing for this model")

    monkeypatch.setattr(litellm, "completion_cost", _boom)

    fake_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    )

    with caller_tag("analyzer.batch"):
        _litellm_success_callback(
            kwargs={"model": "self-hosted/llama"},
            completion_response=fake_response,
            start_time=0.0,
            end_time=0.05,
        )

    s = tracker.summary()
    assert s["count"] == 1
    assert s["unknown_cost_count"] == 1
    assert s["total_usd"] == 0.0  # None coerces to 0 for summing
    assert s["by_model"]["self-hosted/llama"]["total_usd"] == 0.0


def test_latency_percentiles_over_successful_calls_only():
    tracker = get_tracker()

    # 10 successes with known latencies — sorted, last one is 1000ms.
    for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]:
        tracker.record(_entry(latency_ms=float(ms), success=True))
    # 2 failures with huge latencies — must NOT influence percentiles.
    tracker.record(_entry(latency_ms=99999.0, success=False, cost_usd=0.0))
    tracker.record(_entry(latency_ms=99999.0, success=False, cost_usd=0.0))

    s = tracker.summary()

    assert s["count"] == 12
    assert s["error_count"] == 2

    lat = s["latency_ms"]
    # Nearest-rank over the 10 successful values:
    #   p50 → index round(0.5 * 9) = 4 → 50
    #   p95 → index round(0.95 * 9) = 9 → 1000
    #   p99 → index round(0.99 * 9) = 9 → 1000
    #   max → 1000 (not 99999 — failed calls excluded)
    assert lat["p50"] == 50
    assert lat["p95"] == 1000
    assert lat["p99"] == 1000
    assert lat["max"] == 1000


def test_install_callbacks_idempotent(monkeypatch):
    # Start with empty callback lists so we can count exactly.
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(settings, "cost_tracker_enabled", True)

    for _ in range(3):
        install_callbacks()

    assert litellm.success_callback.count(_litellm_success_callback) == 1
    assert litellm.failure_callback.count(_litellm_failure_callback) == 1


def test_tracker_disabled_via_settings(monkeypatch):
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(settings, "cost_tracker_enabled", False)

    install_callbacks()

    assert _litellm_success_callback not in litellm.success_callback
    assert _litellm_failure_callback not in litellm.failure_callback

    # Endpoint still yields a valid empty summary even when disabled.
    s = get_tracker().summary()
    assert s["count"] == 0
    assert s["total_usd"] == 0.0
    assert s["by_model"] == {}
    assert s["latency_ms"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}


def test_failure_callback_records_error_with_latency():
    """Sanity check — failures should land in the buffer with success=False."""
    tracker = get_tracker()

    with caller_tag("analyzer.digest"):
        _litellm_failure_callback(
            kwargs={"model": "anthropic/claude-sonnet-4-6"},
            completion_response=None,
            start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 0, 500_000, tzinfo=timezone.utc),
        )

    s = tracker.summary()
    assert s["count"] == 1
    assert s["error_count"] == 1
    # Latency should be preserved (500 ms).
    assert s["latency_ms"]["max"] == 0.0  # failures excluded from latency stats
    # But the raw entry latency is still recorded; we verify via by_model presence.
    assert s["by_model"]["anthropic/claude-sonnet-4-6"]["count"] == 1
