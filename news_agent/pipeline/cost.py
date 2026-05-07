"""In-memory LLM cost & latency tracker.

Wires up litellm success/failure callbacks so every `litellm.completion` or
`litellm.acompletion` call made anywhere in the codebase is observed and
recorded in a bounded ring buffer.  A `caller_tag("name")` context manager
lets call sites attach a human-readable label (e.g. "analyzer.batch") that
propagates through the async stack via `contextvars.ContextVar`.

Persistence is explicitly out of scope — this module is in-memory only.
See T1-B spec (issue #2) for the accompanying `/api/cost/summary` endpoint.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import litellm

from news_agent.config import settings

logger = logging.getLogger(__name__)


# ── Caller tagging ──────────────────────────────────────────────────────────

# ContextVar propagates across `await` boundaries and is isolated per
# asyncio Task, which is exactly what we want: two concurrent digest streams
# can each tag their own litellm calls without stepping on each other.
current_caller: ContextVar[str] = ContextVar("current_caller", default="unknown")


@contextmanager
def caller_tag(name: str) -> Iterator[None]:
    """Attach a caller label to any litellm calls made inside this block.

    Works for both sync and async code — `ContextVar.set()` / `reset()` is
    the canonical way to scope a value to the current task.  Enter the
    context *before* the `await` so the callback (which may fire on any
    thread) can read the correct caller via `current_caller.get()`.
    """
    token = current_caller.set(name)
    try:
        yield
    finally:
        current_caller.reset(token)


# ── Data model ──────────────────────────────────────────────────────────────


@dataclass
class CostEntry:
    """One observed LLM call."""

    timestamp: datetime
    model: str
    caller: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None  # None when litellm couldn't price the model
    latency_ms: float
    success: bool


# ── Tracker ─────────────────────────────────────────────────────────────────


class CostTracker:
    """Bounded thread-safe ring buffer of `CostEntry` records.

    Uses `threading.Lock` rather than `asyncio.Lock` because litellm
    callbacks can be invoked from worker threads (e.g. when streaming
    responses are joined in `litellm.Router`), not just the event loop.
    """

    def __init__(self, maxlen: int) -> None:
        self._entries: deque[CostEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, entry: CostEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def reset(self) -> None:
        """Clear the buffer.  Intended for tests."""
        with self._lock:
            self._entries.clear()

    def _snapshot(self) -> list[CostEntry]:
        with self._lock:
            return list(self._entries)

    def summary(self, hours: float | None = None) -> dict[str, Any]:
        entries = self._snapshot()
        if hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            entries = [e for e in entries if _to_utc(e.timestamp) >= cutoff]

        by_model: dict[str, dict[str, float | int]] = {}
        by_caller: dict[str, dict[str, float | int]] = {}
        total_usd = 0.0
        unknown_cost_count = 0
        error_count = 0
        successful_latencies: list[float] = []

        for e in entries:
            cost = e.cost_usd if e.cost_usd is not None else 0.0
            total_usd += cost
            if e.cost_usd is None:
                unknown_cost_count += 1
            if not e.success:
                error_count += 1
            else:
                successful_latencies.append(e.latency_ms)

            m = by_model.setdefault(
                e.model, {"count": 0, "total_usd": 0.0, "total_tokens": 0}
            )
            m["count"] = int(m["count"]) + 1
            m["total_usd"] = float(m["total_usd"]) + cost
            m["total_tokens"] = int(m["total_tokens"]) + e.total_tokens

            c = by_caller.setdefault(
                e.caller, {"count": 0, "total_usd": 0.0, "total_tokens": 0}
            )
            c["count"] = int(c["count"]) + 1
            c["total_usd"] = float(c["total_usd"]) + cost
            c["total_tokens"] = int(c["total_tokens"]) + e.total_tokens

        latency_stats = _percentiles(successful_latencies)

        return {
            "window_hours": hours,
            "count": len(entries),
            "total_usd": total_usd,
            "unknown_cost_count": unknown_cost_count,
            "by_model": by_model,
            "by_caller": by_caller,
            "latency_ms": latency_stats,
            "error_count": error_count,
        }


def _to_utc(ts: datetime) -> datetime:
    """Normalise to aware-UTC so naive timestamps compare against `datetime.now(tz)`."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _q(p: float) -> float:
        # Nearest-rank percentile; stable and numpy-free.  idx ∈ [0, n-1].
        if n == 1:
            return sorted_vals[0]
        idx = int(round(p * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return sorted_vals[idx]

    return {
        "p50": _q(0.50),
        "p95": _q(0.95),
        "p99": _q(0.99),
        "max": sorted_vals[-1],
    }


# ── Singleton ───────────────────────────────────────────────────────────────

_tracker: CostTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> CostTracker:
    """Return the process-wide singleton tracker (lazy-init)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = CostTracker(maxlen=settings.cost_tracker_max_entries)
    return _tracker


# ── litellm callbacks ───────────────────────────────────────────────────────


def _extract_usage(completion_response: Any) -> tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens) defensively.

    litellm returns a pydantic ModelResponse; older versions may give a dict.
    Streaming responses don't always carry usage on the final chunk, so we
    default to zeros rather than raising.
    """
    if completion_response is None:
        return 0, 0, 0
    usage = getattr(completion_response, "usage", None)
    if usage is None and isinstance(completion_response, dict):
        usage = completion_response.get("usage")
    if usage is None:
        return 0, 0, 0

    def _read(key: str) -> int:
        val = None
        if hasattr(usage, key):
            val = getattr(usage, key)
        elif isinstance(usage, dict):
            val = usage.get(key)
        try:
            return int(val) if val is not None else 0
        except (TypeError, ValueError):
            return 0

    return _read("prompt_tokens"), _read("completion_tokens"), _read("total_tokens")


def _compute_latency_ms(start_time: Any, end_time: Any) -> float:
    """Compute latency in ms across litellm's assorted time representations.

    Across versions we've seen `datetime.datetime`, `float` (epoch seconds),
    and occasionally `int` (ms).  Handle all three.
    """
    try:
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            return (end_time - start_time).total_seconds() * 1000.0
        if start_time is None or end_time is None:
            return 0.0
        # Numeric path — assume seconds; if the delta looks like ms already
        # (huge value) it still round-trips to something reasonable for stats.
        return float(end_time - start_time) * 1000.0
    except Exception:
        return 0.0


def _litellm_success_callback(
    kwargs: dict[str, Any],
    completion_response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """litellm success hook — record one entry with token counts and cost."""
    try:
        model = str(kwargs.get("model", "unknown")) if kwargs else "unknown"
        caller = current_caller.get()
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(completion_response)

        cost_usd: float | None
        try:
            cost_usd = float(
                litellm.completion_cost(completion_response=completion_response)
            )
        except Exception:
            # Unknown / free models (Groq free tier, self-hosted) have no pricing.
            cost_usd = None

        latency_ms = _compute_latency_ms(start_time, end_time)

        get_tracker().record(
            CostEntry(
                timestamp=datetime.now(timezone.utc),
                model=model,
                caller=caller,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                success=True,
            )
        )
    except Exception:
        # A tracking hook must never break the caller.
        logger.exception("cost tracker success callback failed")


def _litellm_failure_callback(
    kwargs: dict[str, Any],
    completion_response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """litellm failure hook — record a zero-cost entry preserving latency."""
    try:
        model = str(kwargs.get("model", "unknown")) if kwargs else "unknown"
        caller = current_caller.get()
        latency_ms = _compute_latency_ms(start_time, end_time)
        get_tracker().record(
            CostEntry(
                timestamp=datetime.now(timezone.utc),
                model=model,
                caller=caller,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
                success=False,
            )
        )
    except Exception:
        logger.exception("cost tracker failure callback failed")


def install_callbacks() -> None:
    """Append our hooks to `litellm.{success,failure}_callback` idempotently.

    No-op when `settings.cost_tracker_enabled` is False or when our callback
    is already registered (safe to call from every importer).
    """
    if not settings.cost_tracker_enabled:
        return

    success_list = getattr(litellm, "success_callback", None)
    if isinstance(success_list, list) and _litellm_success_callback not in success_list:
        success_list.append(_litellm_success_callback)

    failure_list = getattr(litellm, "failure_callback", None)
    if isinstance(failure_list, list) and _litellm_failure_callback not in failure_list:
        failure_list.append(_litellm_failure_callback)
