# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (includes dev dependencies)
pip install -e ".[dev]"

# Start dev server (auto-reload)
news-agent serve --reload
# or: uvicorn news_agent.web.app:app --reload

# Run all tests
python -m pytest tests/

# Run a single test
python -m pytest tests/storage/test_repository.py::test_upsert_deduplicates_by_id -x

# One-off fetch
news-agent fetch --topics "ai,stocks"

# Background scheduler
news-agent schedule
```

`asyncio_mode = "auto"` is set in `pyproject.toml` — no `@pytest.mark.asyncio` decorator needed on async tests.

## Architecture

The pipeline runs in this order:

1. **Collectors** (`news_agent/collectors/`) — each source (RSS, Reddit, Twitter, YouTube, GitHub) implements `BaseCollector.fetch()`. The RSS collector also handles Google News base64-encoded redirect URLs.

2. **Aggregator** (`pipeline/aggregator.py`) — fans out to all enabled collectors concurrently, swallows per-collector errors, tags languages, and returns a flat list sorted newest-first.

3. **Deduplicator** (`pipeline/deduplicator.py`) — semantic cosine-similarity dedup (threshold 0.82) + title normalization to catch cross-publisher duplicates.

4. **Storage** (`storage/repository.py`) — `NewsRepository` wraps all SQLite ops. `upsert()` is idempotent: item ID is `sha256(f"{source}:{url}")[:16]`. Analysis fields (`summary`, `relevance_score`, `sentiment`, `tags`, `key_entities`) are **write-once** — the `on_conflict_do_update` set intentionally excludes them so re-fetching never overwrites LLM results. `update_analysis_many()` is the only path that writes them.

5. **LLM Analyzer** (`pipeline/analyzer.py`) — `LLMAnalyzer.analyze_batch()` scores items 0–10 for relevance, sentiment, tags, entities. Uses litellm so any provider works. `generate_digest_stream()` / `answer_question_stream()` are async generators for SSE.

6. **Web** (`web/app.py`) — FastAPI + Jinja2. Digest generation and search stream via SSE. The Jinja2 environment is constructed directly (not via Starlette's wrapper) to avoid a Python 3.14 cache-key bug.

## Search

Hybrid search: BM25 (SQLite FTS5) + semantic (all-MiniLM-L6-v2 embeddings) merged via Reciprocal Rank Fusion (`_rrf_merge` in `repository.py`). The vector index (`pipeline/vector_search.py`) is in-memory with a 10-min TTL and disk cache at `data/vector_index.npz`. Call `invalidate_index()` after bulk ingest.

## User Preferences

Interactions (upvote/downvote/click/read) are stored in `UserInteractionORM` and drive `recompute_preferences()` in `preference.py`. Upvote and downvote are **mutually exclusive** — `_cancel_opposite_vote()` in `app.py` auto-records the cancel action before writing the new vote.

## Testing

All async tests use the `isolated_db` autouse fixture in `tests/conftest.py`, which creates a fresh in-memory SQLite per test and patches all module-level `get_session`/`init_db` references. This prevents test pollution of `data/news.db`.

Use `make_item(**kwargs)` from `tests/conftest.py` to create `NewsItem` instances with sensible defaults. FastAPI endpoint tests use `httpx.AsyncClient` with `ASGITransport(app=app)`.

## Configuration

Copy `.env.example` to `.env`. One LLM key is required (`LLM_MODEL` + `LLM_API_KEY`). All source API keys are optional — sources without keys are auto-disabled. Config is loaded via pydantic-settings (`news_agent/config.py`).
