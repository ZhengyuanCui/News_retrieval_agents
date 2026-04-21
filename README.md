# News Retrieval Agent

An AI-powered news aggregator that collects articles, tweets, videos, and code from multiple sources, deduplicates and ranks them, generates AI-powered summaries, and optionally emails you a daily personalized newsletter with MP3 narration — all served through a real-time web UI. Supports any LLM via [litellm](https://docs.litellm.ai): Anthropic Claude, OpenAI GPT, Groq, Google Gemini, Mistral, and more.

See also the [USER_GUIDE.md](./USER_GUIDE.md) for task-oriented walkthroughs.

## Features

- **Multi-source collection** — RSS / Google News / Bing News feeds (including ~140 curated frontier-lab, arXiv, and researcher-blog feeds), X/Twitter, Reddit, YouTube, and GitHub.
- **Any LLM via litellm** — one `LLM_MODEL` + `LLM_API_KEY` pair works for Anthropic, OpenAI, Gemini, Groq, Mistral, etc. A separate cheaper `ANALYSIS_MODEL` (default: Haiku) handles bulk item scoring; optional round-robin across multiple analysis keys for RPM scaling.
- **AI analysis** — scores each item for relevance (0–10), sentiment, tags, and key entities using the analysis LLM.
- **Hybrid retrieval (BM25 + semantic, RRF merged)** — SQLite FTS5 full-text + `all-MiniLM-L6-v2` vector index, fused via Reciprocal Rank Fusion. Question-shaped queries are detected automatically and stream a grounded LLM answer.
- **Source-authority ranking** — a tiered authority dictionary weights frontier-lab blogs / arXiv / researcher personal sites up, and wire services / news aggregators (Google News, Bing News, Reuters, Bloomberg, BBC, TechCrunch, …) down. Top-20 results are re-ranked with the `ms-marco-MiniLM-L-6-v2` cross-encoder when available.
- **Real-time digest streaming** — summaries and Q&A stream token-by-token over Server-Sent Events.
- **User preference learning** — upvotes / downvotes / clicks / read-time feed `UserPreferenceORM` weights that boost or demote future items by tag, source, and entity.
- **Language filtering** — filter by ISO 639-1 code; non-English YouTube videos are excluded.
- **Spam filtering** — engagement floors, ML spam classifier (`mshenoda/roberta-spam`, 125 M RoBERTa), keyword hard-blocks, influencer-shill pattern detection, and YouTube-specific ticker-stuffing regex.
- **Deduplication** — semantic cosine-similarity dedup (threshold 0.82) plus title-suffix normalization to catch cross-publisher duplicates.
- **Podcast** — convert any panel's items into a spoken audio briefing (OpenAI TTS, falls back to gTTS).
- **Daily newsletter (NEW)** — at a configurable time (default 07:00 in your `SCHEDULER_TIMEZONE`), the app runs a full fresh fetch for your saved default topics, waits for LLM analysis to populate summaries / tags / sentiment, and emails you an HTML digest with one MP3 narration per topic attached. See [Daily Newsletter](#daily-newsletter) below.
- **Scheduled background jobs** — automatic fetch every N hours, daily digest, daily newsletter, and daily prune.
- **Export** — dump items to JSON or Markdown.

---

## Project Structure

```
news_agent/
├── __init__.py
├── cli.py                # Click CLI entry point: fetch, schedule, serve, digest,
│                         # analyze, refetch, export, status, sources, newsletter
├── config.py             # Pydantic-settings config (reads .env)
├── models.py             # Pydantic NewsItem + SQLAlchemy ORM models
│                         # (NewsItemORM, DigestORM, UserInteractionORM,
│                         #  UserPreferenceORM, UserSettingORM, CollectorStateORM)
├── orchestrator.py       # run_fetch_cycle, run_keyword_fetch,
│                         # fetch_and_analyze_topics (for newsletter),
│                         # generate_digest, background analysis + backfill loop
├── scheduler.py          # APScheduler jobs: fetch / digest / newsletter / prune
├── emailer.py            # SMTP client (STARTTLS + implicit SSL, per-attachment Content-ID)
├── preference.py         # Interaction → preference score aggregation + boost
├── spam.py               # mshenoda/roberta-spam classifier wrapper
├── lang.py               # langdetect wrapper
│
├── collectors/           # One collector per source; BaseCollector handles rate
│   ├── base.py           #   limiting, language tagging, score normalization
│   ├── rss.py            # RSS/Atom + Google News + Bing News (+ ~140 curated feeds)
│   ├── twitter.py        # X/Twitter API v2 via tweepy (ML-filtered)
│   ├── reddit.py         # Reddit via praw
│   ├── youtube.py        # YouTube Data API v3 (with spam filters)
│   └── github.py         # GitHub trending / search via PyGithub
│
├── pipeline/
│   ├── aggregator.py     # Fan-out: runs all collectors in parallel
│   ├── analyzer.py       # LLM batch analysis + digest generation (streaming),
│   │                     # multi-model rotation with weighted RPM balancing
│   ├── deduplicator.py   # Semantic dedup via sentence-transformers + cosine
│   ├── embeddings.py     # Shared all-MiniLM-L6-v2 singleton (thread-safe)
│   ├── ranker.py         # Blended semantic + LLM + authority + freshness + engagement
│   │                     # re-ranking, with optional cross-encoder rerank
│   ├── vector_search.py  # In-memory 7-day vector index (10-min TTL, disk cache)
│   ├── podcast.py        # OpenAI TTS (→ gTTS fallback) podcast generator
│   └── newsletter.py     # Build+email the daily newsletter: per-topic audio,
│                         # HTML rendering, fresh-fetch orchestration
│
├── storage/
│   ├── database.py       # SQLAlchemy async engine + session factory (SQLite WAL)
│   ├── repository.py     # CRUD: items, digests, user settings, interactions,
│   │                     # collector state; hybrid BM25+semantic search
│   └── exporter.py       # JSON / Markdown export
│
└── web/
    ├── app.py            # FastAPI app — REST + SSE + Jinja2 pages
    ├── static/
    │   ├── app.js        # Panel auto-fetch, SSE digest/Q&A client, podcast polling
    │   └── style.css
    └── templates/
        ├── digest.html           # Two-panel digest page (the only page)
        └── partials/             # Jinja2 fragments for AJAX panel updates

alembic/                  # Database migrations (schema changes are also
                          # handled by best-effort ALTERs in web/app.py startup)
tests/                    # pytest suite (335 tests; see "Running Tests")
data/                     # Runtime state (gitignored — created on first run):
                          #   news.db, news.db-{shm,wal}   SQLite WAL journal
                          #   vector_index.{npz,json}      disk cache for semantic search
                          #   newsletter_audio/            briefing MP3s served at /newsletter/audio/
                          #   podcasts/                    cached podcast MP3s
                          #   exports/                     `news-agent export` output
```

> **What's not in git:** `.env` (your secrets), `data/` (runtime state — DB, embeddings, generated audio, exports), `.venv/`, `__pycache__/`, IDE configs, and OS junk. See `.gitignore` for the full list. The app creates `data/` on first run, so a fresh clone just needs `.env` filled in.

---

## Quick Start

### 1. Install

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

> If PowerShell blocks activation with *"running scripts is disabled on this system"*:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Python 3.12+ is required.

### 2. Configure

```bash
cp .env.example .env
```

Fill in the keys you want to use. **Only one LLM key is required** (for summaries, digests, and Q&A). All source API keys are optional — sources without credentials auto-disable.

#### LLM provider (pick one)

```env
LLM_MODEL=anthropic/claude-sonnet-4-6
LLM_API_KEY=<your-key>
```

| Provider | `LLM_MODEL` example | Where to get a key |
|----------|---------------------|--------------------|
| Anthropic (default) | `anthropic/claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | `openai/gpt-4o` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Groq (fast, free tier) | `groq/llama-3.3-70b-versatile` | [console.groq.com](https://console.groq.com) |
| Google Gemini | `gemini/gemini-2.0-flash` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Mistral | `mistral/mistral-large-latest` | [console.mistral.ai](https://console.mistral.ai) |

If `ANTHROPIC_API_KEY` is already set in your environment and you don't specify `LLM_MODEL` / `LLM_API_KEY`, the default Anthropic model keeps working.

#### Cheap "analysis" model (optional but recommended)

Bulk scoring (relevance / sentiment / tags) runs on a separate, cheaper model — by default Claude Haiku. Set `ANALYSIS_MODEL` + `ANALYSIS_API_KEY` to override (e.g. Gemini Flash on its free tier). You can also rotate across multiple analysis keys to multiply your effective RPM:

```env
ANALYSIS_MODELS=gemini/gemini-2.5-flash,gemini/gemini-2.5-flash
ANALYSIS_API_KEYS=key_account_a,key_account_b
ANALYSIS_RPMS=10,10
ANALYSIS_CONCURRENCY=2
```

#### Other keys (all optional)

| Key | Purpose |
|-----|---------|
| `TWITTER_BEARER_TOKEN` | Enables the Twitter/X collector |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` + `REDDIT_USERNAME` | Enables Reddit |
| `YOUTUBE_API_KEY` | Enables YouTube (curated channel list in `config.py`) |
| `GITHUB_TOKEN` | Raises GitHub API rate limits |
| `OPENAI_API_KEY` | Higher-quality podcast / newsletter TTS (falls back to free gTTS when absent) |
| `GEMINI_API_KEY` | Convenience shortcut for Gemini analysis models |

### 3. Run

```bash
# Web UI (http://localhost:8000) — also starts the scheduler inline
news-agent serve

# Dev mode with auto-reload
news-agent serve --reload

# One-off fetch from all enabled sources
news-agent fetch

# Fetch specific topics only
news-agent fetch --topics "ai,stocks"

# Print a digest to the terminal
news-agent digest --topic ai --hours 24

# Backfill LLM summaries for items currently missing one
news-agent analyze

# Export items to Markdown
news-agent export --format markdown

# Run the scheduler standalone (not needed if you're running `serve`)
news-agent schedule

# Email a newsletter right now (test SMTP before relying on the cron job)
news-agent newsletter --to me@example.com
news-agent newsletter --no-refresh           # use items already in the DB (fast)
```

All commands are also available as `python -m news_agent.cli <cmd>`.

---

## Web UI

Open `http://localhost:8000`. The only page is a two-panel digest; both panels can show a different topic side-by-side (`?topic1=ai&topic2=stocks`).

- **Search bar in each panel** — type a keyword *or* a natural-language question. Short keywords fetch + display a curated list plus a streamed AI digest; question-shaped queries stream a grounded answer from the last 7 days of articles.
- **⚙ Settings** — save default topics that persist in `localStorage` **and** the server-side `user_settings` table (so the newsletter scheduler can read them). Save & Apply does *not* refresh the panels. Includes a **Send test now** button to email the newsletter immediately.
- **🌐 Language filter** — restrict the UI to selected ISO language codes.
- **🎙 Podcast** — generate a spoken audio briefing for a topic.
- **👍 / 👎** — vote on items to teach the ranking what you want more or less of. Upvote and downvote are mutually exclusive; switching auto-cancels the previous vote.

---

## Search & Q&A

### Keyword search

Type a short keyword (`AI`, `NVDA`, `Fed rates`) to see a ranked list of articles plus an AI-generated digest at the top.

### Question answering

Type a natural-language question (`Will the stock market rise if the Iran war ends?`). Detection fires on queries that start with a question word, end with `?`, or have 7+ words. The system:

1. Retrieves the most semantically relevant articles from the last 7 days via vector similarity.
2. Streams a grounded LLM answer — a one-sentence response followed by bullet points citing specific articles by source.
3. Is honest about gaps: if the retrieved articles don't contain enough information, it says so rather than hallucinating.

### How retrieval works

Hybrid search runs two passes in parallel and fuses them via **Reciprocal Rank Fusion**:

1. **BM25 keyword search** — SQLite FTS5 full-text index over title + content.
2. **Semantic vector search** — query embedded with `all-MiniLM-L6-v2`, compared against an in-memory index of all non-duplicate articles from the last 7 days.

The vector index is built lazily on first use (~2–3 s once the model is warm), cached to `data/vector_index.npz`, invalidated after every pipeline ingest, and refreshed every 10 minutes in the background.

### How results are ranked

Each retrieved article gets a blended score:

```
score = 0.35 × semantic_similarity    (embedding cosine sim to query)
      + 0.25 × llm_relevance          (LLM 0–10 score normalised to 0–1; 0.5 if unscored)
      + 0.20 × source_authority       (tiered editorial weight per source)
      + 0.10 × freshness              (exponential decay, half-life 48 h)
      + 0.10 × raw_engagement         (likes/retweets/views, normalised)
```

Items with an LLM score below `4.0/10` are excluded; un-scored items (NULL) are always included so fresh articles appear before analysis completes.

**Source authority** lives in `_SOURCE_AUTHORITY` in `news_agent/pipeline/ranker.py`:

| Tier | Example sources | Weight |
|------|-----------------|--------|
| S — Frontier AI labs | `openai`, `anthropic`, `deepmind`, `meta-ai`, `apple-ml`, `mistral`, `cohere`, `huggingface`, `nvidia-ai` | 0.90 – 1.00 |
| A — Primary research / academic labs | `arxiv`, `mit`, Stanford HAI, researcher personal blogs | 0.80 – 0.90 |
| B — Official developer & platform blogs | `github-blog`, `aws-ml`, `azure-ai`, … | 0.70 – 0.80 |
| C — Reputable business & tech press | `bloomberg`, `reuters`, `wsj`, `ft`, `cnbc` (0.75), `techcrunch` / `wired` (0.60), `bbc` (0.50) | 0.50 – 0.75 |
| D — Syndicated search aggregators | `google-news`, `bing-news` | 0.25 |

Unknown sources default to `0.55`. Top-20 results are further re-ranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` when `sentence-transformers` is available.

---

## Daily Newsletter

At your configured time (default **07:00 in `SCHEDULER_TIMEZONE`**, default `America/Los_Angeles` = PST/PDT) the scheduler:

1. **Fetches fresh news** for each saved default topic (every enabled source, deduped).
2. **Waits for LLM analysis** so every item has a summary, relevance score, sentiment, and tags.
3. **Emails** a multipart HTML message containing:
   - a Claude-generated narrative digest per topic,
   - the same article cards you see in the web UI (source badge, sentiment, tags, direct links),
   - **one MP3 narration per topic**, embedded with an inline HTML5 `<audio>` player and a fallback "Play briefing in browser" link. OpenAI TTS is used when `OPENAI_API_KEY` is set, otherwise the free gTTS fallback.

Because fetch + analysis happens inline, the job takes a few minutes; expect the email slightly after `NEWSLETTER_HOUR`. If the pre-fetch fails (e.g. a source is down) the newsletter still sends with whatever is currently in the DB rather than skipping the day.

### Inline audio playback

The `<audio>` player's behaviour depends on **how recipients' mail clients are allowed to reach the MP3**:

| Mail client | Plays inline when… |
|---|---|
| Apple Mail, Thunderbird | Always — they render `<audio>` tags natively from the `cid:` attachment |
| Gmail web/mobile, Outlook web/desktop, Yahoo | Only when `PUBLIC_BASE_URL` is set to a host the client can fetch from HTTPS. Without it the MP3 is delivered only as a downloadable attachment. |

Set `PUBLIC_BASE_URL` in `.env` to the internet-reachable origin of your running app — e.g. `https://news.example.com`, a Fly.io/Render deployment, a Cloudflare Tunnel, or an ngrok URL for testing. No trailing slash, HTTPS strongly recommended (Gmail blocks mixed-content audio). When set, each briefing is written to `NEWSLETTER_AUDIO_DIR` and served inline from:

- `GET /newsletter/audio/<file>.mp3` — the MP3 stream itself, sent with `Content-Disposition: inline` and HTTP Range support so `<audio>` can seek/stream.
- `GET /newsletter/player/<slug>` — a tiny HTML landing page with `<audio controls autoplay>`. The email's "Play briefing in browser" link points here so the click target never ends in `.mp3` (which some browsers auto-download regardless of headers).

### Setup

1. Save default topics in the web UI (⚙ → Panel 1 / Panel 2 → Save & Apply). These sync to the server-side `user_settings` table so the scheduler reads the exact topics you picked.
2. Configure SMTP in `.env` — see the table below. For Gmail, `SMTP_PASSWORD` must be a **16-character App Password** (create one at https://myaccount.google.com/apppasswords with 2-Step Verification enabled), *not* your normal login password.
3. (Optional, for inline playback in Gmail/Outlook) set `PUBLIC_BASE_URL` to a URL where your server is reachable from the public internet.
4. Run `news-agent newsletter-preview --to you@example.com` to send a dummy-content preview and verify the layout + player without waiting for a real fetch + LLM + TTS cycle.
5. Or click **Send test now** in the ⚙ panel / run `news-agent newsletter --to you@example.com` for a real end-to-end send.
6. Restart `news-agent serve` so the scheduler picks up the new cron time.

Topic resolution order: `--topics` flag → `NEWSLETTER_TOPICS` env var → UI-saved defaults. Set `NEWSLETTER_ENABLED=true` to activate the daily cron.

---

## Spam Filtering

Spam is caught in layers, applied in order of cost:

| Layer | Method | Applies to |
|-------|--------|-----------|
| Engagement floor | Like + retweet / view count < threshold | Twitter, YouTube |
| Hard keyword blocks | Exact phrase match (e.g. "guaranteed profit", "copy my trades") | Twitter |
| Shill-combo detection | `@mention` + `$cashtag` + ≥ 2 soft indicators | Twitter |
| Ticker stuffing | ≥ 3 distinct `$TICKER` cashtags, or 4+ slash-separated tickers | Twitter, YouTube |
| ML classifier | `mshenoda/roberta-spam` (125 M RoBERTa, threshold 0.80) | Twitter |
| YouTube-specific | Spam phrase list + `#/$` symbol count + ticker-list regex | YouTube |

The ML model is loaded once at startup and shared across all requests. Batch inference is used for Twitter (one forward pass per query batch).

---

## Deduplication

- **ID dedup** — item ID is `sha256(f"{source}:{url}")[:16]`, so identical (source, url) pairs collapse at upsert.
- **Semantic dedup** — article embeddings are compared pairwise; cosine similarity ≥ 0.82 → duplicate.
- **Title-suffix normalisation** — trailing ` - Publication Name` suffixes are stripped before hashing so "Article Title - Reuters" and "Article Title" are recognised as the same story.
- **Analysis fields are write-once** — summary / tags / sentiment / relevance / key_entities are never overwritten by a re-fetch (only `update_analysis_many()` writes them).

---

## CLI Reference

```
news-agent fetch                         Run a single fetch cycle
news-agent fetch --topics "ai,stocks"    Fetch specific topics only
news-agent refetch                       Clear DB and re-fetch with current filters
news-agent refetch --yes                 Skip confirmation prompt
news-agent analyze                       Backfill LLM summaries for items with none
news-agent analyze --topic ai            Analyze only items for one topic
news-agent schedule                      Start the background scheduler standalone
news-agent serve                         Start the web server (scheduler runs inline)
news-agent serve --reload                Dev mode with auto-reload
news-agent digest --topic ai             Generate and print a topic digest
news-agent export --format markdown      Export items to Markdown
news-agent export --format json          Export items to JSON
news-agent status                        Show database stats and per-source status
news-agent sources list                  List all sources and their status
news-agent sources enable <name>         Enable a source
news-agent sources disable <name>        Disable a source
news-agent newsletter --to <email>       Fetch → analyze → email newsletter now
news-agent newsletter --no-refresh       Skip the fetch; send using cached DB items
news-agent newsletter --no-audio         Skip the MP3 attachment
news-agent newsletter --topics "ai,stocks" --hours 48
news-agent newsletter-preview            Send a dummy-content preview email (no fetch / LLM / TTS)
news-agent newsletter-preview --no-audio                   Preview HTML layout only
news-agent newsletter-preview --audio-file briefing.mp3    Preview with your own MP3
news-agent newsletter-preview --dump-html /tmp/preview.html  Also save the rendered HTML
```

---

## Configuration

All settings live in `.env`. Full reference:

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `anthropic/claude-sonnet-4-6` | Main LLM for digests, Q&A, streaming (litellm format) |
| `LLM_API_KEY` | _(empty)_ | API key for the main LLM (overrides provider env vars) |
| `ANTHROPIC_API_KEY` | _(empty)_ | Legacy fallback; picked up by litellm for `anthropic/*` models |
| `ANALYSIS_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Cheaper model for bulk item scoring |
| `ANALYSIS_API_KEY` | _(empty)_ | API key for the analysis model (if different provider) |
| `ANALYSIS_MODELS` | _(empty)_ | Comma-separated rotation pool for analysis |
| `ANALYSIS_API_KEYS` | _(empty)_ | Parallel to `ANALYSIS_MODELS` |
| `ANALYSIS_RPMS` | _(empty)_ | Parallel RPM caps; enables weighted batch distribution |
| `ANALYSIS_CONCURRENCY` | `5` | Max concurrent LLM calls across all analysis models |
| `BATCH_SIZE` | `15` | Items per LLM analysis batch |
| `MAX_ITEMS_PER_SOURCE` | `50` | Items fetched per source per cycle |
| `DEDUP_STRATEGY` | `semantic` | `semantic` \| `tfidf` \| `url_only` |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.82` | Cosine threshold for semantic dedup |

### Storage & scheduler

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///data/news.db` | SQLAlchemy DB URL |
| `RETENTION_DAYS` | `30` | Days before items are pruned |
| `SCHEDULE_INTERVAL_HOURS` | `4` | How often the background fetch runs |
| `SCHEDULER_TIMEZONE` | `America/Los_Angeles` | IANA tz for all cron jobs |

### Sources

| Variable | Description |
|----------|-------------|
| `TWITTER_BEARER_TOKEN`, `TWITTER_ENABLED` | X/Twitter credentials and toggle |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_USER_AGENT`, `REDDIT_ENABLED` | Reddit |
| `YOUTUBE_API_KEY`, `YOUTUBE_ENABLED`, `YOUTUBE_CHANNEL_IDS` | YouTube (defaults to the curated list in `config.py`) |
| `GITHUB_TOKEN`, `GITHUB_ENABLED`, `GITHUB_WATCH_REPOS` | GitHub |

### Podcast / TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(empty)_ | If set → OpenAI TTS; if empty → gTTS fallback |
| `PODCAST_VOICE` | `alloy` | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |

### Newsletter

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWSLETTER_ENABLED` | `false` | Turn on the daily cron |
| `NEWSLETTER_EMAIL_TO` | _(empty)_ | Recipient address(es), comma-separated |
| `NEWSLETTER_EMAIL_FROM` | _(SMTP user)_ | Sender address |
| `NEWSLETTER_HOUR` / `NEWSLETTER_MINUTE` | `7` / `0` | Send time in `SCHEDULER_TIMEZONE` |
| `NEWSLETTER_HOURS_LOOKBACK` | `24` | How many hours of items to include per topic |
| `NEWSLETTER_INCLUDE_AUDIO` | `true` | Generate per-topic MP3 narrations |
| `NEWSLETTER_ATTACH_AUDIO` | `true` | Also attach the MP3 to the email. Safe default; set `false` once `PUBLIC_BASE_URL` works to keep emails small |
| `NEWSLETTER_AUDIO_DIR` | `data/newsletter_audio` | Directory where briefing MP3s are persisted so the web app can stream them |
| `PUBLIC_BASE_URL` | _(empty)_ | Internet-reachable origin of the running app (e.g. `https://news.example.com`). Required for inline audio playback in Gmail / Outlook / Yahoo. No trailing slash. |
| `NEWSLETTER_TOPICS` | _(empty)_ | Optional override; empty = use UI-saved defaults |

### SMTP (for the newsletter)

| Variable | Default | Description |
|----------|---------|-------------|
| `SMTP_HOST` | `smtp.gmail.com` | SMTP server |
| `SMTP_PORT` | `587` | `587` for STARTTLS, `465` for implicit SSL |
| `SMTP_USER` | _(empty)_ | SMTP username |
| `SMTP_PASSWORD` | _(empty)_ | SMTP password (Gmail: **16-char App Password**, not your login password) |
| `SMTP_USE_TLS` | `true` | `true` = STARTTLS on 587, `false` = implicit SSL on 465 |

---

## Running Tests

```bash
pytest                      # full suite (335 tests)
pytest --cov=news_agent     # with coverage
pytest tests/pipeline/      # one directory
pytest tests/storage/test_repository.py::test_upsert_deduplicates_by_id -x
```

`asyncio_mode = "auto"` is set in `pyproject.toml` — async tests don't need a decorator. The autouse `isolated_db` fixture in `tests/conftest.py` creates a fresh in-memory SQLite per test and patches every module-level `get_session` / `init_db` reference so tests never pollute `data/news.db`.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
