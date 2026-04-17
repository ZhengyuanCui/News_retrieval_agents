# News Retrieval Agent

An AI-powered news aggregator that collects articles, tweets, videos, and posts from multiple sources, deduplicates and ranks them, then generates AI-powered summaries — all served through a real-time web UI. Supports any LLM: Anthropic Claude, OpenAI GPT, Groq, Google Gemini, Mistral, and more.

## Features

- **Multi-source collection** — RSS/Google News/Bing News, X/Twitter, Reddit, YouTube, GitHub, LinkedIn
- **AI analysis** — scores each item for relevance (0–10), sentiment, key entities, and tags using any LLM
- **Semantic search** — natural-language queries (e.g. "will the market rise due to the Iran war?") are answered via vector similarity search over all stored articles
- **Semantic re-ranking** — search results are blended: 50 % embedding similarity to query + 40 % LLM relevance score + 10 % raw engagement score
- **Real-time digest streaming** — summaries stream token-by-token like a chat response
- **Keyword search** — fetch any topic on demand; results appear as they are collected
- **Language filtering** — filter news by language; non-English YouTube videos are excluded
- **Spam filtering** — engagement floors, ML spam classifier (`mshenoda/roberta-spam`, 125 M RoBERTa), keyword hard-blocks, and influencer-shill pattern detection
- **Deduplication** — semantic cosine-similarity dedup (threshold 0.82) plus title-normalisation to catch cross-publisher duplicates
- **Podcast generation** — convert any topic digest into an audio podcast via OpenAI TTS
- **Scheduled background fetch** — automatic fetch every N hours (configurable)
- **Export** — dump items to JSON or Markdown

---

## Project Structure

```
news_agent/
├── collectors/          # One collector per source
│   ├── base.py          # BaseCollector: rate limiting, language tagging, score normalization
│   ├── rss.py           # RSS/Atom + Google News + Bing News search feeds
│   ├── twitter.py       # X/Twitter API v2 via tweepy
│   ├── reddit.py        # Reddit via praw
│   ├── youtube.py       # YouTube Data API v3
│   ├── github.py        # GitHub trending / search via PyGithub
│   └── linkedin.py      # LinkedIn RSS feeds
│
├── pipeline/
│   ├── aggregator.py    # Fan-out: runs all collectors in parallel
│   ├── analyzer.py      # LLM batch analysis + digest generation (streaming)
│   ├── deduplicator.py  # Semantic dedup via sentence-transformers + cosine similarity
│   ├── embeddings.py    # Shared all-MiniLM-L6-v2 singleton (thread-safe)
│   ├── ranker.py        # Blended semantic + LLM + engagement re-ranking
│   ├── vector_search.py # In-memory vector index for semantic fallback search
│   └── spam.py          # ML spam classifier (mshenoda/roberta-spam)
│
├── storage/
│   ├── database.py      # SQLAlchemy async engine + session factory (SQLite WAL)
│   ├── repository.py    # CRUD: news items, digests, collector state
│   └── exporter.py      # JSON / Markdown export
│
├── web/
│   ├── app.py           # FastAPI app — REST + SSE endpoints + Jinja2 pages
│   ├── static/
│   │   ├── app.js       # Panel auto-fetch, digest SSE client, podcast polling
│   │   └── style.css
│   └── templates/
│       ├── digest.html           # Main two-panel page
│       └── partials/             # Jinja2 fragments for AJAX updates
│
├── models.py            # Pydantic NewsItem + SQLAlchemy ORM models
├── config.py            # Pydantic-settings config (reads .env)
├── spam.py              # ML spam classifier wrapper
├── lang.py              # Language detection via langdetect
├── orchestrator.py      # Top-level: fetch cycle, keyword fetch, digest generation
├── scheduler.py         # APScheduler background loop
└── cli.py               # Click CLI entry point

alembic/                 # Database migrations
tests/                   # pytest test suite
data/                    # SQLite database (gitignored)
```

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

> If PowerShell blocks the activation script with *"running scripts is disabled on this system"*, run this once then try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in the API keys you want to use. One LLM key is required for analysis and digests; all source keys are optional — sources without keys are automatically disabled.

#### LLM provider (pick one)

The app uses [litellm](https://docs.litellm.ai) under the hood, so you can plug in any supported provider by setting two variables:

```env
LLM_MODEL=anthropic/claude-sonnet-4-6   # model string (see examples below)
LLM_API_KEY=<your-key>                  # API key for that provider
```

| Provider | `LLM_MODEL` example | Where to get a key |
|----------|--------------------|--------------------|
| Anthropic (default) | `anthropic/claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | `openai/gpt-4o` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Groq (fast, free tier) | `groq/llama-3.3-70b-versatile` | [console.groq.com](https://console.groq.com) |
| Google Gemini | `gemini/gemini-2.0-flash` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| Mistral | `mistral/mistral-large-latest` | [console.mistral.ai](https://console.mistral.ai) |

If you already have `ANTHROPIC_API_KEY` set in your environment and don't set `LLM_MODEL`/`LLM_API_KEY`, it will continue to work without any changes.

#### Other keys

| Key | Required | Where to get it |
|-----|----------|----------------|
| `TWITTER_BEARER_TOKEN` | No | [developer.twitter.com](https://developer.twitter.com) |
| `REDDIT_CLIENT_ID` / `SECRET` | No | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| `YOUTUBE_API_KEY` | No | [console.cloud.google.com](https://console.cloud.google.com) |
| `GITHUB_TOKEN` | No | [github.com/settings/tokens](https://github.com/settings/tokens) |
| `OPENAI_API_KEY` | No (for podcasts only) | [platform.openai.com](https://platform.openai.com) |

### 3. Run

```bash
# Start the web UI (http://localhost:8000)
news-agent serve

# Development mode with auto-reload
news-agent serve --reload

# One-off fetch from all enabled sources
news-agent fetch

# Fetch specific topics only
news-agent fetch --topics "ai,stocks"

# Print a digest to the terminal
news-agent digest --topic ai --hours 24

# Export items to Markdown
news-agent export --format markdown

# Start the background scheduler (fetches every 4h by default)
news-agent schedule
```

---

## Web UI

Open `http://localhost:8000` in your browser.

- **Search bar** — type any keyword *or* natural-language question (e.g. `will markets rise if Iran war ends?`) to fetch and display items on demand
- **Two-panel layout** — compare two topics side by side via URL params: `?topic1=ai&topic2=stocks`
- **Streaming summary** — an AI digest streams in above the news cards as soon as enough items are collected
- **Language filter** — click the globe button to select which languages to display
- **Podcast** — click the microphone button to generate an audio digest for a topic
- **Thumbs up / down** — vote on items to teach the ranking what you want more or less of

---

## Search & Q&A

### Keyword search

Type a short keyword (`AI`, `NVDA`, `Fed rates`) to fetch and display a curated list of articles plus an AI-generated digest at the top.

### Question answering

Type a natural-language question (`Will the stock market rise if the Iran war ends?`, `Which AI company has the best model right now?`) and the system:

1. Detects it is a question (starts with a question word, ends with `?`, or is 7+ words long).
2. Retrieves the most semantically relevant articles from the last 7 days using vector similarity search.
3. Streams a grounded LLM answer — a direct one-sentence response followed by bullet points citing specific articles by source.

The answer is honest about gaps: if the retrieved articles don't contain enough information, it says so rather than hallucinating.

### How retrieval works

1. **Topic-exact match** (fast DB query) — if items have been fetched specifically for this keyword, they are returned immediately within the selected time window.
2. **Semantic vector search** (fallback) — if no exact match, the query is encoded with `all-MiniLM-L6-v2` and compared against an in-memory index of all articles from the last 7 days via cosine similarity. This handles natural-language questions and paraphrased queries with no keyword overlap.

The vector index holds embeddings for all non-duplicate articles from the last 7 days (~4 k items, 5–6 MB). It is built lazily on first use (~2–3 s once the model is warm) and refreshed every 10 minutes or after each pipeline ingest.

### How results are ranked

Retrieved articles are re-ranked using a blended score:

```
score = 0.5 × semantic_similarity   (embedding cosine sim to query)
      + 0.4 × llm_relevance          (LLM 0–10 score normalised to 0–1)
      + 0.1 × raw_engagement         (likes/retweets/views, normalised)
```

Items LLM-scored below 4.0/10 are excluded. Un-scored items (NULL) are always included so fresh articles appear before analysis completes.

---

## Spam Filtering

Spam is caught in layers, applied in order of cost:

| Layer | Method | Applies to |
|-------|--------|-----------|
| Engagement floor | Like+retweet count < threshold | Twitter, YouTube |
| Hard keyword blocks | Exact phrase match (e.g. "guaranteed profit", "copy my trades") | Twitter |
| Shill-combo detection | @mention + $cashtag + ≥ 2 soft indicators | Twitter |
| Ticker stuffing | ≥ 3 distinct `$TICKER` cashtags, or 4+ slash-separated tickers | Twitter, YouTube |
| ML classifier | `mshenoda/roberta-spam` (125 M RoBERTa, threshold 0.80) | Twitter |
| YouTube-specific | Spam phrase list + `#/$` symbol count + ticker-list regex | YouTube |

The ML model is loaded once at startup and shared across all requests. Batch inference is used for Twitter (one forward pass per query batch).

---

## Deduplication

- **URL dedup** — items with identical URLs are dropped before storage.
- **Semantic dedup** — article embeddings are compared pairwise; items with cosine similarity ≥ 0.82 are marked as duplicates.
- **Title normalisation** — trailing ` - Publication Name` suffixes are stripped before hashing so "Article Title - Reuters" and "Article Title" are recognised as the same story.

---

## CLI Reference

```
news-agent fetch                      Run a single fetch cycle
news-agent fetch --topics "ai,stocks" Fetch specific topics only
news-agent refetch                    Clear DB and re-fetch with current filters
news-agent refetch --yes              Skip confirmation prompt
news-agent schedule                   Start the background scheduler
news-agent serve                      Start the web server
news-agent serve --reload             Dev mode with auto-reload
news-agent digest --topic ai          Generate and print a topic digest
news-agent export --format markdown   Export items to Markdown
news-agent status                     Show database stats and per-source status
news-agent sources list               List all sources and their status
news-agent sources enable <name>      Enable a source
news-agent sources disable <name>     Disable a source
```

---

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `anthropic/claude-sonnet-4-6` | LLM for digests, Q&A, and streaming (litellm format) |
| `LLM_API_KEY` | _(empty)_ | API key for the LLM provider (or set provider env var directly) |
| `ANALYSIS_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Faster model used for bulk item scoring (relevance, tags, sentiment). Set to `LLM_MODEL` to use one model for everything. |
| `ANALYSIS_CONCURRENCY` | `5` | Max concurrent LLM calls during batch analysis |
| `BATCH_SIZE` | `15` | Items per LLM analysis batch |
| `MAX_ITEMS_PER_SOURCE` | `50` | Items fetched per source per cycle |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.82` | Cosine similarity threshold for deduplication |
| `SCHEDULE_INTERVAL_HOURS` | `4` | How often the background scheduler runs |
| `RETENTION_DAYS` | `30` | Days before items are pruned from the database |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/news.db` | SQLAlchemy DB URL |

---

## Running Tests

```bash
pytest
pytest --cov=news_agent    # with coverage
```
