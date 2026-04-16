# News Retrieval Agent

An AI-powered news aggregator that collects articles, tweets, videos, and posts from multiple sources, deduplicates and ranks them, then generates AI-powered summaries — all served through a real-time web UI. Supports any LLM: Anthropic Claude, OpenAI GPT, Groq, Google Gemini, Mistral, and more.

## Features

- **Multi-source collection** — RSS feeds, X/Twitter, Reddit, YouTube, GitHub, LinkedIn
- **AI analysis** — scores each item for relevance, sentiment, key entities, and tags using any LLM
- **Real-time digest streaming** — summaries stream token-by-token like a chat response
- **Keyword search** — search any topic on demand; results appear as they are fetched
- **Language filtering** — filter news by language; non-English YouTube videos are excluded
- **Spam filtering** — engagement floors, content-based spam detection, and `-is:nullcast has:links` on Twitter queries
- **Podcast generation** — convert any topic digest into an audio podcast via OpenAI TTS
- **Scheduled background fetch** — automatic fetch every N hours (configurable)
- **Export** — dump items to JSON or Markdown

---

## Project Structure

```
news_agent/
├── collectors/          # One collector per source
│   ├── base.py          # BaseCollector: rate limiting, language tagging, score normalization
│   ├── rss.py           # RSS/Atom feeds (ESPN NBA, tech blogs, finance sites, …)
│   ├── twitter.py       # X/Twitter API v2 via tweepy
│   ├── reddit.py        # Reddit via praw
│   ├── youtube.py       # YouTube Data API v3
│   ├── github.py        # GitHub trending / search via PyGithub
│   └── linkedin.py      # LinkedIn RSS feeds
│
├── pipeline/
│   ├── aggregator.py    # Fan-out: runs all collectors in parallel
│   ├── analyzer.py      # Claude batch analysis + digest generation (streaming)
│   └── deduplicator.py  # Semantic dedup via sentence-transformers + cosine similarity
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

- **Search bar** — type any keyword (e.g. `NVDA`, `basketball`, `climate`) to fetch and display items on demand
- **Two-panel layout** — compare two topics side by side via URL params: `?topic1=ai&topic2=stocks`
- **Streaming summary** — an AI digest streams in above the news cards as soon as enough items are collected
- **Language filter** — click the globe button to select which languages to display
- **Podcast** — click the microphone button to generate an audio digest for a topic
- **Thumbs up / down** — vote on items to teach the ranking what you want more or less of

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
| `LLM_MODEL` | `anthropic/claude-sonnet-4-6` | LLM used for analysis and digests (litellm format) |
| `LLM_API_KEY` | _(empty)_ | API key for the LLM provider (or set provider env var directly) |
| `BATCH_SIZE` | `15` | Items per LLM analysis batch |
| `MAX_ITEMS_PER_SOURCE` | `50` | Items fetched per source per cycle |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold for deduplication |
| `SCHEDULE_INTERVAL_HOURS` | `4` | How often the background scheduler runs |
| `RETENTION_DAYS` | `30` | Days before items are pruned from the database |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/news.db` | SQLAlchemy DB URL |

---

## Running Tests

```bash
pytest
pytest --cov=news_agent    # with coverage
```
