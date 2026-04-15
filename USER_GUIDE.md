# News Retrieval Agent — User Guide

## Web UI

Start the server, then open **http://localhost:8000** in your browser.

```bash
python -m news_agent.cli serve
```

The scheduler starts automatically with the server — no second process needed.

### Main page (`/`)

| Element | How to use |
|---|---|
| **Topic panels** | Type any keyword in the panel header input and press Enter to fetch news for that topic |
| **⚙ Settings** | Open the gear icon to save default topics — restored on every visit. Clear both fields and save to reset to empty start |
| **Hours selector** | Filter items by how recently they were published (10 min → 72 h) |
| **Star button ★** | Star items to bookmark them; starred items are preserved across fetches |
| **Card expand** | Click a card to expand and see full content and source link |
| **🎙 Podcast** | Generate a spoken audio briefing from the current panel's news |

### Search page (`/search`)

Type any keyword in the search bar. The system immediately fetches from Google News, Bing News, Reddit, YouTube, and Twitter for that keyword and shows results as they arrive.

---

## CLI Commands

```
python -m news_agent.cli [--debug] COMMAND
```

### `serve` — Start the web UI

```bash
python -m news_agent.cli serve
python -m news_agent.cli serve --port 9000
python -m news_agent.cli serve --reload        # auto-reload on code changes
```

The scheduler runs inside the server process. No need to run `schedule` separately.

---

### `fetch` — Run a one-off fetch

Fetches from all enabled sources and stores results in the database.

```bash
python -m news_agent.cli fetch                         # fetch everything (no topic filter)
python -m news_agent.cli fetch --topics "basketball"
python -m news_agent.cli fetch --topics "ADBE,NVDA"
python -m news_agent.cli fetch --topics "climate,politics,sports"
```

---

### `digest` — Generate a Claude summary

Generates a bullet-point digest for a topic from items already in the database.

```bash
python -m news_agent.cli digest --topic "basketball"
python -m news_agent.cli digest --topic "ai" --hours 48
```

Requires `ANTHROPIC_API_KEY` in `.env`.

---

### `export` — Export to file

```bash
python -m news_agent.cli export                        # markdown, last 24 h
python -m news_agent.cli export --format json
python -m news_agent.cli export --hours 72
python -m news_agent.cli export --date 2026-04-10
```

Output is written to the `data/` directory.

---

### `status` — Database and source health

```bash
python -m news_agent.cli status
```

Shows total item count broken down by topic, plus last-run time and error state for each collector.

---

### `sources` — Manage collectors

```bash
python -m news_agent.cli sources list              # show all sources and enabled/disabled state
python -m news_agent.cli sources enable reddit
python -m news_agent.cli sources disable youtube
```

Available sources: `reddit`, `github`, `youtube`, `x`, `rss`

---

### `schedule` — Run the scheduler standalone

Only needed if you want the scheduler running separately from the web server.

```bash
python -m news_agent.cli schedule
```

Schedule (configured in `.env`):
- **Every 4 hours** — fetch from all sources (set `SCHEDULE_INTERVAL_HOURS` to change)
- **Daily at 08:00** — generate digests for all topics in the DB
- **Daily at 03:00** — prune items older than 30 days

---

## Configuration (`.env`)

| Key | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for digests and Claude analysis |
| `OPENAI_API_KEY` | Optional — enables higher-quality podcast TTS |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit API credentials |
| `TWITTER_BEARER_TOKEN` | Twitter/X API bearer token |
| `YOUTUBE_API_KEY` | YouTube Data API key |
| `GITHUB_TOKEN` | Optional — higher GitHub API rate limits |
| `SCHEDULE_INTERVAL_HOURS` | How often the background fetch runs (default: `4`) |
| `RETENTION_DAYS` | How long items stay in the DB (default: `30`) |
| `PODCAST_VOICE` | OpenAI TTS voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
