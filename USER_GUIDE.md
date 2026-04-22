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
| **Send test now** (in ⚙ Settings) | Email the newsletter for your default topics right now — useful to verify SMTP setup |

### Search page (`/search`)

Type any keyword in the search bar. Results are retrieved from the database using hybrid BM25 + semantic search (RRF merge). For keywords not already in the database, the system fetches live from Google News, Bing News, Reddit, YouTube, and GitHub for that keyword and streams results as they arrive.

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

Requires `LLM_API_KEY` (or `ANTHROPIC_API_KEY`) in `.env`.

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

Available sources: `reddit`, `github`, `youtube`, `twitter`, `rss`

---

### `newsletter` — Email your daily digest right now

Runs a full fetch for every configured topic, waits for LLM analysis to
populate summaries / tags / sentiment, and then emails the newsletter
(HTML summary + article list + MP3 audio). Useful for testing SMTP setup
before relying on the daily cron.

```bash
python -m news_agent.cli newsletter                     # fetch + analyze + send (slow)
python -m news_agent.cli newsletter --no-refresh        # skip fetch; use cached DB (fast)
python -m news_agent.cli newsletter --to me@example.com
python -m news_agent.cli newsletter --topics "ai,stocks" --hours 48
python -m news_agent.cli newsletter --no-audio          # skip MP3 attachment
```

Topics are resolved in this order:
1. `--topics` flag
2. `NEWSLETTER_TOPICS` env var
3. Default topics saved via the web UI's ⚙ gear icon (stored server-side)

Requires `SMTP_*` configured in `.env` (see below).

> Using `--no-refresh` is the fastest way to iterate on SMTP/template
> issues because it skips the minute-scale fetch+LLM step.

---

### `schedule` — Run the scheduler standalone

Only needed if you want the scheduler running separately from the web server.

```bash
python -m news_agent.cli schedule
```

Schedule (configured in `.env`, cron times use `SCHEDULER_TIMEZONE`):
- **Every 4 hours** — fetch from all sources (set `SCHEDULE_INTERVAL_HOURS` to change)
- **Daily at 07:00** — email newsletter for default topics (if `NEWSLETTER_ENABLED=true`)
- **Daily at 08:00** — generate digests for all topics in the DB
- **Daily at 03:00** — prune items older than 30 days

---

## Daily Newsletter

At your configured time (default 07:00 PST) the scheduler:

1. **Fetches fresh news** for every default topic (same pipeline as the web
   UI's refresh button — every enabled source, deduped).
2. **Waits for LLM analysis** to finish so every item has a summary,
   relevance score, sentiment, and tags.
3. **Emails** you a message containing:
   - a Claude-generated narrative digest per topic,
   - the same article cards you see at `localhost:8000` (source badges,
     sentiment, tags, direct links),
   - **one MP3 narration per topic** attached to the email (e.g.
     `ai-briefing-2026-04-20.mp3`, `stocks-briefing-2026-04-20.mp3`). Each
     topic section in the email shows its audio's filename and a `cid:`
     link so mail clients that support inline attachments (Apple Mail,
     many webmail clients) surface a one-click "Listen" button.  OpenAI
     TTS is used when `OPENAI_API_KEY` is set, otherwise the free gTTS
     fallback.

Because the fetch + analysis happens inline, the daily job can take several
minutes to complete — expect the email to arrive slightly after your
`NEWSLETTER_HOUR` rather than exactly on the minute. If the pre-fetch
fails (e.g. a source is down), the newsletter still sends with whatever
is currently in the DB rather than skipping the day entirely.

### Setup

1. **Save your default topics** in the web UI (click ⚙ → fill in Panel 1 / Panel 2 → Save). These are synced to the server so the scheduler can read them.
2. **Configure SMTP** in `.env`:
   ```
   NEWSLETTER_ENABLED=true
   NEWSLETTER_EMAIL_TO=you@example.com
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your_gmail@gmail.com
   SMTP_PASSWORD=your_16_char_app_password
   ```
3. **Gmail users**: the `SMTP_PASSWORD` must be an **App Password**, not your regular login password. Create one at https://myaccount.google.com/apppasswords (requires 2-Step Verification).
4. **Test it** — either use the **Send test now** button in the web UI's ⚙ panel, or run `python -m news_agent.cli newsletter`.
5. **Restart the server** so the scheduler picks up the new cron time:
   ```bash
   python -m news_agent.cli serve
   ```

### Override topics via env var

If you don't want to use the UI-saved defaults, set `NEWSLETTER_TOPICS=ai,stocks` in `.env` to hardcode which topics go into the newsletter.

---

## Configuration (`.env`)

| Key | Description |
|---|---|
| `LLM_MODEL` | LLM to use (default: `anthropic/claude-sonnet-4-6`) |
| `LLM_API_KEY` | Required — API key for the LLM provider (Anthropic, OpenAI, Gemini, etc.) |
| `ANALYSIS_MODEL` | Cheaper model for bulk item scoring (default: `anthropic/claude-haiku-4-5-20251001`) |
| `OPENAI_API_KEY` | Optional — enables higher-quality podcast TTS via OpenAI |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit API credentials |
| `TWITTER_BEARER_TOKEN` | Twitter/X API bearer token |
| `YOUTUBE_API_KEY` | YouTube Data API key |
| `GITHUB_TOKEN` | Optional — higher GitHub API rate limits |
| `SCHEDULE_INTERVAL_HOURS` | How often the background fetch runs (default: `4`) |
| `RETENTION_DAYS` | How long items stay in the DB (default: `30`) |
| `PODCAST_VOICE` | OpenAI TTS voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `SCHEDULER_TIMEZONE` | IANA tz for cron jobs, e.g. `America/Los_Angeles` (default) |
| `NEWSLETTER_ENABLED` | Set `true` to enable the daily email (default: `false`) |
| `NEWSLETTER_EMAIL_TO` | Recipient address (comma-separate for multiple) |
| `NEWSLETTER_EMAIL_FROM` | Sender address (defaults to `SMTP_USER`) |
| `NEWSLETTER_HOUR` / `NEWSLETTER_MINUTE` | Send time in `SCHEDULER_TIMEZONE` (default: `07:00`) |
| `NEWSLETTER_HOURS_LOOKBACK` | How many hours of items to include (default: `24`) |
| `NEWSLETTER_INCLUDE_AUDIO` | Set `false` to skip the MP3 attachment |
| `NEWSLETTER_TOPICS` | Optional override; comma-separated. Empty = use UI-saved defaults |
| `SMTP_HOST` / `SMTP_PORT` | SMTP server (Gmail: `smtp.gmail.com` / `587`) |
| `SMTP_USER` / `SMTP_PASSWORD` | SMTP credentials. Gmail requires an **App Password** |
| `SMTP_USE_TLS` | `true` for STARTTLS on 587, `false` for implicit SSL on 465 |
