from __future__ import annotations

import json

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


def _csv_decode_complex(self, field_name, field_info, value):
    """Decode complex env values: try JSON first, fall back to raw string.

    pydantic_settings v2 normally raises SettingsError when a list[str] field
    contains a comma-separated string (not valid JSON).  By returning the raw
    string on failure we let the field_validator(mode="before") handle CSV
    parsing instead of blowing up at source-load time.
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value  # field_validator will split on ","


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # ── LLM (any provider via litellm) ───────────────────────────────────────
    # Model format: "anthropic/claude-sonnet-4-6", "openai/gpt-4o",
    #               "groq/llama-3.3-70b-versatile", "gemini/gemini-2.0-flash", etc.
    llm_model: str = "anthropic/claude-sonnet-4-6"
    llm_api_key: str = ""   # if set, overrides provider env vars (ANTHROPIC_API_KEY etc.)

    # Faster/cheaper model used for bulk item scoring (relevance, tags, sentiment).
    # Defaults to Haiku — 5-10× faster than Sonnet with negligible quality loss for
    # structured classification. Set to llm_model to use the same model for everything.
    analysis_model: str = "anthropic/claude-haiku-4-5-20251001"

    # API key for the analysis model. Required when analysis_model uses a different
    # provider than the main LLM (e.g. Gemini for analysis, Anthropic for digest/Q&A).
    # Leave empty to auto-select based on analysis_model provider prefix.
    analysis_api_key: str = ""

    # Multi-model rotation: comma-separated list of models to distribute batches across.
    # Each model uses the corresponding entry in ANALYSIS_API_KEYS.
    # Leave empty to use only analysis_model above.
    # Example: ANALYSIS_MODELS=gemini/gemini-2.5-flash,gemini/gemini-2.5-flash
    analysis_models: list[str] = []
    analysis_api_keys: list[str] = []   # parallel to analysis_models
    # RPM limit for each model in ANALYSIS_MODELS (parallel list).
    # Controls weighted batch assignment and per-model rate limiting.
    # Leave empty to default all models to 10 RPM.
    # Example: ANALYSIS_RPMS=10,50,500
    analysis_rpms: list[int] = []

    # Max concurrent LLM calls across all models combined (safety cap).
    analysis_concurrency: int = 5

    # ── Anthropic (kept for backward compatibility) ───────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = ""  # deprecated — use llm_model

    # ── Google / Gemini ───────────────────────────────────────────────────────
    gemini_api_key: str = ""

    # ── OpenAI TTS ────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    podcast_voice: str = "alloy"  # alloy | echo | fable | onyx | nova | shimmer

    # ── Twitter / X ───────────────────────────────────────────────────────────
    twitter_bearer_token: str | None = None
    twitter_enabled: bool = True

    @field_validator("twitter_bearer_token", mode="before")
    @classmethod
    def decode_twitter_token(cls, v: str | None) -> str | None:
        """URL-decode the bearer token in case it was pasted with %3D instead of =."""
        if v:
            from urllib.parse import unquote
            return unquote(v)
        return v

    # ── Reddit ────────────────────────────────────────────────────────────────
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_username: str | None = None
    reddit_user_agent: str = "NewsAgent/1.0 by u/news_agent_bot"
    reddit_enabled: bool = True

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: str | None = None
    github_enabled: bool = True
    github_watch_repos: list[str] = [
        "anthropics/anthropic-sdk-python",
        "openai/openai-python",
        "huggingface/transformers",
        "langchain-ai/langchain",
        "microsoft/autogen",
        "ollama/ollama",
        "ggerganov/llama.cpp",
        "mistralai/mistral-src",
    ]

    # ── YouTube ───────────────────────────────────────────────────────────────
    youtube_api_key: str | None = None
    youtube_enabled: bool = True
    youtube_channel_ids: list[str] = [
        # ── Frontier Labs ──────────────────────────────────────────────────────
        "UCXZCJLdBC09xxGZ6gcdrc6A",  # OpenAI
        "UCrDwWp7EBBv4NwvScIpBDOA",  # Anthropic
        "UCnUYZLuoy1rq1aVMwx4aTzw",  # Google DeepMind
        "UCCb9_Kn8F_Opb3UCGm-lILQ",  # Microsoft Research
        "UCbmNph6atAoGfqLoCL_duAg",  # Two Minute Papers
        # ── Top AI Researchers ─────────────────────────────────────────────────
        "UCPk8m_r6fkUSYmvgCBwq-sw",  # Andrej Karpathy (active channel)
        "UCWX3yGbODI3HLa-7k-4FNHA",  # Yannic Kilcher
        "UCtYLUTtgS3k1Fg4y5tAhLbw",  # Lex Fridman
        "UCMU7l2bIv6MXlgJR3-E33Dw",  # Yann LeCun
        "UCX7Y2qWriXpqocG97SFW2OQ",  # Jeremy Howard (fast.ai)
        "UCMLtBahI5DMrt0NPvDSoIRQ",  # ML Street Talk
        "UCNJ1Ymd5yFuUPtn21xtRbbw",  # AI Explained
        "UCYO_jab_esuFRV4b17AJtAw",  # 3Blue1Brown (math/ML foundations)
        # ── AI Education & Commentary ──────────────────────────────────────────
        "UCsBjURrPoezykLs9EqgamOA",  # Fireship (dev/AI news)
        "UChugFTK0KyrES9terTid8vA",  # Stanford HAI
        # ── Interpretability & Safety ──────────────────────────────────────────
        "UCBMJ0D-omcRay8dh4QT0doQ",  # Neel Nanda (mechanistic interpretability)
        "UCY_K5gXsXHtuiP8mj3BiWxA",  # Center for AI Safety (Dan Hendrycks)
        # ── Pioneer Researchers ────────────────────────────────────────────────
        "UCCbqqckmMxVKaEs5jDBxukA",  # Yoshua Bengio
        # ── Long-form Researcher Interviews ───────────────────────────────────
        "UCXl4i9dYBrFOabk0xGmbkRA",  # Dwarkesh Patel Podcast
        # ── Industry Research Labs ─────────────────────────────────────────────
        "UC5qxlwEKM7-5YZudb24l0bg",  # AI at Meta
        "UCcr5vuAH5TPlYox-QLj4ySw",  # The Alan Turing Institute
        # ── Finance / Markets ──────────────────────────────────────────────────
        "UCrM7B7SL_g1edFOnmj-SDKg",  # Bloomberg Markets
        "UCrp_UI8XtuYfpiqluWLD7Lw",  # CNBC Television
    ]

    # ── Pipeline ──────────────────────────────────────────────────────────────
    dedup_strategy: str = "semantic"  # "semantic" | "tfidf" | "url_only"
    dedup_similarity_threshold: float = 0.82
    batch_size: int = 15
    max_items_per_source: int = 50

    # ── Search ────────────────────────────────────────────────────────────────
    # Blend factor for hybrid BM25 + semantic RRF fusion. 1.0 = pure BM25
    # (keyword-dominant; good for tickers, proper nouns). 0.0 = pure semantic
    # (good for paraphrases, concepts). 0.5 reproduces the legacy unweighted
    # RRF behaviour. Per-request override via /api/panel?alpha=... .
    default_hybrid_alpha: float = 0.5
    # When T2-A (smart filter) detects a ticker-like query, this alpha is used
    # instead of default_hybrid_alpha. Unused until T2-A lands.
    ticker_alpha: float = 0.75
    smart_filter_enabled: bool = False
    smart_filter_timeout_seconds: float = 3.0

    # ── Personalization ───────────────────────────────────────────────────────
    # When True, a user downvote inserts a row in DismissedItemORM and the item
    # is hidden from get_recent() / search() across sessions and re-fetches.
    # Upvoting (or explicitly un-downvoting) a dismissed item clears its tombstone.
    # Set False to preserve legacy behaviour where downvotes only influenced
    # future ranking scores without hiding the downvoted item itself.
    dismiss_on_downvote: bool = True

    # ── Storage ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///data/news.db"
    retention_days: int = 30

    # ── Scheduler ─────────────────────────────────────────────────────────────
    schedule_interval_hours: int = 4
    scheduler_timezone: str = "America/Los_Angeles"  # IANA name; used for cron jobs

    # ── Newsletter (daily email) ──────────────────────────────────────────────
    newsletter_enabled: bool = False
    newsletter_email_to: str = ""        # recipient address (one or more, comma-sep)
    newsletter_email_from: str = ""      # sender address (defaults to SMTP user)
    newsletter_hour: int = 7             # local hour (scheduler_timezone) to send
    newsletter_minute: int = 0
    newsletter_hours_lookback: int = 24  # include items from the last N hours
    newsletter_include_audio: bool = True  # attach MP3 narration
    # Optional override for which topics to include. Empty = use topics saved
    # via the web UI's "Default Topics" settings panel (UserSettingORM).
    newsletter_topics: list[str] = []

    # Public base URL where the web app is reachable (e.g.
    # "https://news.example.com").  When set, the newsletter embeds an HTML5
    # <audio> player that streams the MP3 from
    # {public_base_url}/newsletter/audio/<filename>, which works in Gmail,
    # Outlook and other clients that strip cid: and <audio> when served from
    # an attachment.  Leave empty to fall back to MP3 attachments only.
    public_base_url: str = ""
    # Directory where newsletter MP3s are persisted so the web app can
    # stream them.  Relative paths are resolved against the project root.
    newsletter_audio_dir: str = "data/newsletter_audio"
    # Keep attaching the MP3 to the email in addition to embedding the
    # player URL.  Set False once you've confirmed the URL works to keep
    # emails small.
    newsletter_attach_audio: bool = True

    # ── SMTP (for newsletter) ─────────────────────────────────────────────────
    # Gmail: host=smtp.gmail.com, port=587, user=your@gmail.com,
    #   password=APP-PASSWORD (16-char app password, not your login password).
    #   Create one at https://myaccount.google.com/apppasswords
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True  # STARTTLS on port 587; set False + port 465 for SSL

    @field_validator("youtube_channel_ids", "github_watch_repos",
                     "analysis_models", "analysis_api_keys", "analysis_rpms",
                     "newsletter_topics", mode="before")
    @classmethod
    def split_comma_list(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @classmethod
    def settings_customise_sources(cls, settings_cls: type[BaseSettings], **kwargs) -> tuple[PydanticBaseSettingsSource, ...]:
        # Patch env/dotenv sources so comma-separated list values don't crash
        # with JSONDecodeError before field_validators run.
        import types
        sources = tuple(kwargs.values())
        for source in sources:
            if source is not None and hasattr(source, "decode_complex_value"):
                source.decode_complex_value = types.MethodType(_csv_decode_complex, source)
        return sources


settings = Settings()
