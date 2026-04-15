from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # ── Claude ────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

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
        "UCbmNph6atAoGfqLoCL_duAg",  # Two Minute Papers
        "UCWX3yGbODI3HLa-7k-4FNHA",  # Yannic Kilcher
        "UCZHmQk67mSJgfCCTn7xBfew",  # Andrej Karpathy
        "UCrM7B7SL_g1edFOnmj-SDKg",  # Bloomberg Markets
        "UCrp_UI8XtuYfpiqluWLD7Lw",  # CNBC Television
        "UCtYLUTtgS3k1Fg4y5tAhLbw",  # Lex Fridman
        "UCnUYZLuoy1rq1aVMwx4aTzw",  # Google DeepMind
    ]

    # ── LinkedIn ──────────────────────────────────────────────────────────────
    linkedin_enabled: bool = True
    linkedin_rss_feeds: list[str] = [
        "https://www.linkedin.com/newsletters/ai-weekly/",
    ]

    # ── Pipeline ──────────────────────────────────────────────────────────────
    dedup_strategy: str = "semantic"  # "semantic" | "tfidf" | "url_only"
    dedup_similarity_threshold: float = 0.85
    batch_size: int = 15
    max_items_per_source: int = 50

    # ── Storage ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///data/news.db"
    retention_days: int = 30

    # ── Scheduler ─────────────────────────────────────────────────────────────
    schedule_interval_hours: int = 4

    @field_validator("youtube_channel_ids", "github_watch_repos", "linkedin_rss_feeds", mode="before")
    @classmethod
    def split_comma_list(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


settings = Settings()
