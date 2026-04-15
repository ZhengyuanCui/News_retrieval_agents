from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal  # still used for sentiment field

from pydantic import BaseModel, Field, computed_field
from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ── Pydantic schema (the contract between all layers) ────────────────────────

TopicLiteral = str  # any keyword — not restricted to a fixed set
SourceLiteral = str  # open-ended: twitter, reddit, github, youtube, bloomberg, techcrunch, etc.


class NewsItem(BaseModel):
    source: SourceLiteral
    topic: TopicLiteral
    title: str
    url: str
    content: str
    author: str | None = None
    published_at: datetime
    raw_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized engagement score")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "negative", "neutral"] | None = None
    relevance_score: float | None = Field(default=None, ge=0.0, le=10.0)
    key_entities: list[str] = Field(default_factory=list)
    is_duplicate: bool = False
    duplicate_of: str | None = None
    is_starred: bool = False
    language: str = "en"  # ISO 639-1 code, e.g. "en", "zh", "es"

    @computed_field
    @property
    def id(self) -> str:
        return hashlib.sha256(f"{self.source}:{self.url}".encode()).hexdigest()[:16]


class Digest(BaseModel):
    date: str  # YYYY-MM-DD
    topic: TopicLiteral
    content: str
    item_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ── SQLAlchemy ORM models ─────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class NewsItemORM(Base):
    __tablename__ = "news_items"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    topic: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    raw_score: Mapped[float] = mapped_column(Float, default=0.0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    key_entities: Mapped[list] = mapped_column(JSON, default=list)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    duplicate_of: Mapped[str | None] = mapped_column(String(16), nullable=True)
    is_starred: Mapped[bool] = mapped_column(Boolean, default=False)
    language: Mapped[str] = mapped_column(String(8), default="en", index=True)

    def to_pydantic(self) -> NewsItem:
        return NewsItem(
            source=self.source,
            topic=self.topic,
            title=self.title,
            url=self.url,
            content=self.content,
            author=self.author,
            published_at=self.published_at,
            raw_score=self.raw_score,
            fetched_at=self.fetched_at,
            summary=self.summary,
            tags=self.tags or [],
            sentiment=self.sentiment,
            relevance_score=self.relevance_score,
            key_entities=self.key_entities or [],
            is_duplicate=self.is_duplicate,
            duplicate_of=self.duplicate_of,
            language=self.language or "en",
        )


class DigestORM(Base):
    __tablename__ = "digests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # "{date}_{topic}"
    date: Mapped[str] = mapped_column(String(10), index=True)
    topic: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    item_count: Mapped[int] = mapped_column(default=0)
    generated_at: Mapped[datetime] = mapped_column(DateTime)


class UserInteractionORM(Base):
    """Tracks every user action on a news item."""
    __tablename__ = "user_interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    item_id: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(16))  # "click" | "star" | "unstar" | "read"
    read_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)  # for "read" actions
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class UserPreferenceORM(Base):
    """Aggregated preference weights per dimension (tag, source, topic keyword)."""
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dimension: Mapped[str] = mapped_column(String(16))   # "tag" | "source" | "entity"
    value: Mapped[str] = mapped_column(String(128), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CollectorStateORM(Base):
    __tablename__ = "collector_state"

    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    last_run: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    items_fetched: Mapped[int] = mapped_column(default=0)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    youtube_quota_used: Mapped[int] = mapped_column(default=0)
    youtube_quota_reset_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
