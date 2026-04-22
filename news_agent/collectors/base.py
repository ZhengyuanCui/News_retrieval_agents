from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel

from news_agent.models import NewsItem

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    source_name: str = "base"
    default_topics: list[str] = []
    StateSchema: type[BaseModel] | None = None
    rate_limit_delay: float = 1.0  # seconds between requests

    def __init__(self, topics: list[str] | None = None) -> None:
        # None → use collector defaults; [] → no topic filter (fetch everything)
        self.topics = topics if topics is not None else self.default_topics
        self._last_call: datetime | None = None

    @abstractmethod
    async def fetch(self) -> list[NewsItem]:
        """Fetch news items from this source. Must be implemented by subclasses."""
        ...

    async def fetch_keyword(self, keyword: str) -> list[NewsItem]:
        """Fetch items matching an arbitrary keyword. Override in subclasses that support it."""
        return []

    async def load_state(self) -> BaseModel | dict | None:
        """Load validated collector state from persistent storage."""
        if self.StateSchema is None:
            return None
        from news_agent.storage import get_session
        from news_agent.storage.repository import NewsRepository

        async with get_session() as session:
            repo = NewsRepository(session)
            row = await repo.get_collector_state(self.source_name)
        payload = row.state if row and row.state else {}
        return self.StateSchema.model_validate(payload)

    async def save_state(self, state: BaseModel | dict) -> BaseModel | dict:
        """Validate and persist collector state."""
        if self.StateSchema is None:
            return state
        validated = state if isinstance(state, self.StateSchema) else self.StateSchema.model_validate(state)
        from news_agent.storage import get_session
        from news_agent.storage.repository import NewsRepository

        async with get_session() as session:
            repo = NewsRepository(session)
            await repo.update_collector_state(
                source=self.source_name,
                state=validated.model_dump(mode="json"),
            )
        return validated

    def is_enabled(self) -> bool:
        """Check if this collector is enabled via config."""
        return True

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        if self._last_call is not None:
            elapsed = (datetime.utcnow() - self._last_call).total_seconds()
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_call = datetime.utcnow()

    @staticmethod
    def normalize_score(value: float, min_val: float, max_val: float) -> float:
        """Normalize a raw engagement score to 0-1."""
        if max_val <= min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def tag_language(item: NewsItem) -> NewsItem:
        """Detect and set item.language from its title+content. Returns item."""
        from news_agent.lang import detect_language
        item.language = detect_language(item.title + " " + (item.content or "")[:200])
        return item

    @classmethod
    def tag_languages(cls, items: list[NewsItem]) -> list[NewsItem]:
        """Detect language for all items in-place."""
        return [cls.tag_language(i) for i in items]
