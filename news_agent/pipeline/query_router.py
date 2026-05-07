from __future__ import annotations

import asyncio
import json
import re

import litellm
from pydantic import BaseModel, Field

from news_agent.config import settings
from news_agent.pipeline.analyzer import _analysis_api_key

_TICKER_RE = re.compile(r"(?:\$[A-Z]{1,6}\b|\b[A-Z]{2,5}\b)")
_HOURS_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s*(hour|hours|hr|hrs|day|days)\b", re.IGNORECASE)

ROUTER_PROMPT = """\
Extract structured retrieval filters from this news query.

Query: "{query}"

Return ONLY valid JSON with:
- "sources": array of source names like ["openai", "anthropic", "github", "youtube"]
- "topics": array of topic keywords
- "entities": array of named entities or companies
- "hours": integer or null
- "is_ticker": boolean

Prefer empty arrays when uncertain. Do not invent unsupported filters.
"""


class QueryFilters(BaseModel):
    sources: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    hours: int | None = None
    is_ticker: bool = False

    def has_constraints(self) -> bool:
        return bool(self.sources or self.topics or self.entities or self.hours is not None)


def _fallback_filters(query: str) -> QueryFilters:
    hours = None
    match = _HOURS_RE.search(query)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        hours = amount * 24 if unit.startswith("day") else amount
    return QueryFilters(is_ticker=bool(_TICKER_RE.search(query)), hours=hours)


async def extract_filters(query: str) -> QueryFilters:
    fallback = _fallback_filters(query)
    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=settings.analysis_model,
                messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"},
                **({"api_key": _analysis_api_key()} if _analysis_api_key() else {}),
            ),
            timeout=settings.smart_filter_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return fallback
    except Exception:
        return fallback

    try:
        payload = json.loads(response.choices[0].message.content or "{}")
        parsed = QueryFilters.model_validate(payload)
    except Exception:
        return fallback

    if fallback.hours is not None and parsed.hours is None:
        parsed.hours = fallback.hours
    parsed.is_ticker = parsed.is_ticker or fallback.is_ticker
    return parsed
