from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from news_agent.models import NewsItem, UserInteractionORM, UserPreferenceORM

logger = logging.getLogger(__name__)

# Interaction weights — how much each action shifts the score
WEIGHTS = {
    "star": 5.0,
    "unstar": -5.0,
    "click": 1.5,
    "read_short": 0.5,   # 15–60s
    "read_medium": 1.5,  # 60–180s
    "read_long": 3.0,    # 180s+
}


async def record_interaction(
    session: AsyncSession,
    item_id: str,
    action: str,
    read_seconds: float | None = None,
) -> None:
    session.add(UserInteractionORM(
        item_id=item_id,
        action=action,
        read_seconds=read_seconds,
        created_at=datetime.utcnow(),
    ))


async def recompute_preferences(session: AsyncSession) -> None:
    """
    Rebuild preference scores from all recorded interactions.
    Called after each fetch cycle so scores reflect latest behaviour.
    """
    # Load all interactions with their item metadata
    interactions_q = await session.execute(select(UserInteractionORM))
    interactions = list(interactions_q.scalars())

    if not interactions:
        return

    from news_agent.models import NewsItemORM
    # Accumulate deltas: {(dimension, value) -> delta}
    deltas: dict[tuple[str, str], float] = {}

    def bump(dim: str, val: str, amount: float):
        if val:
            key = (dim, val.lower()[:128])
            deltas[key] = deltas.get(key, 0.0) + amount

    for interaction in interactions:
        item_orm = await session.get(NewsItemORM, interaction.item_id)
        if not item_orm:
            continue

        weight = _interaction_weight(interaction)
        if weight == 0:
            continue

        # Dimensions: source, each tag, each key entity
        bump("source", item_orm.source, weight)
        for tag in (item_orm.tags or []):
            bump("tag", tag, weight)
        for entity in (item_orm.key_entities or []):
            bump("entity", entity, weight)

    # Upsert all deltas
    for (dimension, value), delta in deltas.items():
        existing = await session.execute(
            select(UserPreferenceORM)
            .where(UserPreferenceORM.dimension == dimension)
            .where(UserPreferenceORM.value == value)
        )
        row = existing.scalar_one_or_none()
        if row:
            row.score += delta
            row.updated_at = datetime.utcnow()
        else:
            session.add(UserPreferenceORM(
                dimension=dimension,
                value=value,
                score=delta,
                updated_at=datetime.utcnow(),
            ))

    logger.info("Preferences recomputed from %d interactions", len(interactions))


def _interaction_weight(interaction: UserInteractionORM) -> float:
    if interaction.action == "star":
        return WEIGHTS["star"]
    if interaction.action == "unstar":
        return WEIGHTS["unstar"]
    if interaction.action == "click":
        return WEIGHTS["click"]
    if interaction.action == "read" and interaction.read_seconds:
        secs = interaction.read_seconds
        if secs >= 180:
            return WEIGHTS["read_long"]
        elif secs >= 60:
            return WEIGHTS["read_medium"]
        elif secs >= 15:
            return WEIGHTS["read_short"]
    return 0.0


async def get_preference_scores(session: AsyncSession) -> dict[tuple[str, str], float]:
    """Return all preference scores as {(dimension, value): score}."""
    result = await session.execute(select(UserPreferenceORM))
    return {(row.dimension, row.value): row.score for row in result.scalars()}


def apply_preference_boost(items: list[NewsItem], prefs: dict[tuple[str, str], float]) -> list[NewsItem]:
    """
    Adjust each item's relevance_score using learned preferences.
    Boost/demote by up to ±3 points based on matching tags, entities, source.
    """
    if not prefs:
        return items

    for item in items:
        boost = 0.0
        # Source preference
        src_score = prefs.get(("source", item.source), 0.0)
        boost += _sigmoid_boost(src_score, scale=0.3)

        # Tag preferences
        for tag in item.tags:
            boost += _sigmoid_boost(prefs.get(("tag", tag.lower()), 0.0), scale=0.5)

        # Entity preferences
        for entity in item.key_entities:
            boost += _sigmoid_boost(prefs.get(("entity", entity.lower()), 0.0), scale=0.4)

        # Cap boost at ±3 and apply
        boost = max(-3.0, min(3.0, boost))
        if item.relevance_score is not None:
            item.relevance_score = max(0.0, min(10.0, item.relevance_score + boost))

    return items


def _sigmoid_boost(raw_score: float, scale: float = 0.5) -> float:
    """Map a raw preference score to a small boost using a soft sigmoid."""
    import math
    return 2.0 / (1.0 + math.exp(-scale * raw_score)) - 1.0
