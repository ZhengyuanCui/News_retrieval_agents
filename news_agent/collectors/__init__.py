from __future__ import annotations

import logging

from news_agent.collectors.base import BaseCollector
from news_agent.collectors.github import GitHubCollector
from news_agent.collectors.reddit import RedditCollector
from news_agent.collectors.rss import RSSCollector
from news_agent.collectors.twitter import TwitterCollector
from news_agent.collectors.youtube import YouTubeCollector

logger = logging.getLogger(__name__)

ALL_COLLECTORS: list[type[BaseCollector]] = [
    RedditCollector,
    GitHubCollector,
    YouTubeCollector,
    TwitterCollector,
    RSSCollector,
]


def get_enabled_collectors(topics: list[str] | None = None) -> list[BaseCollector]:
    """Return instances of all enabled collectors."""
    collectors = []
    for cls in ALL_COLLECTORS:
        instance = cls(topics=topics)
        if instance.is_enabled():
            collectors.append(instance)
        else:
            logger.debug("%s is disabled (missing credentials or config)", cls.source_name)
    return collectors


__all__ = [
    "BaseCollector",
    "RedditCollector",
    "GitHubCollector",
    "YouTubeCollector",
    "TwitterCollector",
    "get_enabled_collectors",
]
