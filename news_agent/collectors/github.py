from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup
from github import Github, GithubException
from tenacity import retry, stop_after_attempt, wait_exponential

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

TRENDING_URL = "https://github.com/trending?since=daily"
TRENDING_AI_URLS = [
    "https://github.com/trending/python?since=daily",
    "https://github.com/trending/jupyter-notebook?since=daily",
]


class GitHubCollector(BaseCollector):
    source_name = "github"
    rate_limit_delay = 2.0

    def is_enabled(self) -> bool:
        return settings.github_enabled

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3))
    async def _scrape_trending(self, url: str, topic: str) -> list[NewsItem]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsAgent/1.0)"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        items = []
        repo_articles = soup.select("article.Box-row")

        for i, article in enumerate(repo_articles[:25]):
            title_el = article.select_one("h2 a")
            if not title_el:
                continue
            repo_path = title_el.get("href", "").lstrip("/")
            if not repo_path or "/" not in repo_path:
                continue

            description_el = article.select_one("p")
            description = description_el.get_text(strip=True) if description_el else ""

            stars_el = article.select_one("a[href$='/stargazers']")
            stars_text = stars_el.get_text(strip=True).replace(",", "") if stars_el else "0"
            try:
                stars = int(stars_text)
            except ValueError:
                stars = 0

            items.append(
                NewsItem(
                    source="github",
                    topic=topic,
                    title=repo_path,
                    url=f"https://github.com/{repo_path}",
                    content=description or f"Trending GitHub repository: {repo_path}",
                    published_at=datetime.utcnow(),
                    raw_score=self.normalize_score(stars, 0, 50000),
                )
            )

        return items

    async def _fetch_releases(self, topic: str) -> list[NewsItem]:
        g = Github(settings.github_token) if settings.github_token else Github()
        items = []

        for repo_name in settings.github_watch_repos:
            try:
                await self._rate_limit()
                repo = g.get_repo(repo_name)
                releases = repo.get_releases()
                for release in releases[:3]:
                    if release.draft or release.prerelease:
                        continue
                    published = release.published_at
                    if published and published.tzinfo:
                        published = published.replace(tzinfo=None)
                    items.append(
                        NewsItem(
                            source="github",
                            topic=topic,
                            title=f"{repo_name} released {release.tag_name}",
                            url=release.html_url,
                            content=(release.body or "")[:1500] or f"New release {release.tag_name} for {repo_name}",
                            author=release.author.login if release.author else None,
                            published_at=published or datetime.utcnow(),
                            raw_score=0.6,
                        )
                    )
                    break  # only the latest release per repo
            except GithubException as e:
                logger.error("GitHub API error for %s: %s", repo_name, e)

        return items

    async def fetch(self) -> list[NewsItem]:
        if not self.is_enabled():
            logger.warning("GitHubCollector disabled")
            return []

        items: list[NewsItem] = []

        ai_label = "ai" if self.topics else "github"
        stocks_label = "stocks" if self.topics else "github"

        if not self.topics or "ai" in self.topics:
            for url in TRENDING_AI_URLS:
                try:
                    await self._rate_limit()
                    items += await self._scrape_trending(url, ai_label)
                except Exception as e:
                    logger.error("GitHub trending scrape error (%s): %s", url, e)

            try:
                items += await self._fetch_releases(ai_label)
            except Exception as e:
                logger.error("GitHub releases error: %s", e)

        if not self.topics or "stocks" in self.topics:
            try:
                await self._rate_limit()
                items += await self._scrape_trending(
                    "https://github.com/trending?since=daily&q=finance+OR+trading+OR+quant",
                    stocks_label,
                )
            except Exception as e:
                logger.error("GitHub stocks trending error: %s", e)

        # General trending repos (always fetched — high-attention content regardless of topic)
        if not self.topics:
            try:
                await self._rate_limit()
                items += await self._scrape_trending(TRENDING_URL, "github")
            except Exception as e:
                logger.error("GitHub general trending error: %s", e)

        logger.info("GitHubCollector fetched %d items", len(items))
        return items[: settings.max_items_per_source * 2]

    async def fetch_keyword(self, keyword: str) -> list[NewsItem]:
        if not self.is_enabled():
            return []
        g = Github(settings.github_token) if settings.github_token else Github()
        items = []
        try:
            results = g.search_repositories(query=keyword, sort="stars", order="desc")
            for repo in results[:20]:
                pushed = repo.pushed_at
                if pushed and pushed.tzinfo:
                    pushed = pushed.replace(tzinfo=None)
                items.append(NewsItem(
                    source="github",
                    topic=keyword,
                    title=repo.full_name,
                    url=repo.html_url,
                    content=repo.description or f"GitHub repository: {repo.full_name}",
                    published_at=pushed or datetime.utcnow(),
                    raw_score=self.normalize_score(repo.stargazers_count, 0, 100000),
                ))
        except Exception as e:
            logger.error("GitHubCollector keyword fetch error (%r): %s", keyword, e)
        logger.info("GitHubCollector keyword=%r fetched %d items", keyword, len(items))
        return items
