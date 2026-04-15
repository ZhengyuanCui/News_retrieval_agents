from __future__ import annotations

import logging
from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential

from news_agent.collectors.base import BaseCollector
from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

# Channel-to-topic mapping: add entries here when you add new channels.
# Topic is just a label — it does not restrict what gets fetched.
CHANNEL_TOPICS: dict[str, str] = {
    "UCbmNph6atAoGfqLoCL_duAg": "ai",       # Two Minute Papers
    "UCWX3yGbODI3HLa-7k-4FNHA": "ai",       # Yannic Kilcher
    "UCZHmQk67mSJgfCCTn7xBfew": "ai",       # Andrej Karpathy
    "UCtYLUTtgS3k1Fg4y5tAhLbw": "ai",       # Lex Fridman
    "UCnUYZLuoy1rq1aVMwx4aTzw": "ai",       # Google DeepMind
    "UCrM7B7SL_g1edFOnmj-SDKg": "stocks",   # Bloomberg Markets
    "UCrp_UI8XtuYfpiqluWLD7Lw": "stocks",   # CNBC Television
}


class YouTubeCollector(BaseCollector):
    source_name = "youtube"
    rate_limit_delay = 0.5

    def is_enabled(self) -> bool:
        return settings.youtube_enabled and bool(settings.youtube_api_key)

    def _build_service(self):
        return build("youtube", "v3", developerKey=settings.youtube_api_key)

    def _classify_topic(self, channel_id: str) -> str:
        return CHANNEL_TOPICS.get(channel_id, "youtube")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3))
    def _get_uploads_playlist_id(self, service, channel_id: str) -> str | None:
        resp = service.channels().list(part="contentDetails", id=channel_id).execute()
        items = resp.get("items", [])
        if not items:
            return None
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3))
    def _get_recent_videos(self, service, playlist_id: str, max_results: int = 10) -> list[dict]:
        resp = (
            service.playlistItems()
            .list(part="snippet", playlistId=playlist_id, maxResults=max_results)
            .execute()
        )
        return resp.get("items", [])

    async def fetch(self) -> list[NewsItem]:
        if not self.is_enabled():
            logger.warning("YouTubeCollector disabled or missing API key")
            return []

        service = self._build_service()
        items: list[NewsItem] = []

        channel_ids = settings.youtube_channel_ids
        relevant_ids = list(dict.fromkeys(channel_ids))  # all channels, deduped, preserving order

        for channel_id in relevant_ids:
            try:
                await self._rate_limit()
                playlist_id = self._get_uploads_playlist_id(service, channel_id)
                if not playlist_id:
                    continue

                videos = self._get_recent_videos(service, playlist_id, max_results=5)
                # During broad fetch (no topic filter), label by source not by category
                topic = self._classify_topic(channel_id) if self.topics else "youtube"

                for video in videos:
                    snippet = video.get("snippet", {})
                    video_id = snippet.get("resourceId", {}).get("videoId")
                    if not video_id:
                        continue

                    published_str = snippet.get("publishedAt", "")
                    try:
                        published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except (ValueError, AttributeError):
                        published_at = datetime.utcnow()

                    description = snippet.get("description", "")[:1000]
                    title = snippet.get("title", "")

                    items.append(
                        NewsItem(
                            source="youtube",
                            topic=topic,
                            title=title,
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            content=description or title,
                            author=snippet.get("channelTitle"),
                            published_at=published_at,
                            raw_score=0.5,
                        )
                    )

            except HttpError as e:
                logger.error("YouTube API error for channel %s: %s", channel_id, e)
            except Exception as e:
                logger.error("Unexpected YouTube error for channel %s: %s", channel_id, e)

        logger.info("YouTubeCollector fetched %d items", len(items))
        return items[: settings.max_items_per_source]

    async def fetch_keyword(self, keyword: str) -> list[NewsItem]:
        if not self.is_enabled():
            return []
        service = self._build_service()
        items: list[NewsItem] = []
        try:
            await self._rate_limit()
            resp = service.search().list(
                part="snippet",
                q=keyword,
                type="video",
                order="date",
                maxResults=15,
                relevanceLanguage="en",
            ).execute()
            for result in resp.get("items", []):
                snippet = result.get("snippet", {})
                video_id = result.get("id", {}).get("videoId")
                if not video_id:
                    continue
                published_str = snippet.get("publishedAt", "")
                try:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except (ValueError, AttributeError):
                    published_at = datetime.utcnow()
                items.append(NewsItem(
                    source="youtube", topic=keyword,
                    title=snippet.get("title", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    content=snippet.get("description", "")[:1000] or snippet.get("title", ""),
                    author=snippet.get("channelTitle"),
                    published_at=published_at,
                    raw_score=0.5,
                ))
        except Exception as e:
            logger.error("YouTube keyword fetch error (%r): %s", keyword, e)
        # Tag languages and drop non-English videos (can't translate video content)
        items = self.tag_languages(items)
        items = [i for i in items if i.language == "en"]
        logger.info("YouTubeCollector keyword=%r fetched %d items", keyword, len(items))
        return items
