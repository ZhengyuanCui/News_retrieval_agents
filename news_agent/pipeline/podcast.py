from __future__ import annotations

import logging
from pathlib import Path

import litellm

from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

SCRIPT_PROMPT = """\
You are a professional news podcast host. Write a natural, engaging spoken-word podcast script \
for today's {topic_label} briefing based on the news items below.

Guidelines:
- Open with a warm intro: "Welcome to today's {topic_label} briefing. I'm your AI host. Here's what's happening."
- Cover 5-8 of the most important stories. For each, speak naturally in 2-4 sentences — \
  no bullet points, no markdown, no headers. Use transitions like "Moving on...", "Also today...", \
  "In other news...", "And finally..."
- Close with: "That's your {topic_label} briefing for today. Stay informed."
- Write ONLY the spoken words. No stage directions, no sound cues, no labels.
- Keep the total script under 700 words so the audio stays under 5 minutes.

News items:
{items_text}
"""


def _llm_model() -> str:
    if settings.claude_model:
        return f"anthropic/{settings.claude_model}"
    return settings.llm_model


def _api_key() -> str | None:
    return settings.llm_api_key or settings.anthropic_api_key or None


class PodcastGenerator:
    def _write_script(self, items: list[NewsItem], topic: str) -> str:
        topic_label = topic.title()
        relevant = sorted(
            [i for i in items if not i.is_duplicate],
            key=lambda x: x.relevance_score or x.raw_score or 0,
            reverse=True,
        )[:15]

        items_text = "\n\n".join(
            f"[{i+1}] {item.title}\n{item.summary or item.content[:300]}"
            for i, item in enumerate(relevant)
        )

        api_key = _api_key()
        response = litellm.completion(
            model=_llm_model(),
            max_tokens=1500,
            messages=[{"role": "user", "content": SCRIPT_PROMPT.format(
                topic_label=topic_label,
                items_text=items_text,
            )}],
            **({"api_key": api_key} if api_key else {}),
        )
        return response.choices[0].message.content.strip()

    def _tts_openai(self, script: str, output_path: Path) -> None:
        """Primary TTS via OpenAI."""
        from openai import OpenAI
        tts_client = OpenAI(api_key=settings.openai_api_key)
        with tts_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=settings.podcast_voice,
            input=script,
            response_format="mp3",
        ) as resp:
            resp.stream_to_file(output_path)

    def _tts_gtts(self, script: str, output_path: Path) -> None:
        """Fallback TTS via gTTS (Google, free, no API key required)."""
        from gtts import gTTS
        gTTS(text=script, lang="en").save(str(output_path))

    def generate(self, items: list[NewsItem], topic: str, output_path: Path) -> Path:
        """Generate a podcast MP3. Uses OpenAI TTS if available, falls back to gTTS."""
        if not (settings.llm_api_key or settings.anthropic_api_key):
            raise RuntimeError("No LLM API key configured (set LLM_API_KEY or ANTHROPIC_API_KEY)")

        script = self._write_script(items, topic)
        logger.info("Podcast script written (%d chars)", len(script))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try OpenAI first
        if settings.openai_api_key:
            try:
                self._tts_openai(script, output_path)
                logger.info("Podcast generated via OpenAI TTS → %s", output_path)
                return output_path
            except Exception as e:
                logger.warning("OpenAI TTS failed (%s), falling back to gTTS", e)

        # Fallback: gTTS (Google, free)
        logger.info("Using gTTS fallback")
        self._tts_gtts(script, output_path)
        logger.info("Podcast generated via gTTS → %s", output_path)
        return output_path
