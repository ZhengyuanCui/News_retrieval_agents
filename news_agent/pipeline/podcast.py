from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

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

DIALOGUE_PROMPT = """\
You are writing a two-speaker news podcast dialogue for today's {topic_label} briefing.

Write exactly alternating lines between HOST and ANALYST, starting with HOST.
- Keep it natural and concise.
- Cover the most important stories from the items below.
- No stage directions, markdown, bullets, or headers.
- Each line must begin with either "HOST:" or "ANALYST:".
- Keep the total conversation to at most {max_turns} lines.

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
    def _relevant_items(self, items: list[NewsItem]) -> list[NewsItem]:
        return sorted(
            [i for i in items if not i.is_duplicate],
            key=lambda x: x.relevance_score or x.raw_score or 0,
            reverse=True,
        )[:15]

    def _items_text(self, items: list[NewsItem]) -> str:
        return "\n\n".join(
            f"[{i+1}] {item.title}\n{item.summary or item.content[:300]}"
            for i, item in enumerate(items)
        )

    def _write_script(self, items: list[NewsItem], topic: str) -> str:
        topic_label = topic.title()
        items_text = self._items_text(self._relevant_items(items))

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

    def _generate_dialogue_transcript(self, items: list[NewsItem], topic: str) -> list[tuple[str, str]]:
        topic_label = topic.title()
        items_text = self._items_text(self._relevant_items(items))
        api_key = _api_key()
        response = litellm.completion(
            model=_llm_model(),
            max_tokens=1500,
            messages=[{"role": "user", "content": DIALOGUE_PROMPT.format(
                topic_label=topic_label,
                items_text=items_text,
                max_turns=settings.podcast_dialogue_max_turns,
            )}],
            **({"api_key": api_key} if api_key else {}),
        )
        raw = response.choices[0].message.content.strip().splitlines()
        transcript: list[tuple[str, str]] = []
        expected = "host"
        for line in raw:
            text = line.strip()
            if not text or ":" not in text:
                continue
            speaker_label, spoken = text.split(":", 1)
            speaker = speaker_label.strip().lower()
            if speaker not in {"host", "analyst"}:
                continue
            if speaker != expected:
                speaker = expected
            spoken = spoken.strip()
            if not spoken:
                continue
            transcript.append((speaker, spoken))
            if len(transcript) >= settings.podcast_dialogue_max_turns:
                break
            expected = "analyst" if expected == "host" else "host"
        return transcript

    def _tts_openai_segment(self, script: str, voice: str, output_path: Path) -> None:
        """Primary TTS via OpenAI."""
        from openai import OpenAI
        tts_client = OpenAI(api_key=settings.openai_api_key)
        with tts_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=script,
            response_format="mp3",
        ) as resp:
            resp.stream_to_file(output_path)

    def _tts_openai(self, script: str, output_path: Path) -> None:
        self._tts_openai_segment(script, settings.podcast_voice, output_path)

    def _tts_openai_dialogue(self, transcript: list[tuple[str, str]], output_path: Path) -> None:
        combined = bytearray()
        with TemporaryDirectory() as tmpdir:
            for i, (speaker, line) in enumerate(transcript):
                segment_path = Path(tmpdir) / f"segment-{i}.mp3"
                voice = (
                    settings.podcast_host_voice
                    if speaker == "host"
                    else settings.podcast_analyst_voice
                )
                text = line.strip()
                if i < len(transcript) - 1 and text and not text.endswith(("...", "…")):
                    text = f"{text} ..."
                self._tts_openai_segment(text, voice, segment_path)
                combined.extend(segment_path.read_bytes())
        output_path.write_bytes(bytes(combined))

    def _tts_gtts(self, script: str, output_path: Path) -> None:
        """Fallback TTS via gTTS (Google, free, no API key required)."""
        from gtts import gTTS
        gTTS(text=script, lang="en").save(str(output_path))

    def _tts_gtts_dialogue(self, transcript: list[tuple[str, str]], output_path: Path) -> None:
        script = "\n\n".join(f"{speaker.title()}: {line}" for speaker, line in transcript)
        self._tts_gtts(script, output_path)

    def generate(
        self,
        items: list[NewsItem],
        topic: str,
        output_path: Path,
        format: str | None = None,
    ) -> Path:
        """Generate a podcast MP3. Uses OpenAI TTS if available, falls back to gTTS."""
        if not (settings.llm_api_key or settings.anthropic_api_key):
            raise RuntimeError("No LLM API key configured (set LLM_API_KEY or ANTHROPIC_API_KEY)")

        format = format or settings.podcast_format
        transcript: list[tuple[str, str]] | None = None
        script: str | None = None
        if format == "dialogue":
            transcript = self._generate_dialogue_transcript(items, topic)
            logger.info("Podcast dialogue transcript written (%d turns)", len(transcript))
        else:
            script = self._write_script(items, topic)
            logger.info("Podcast script written (%d chars)", len(script))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try OpenAI first
        if settings.openai_api_key:
            try:
                if format == "dialogue":
                    self._tts_openai_dialogue(transcript or [], output_path)
                else:
                    self._tts_openai(script or "", output_path)
                logger.info("Podcast generated via OpenAI TTS → %s", output_path)
                return output_path
            except Exception as e:
                logger.warning("OpenAI TTS failed (%s), falling back to gTTS", e)

        # Fallback: gTTS (Google, free)
        logger.info("Using gTTS fallback")
        if format == "dialogue":
            self._tts_gtts_dialogue(transcript or [], output_path)
        else:
            self._tts_gtts(script or "", output_path)
        logger.info("Podcast generated via gTTS → %s", output_path)
        return output_path
