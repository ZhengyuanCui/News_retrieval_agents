from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from news_agent.config import settings
from news_agent.models import NewsItem
from news_agent.pipeline.podcast import PodcastGenerator


@pytest.fixture
async def isolated_db():
    """Override the global autouse DB fixture; these tests are pure unit tests."""
    yield


def make_item(**kwargs) -> NewsItem:
    defaults = dict(
        source="rss",
        topic="ai",
        title="Default test title",
        url="https://example.com/article",
        content="Some content about artificial intelligence and machine learning.",
        published_at=datetime.utcnow(),
        raw_score=0.5,
        relevance_score=5.0,
    )
    defaults.update(kwargs)
    return NewsItem(**defaults)


def _completion_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_generate_dialogue_transcript_caps_and_alternates(monkeypatch):
    monkeypatch.setattr(settings, "podcast_dialogue_max_turns", 4)
    monkeypatch.setattr(
        "news_agent.pipeline.podcast.litellm.completion",
        lambda **_: _completion_response(
            "\n".join([
                "HOST: Welcome back.",
                "ANALYST: The chip story leads today.",
                "HOST: Markets reacted quickly.",
                "ANALYST: Watch guidance next.",
                "HOST: This line should be trimmed.",
            ])
        ),
    )

    transcript = PodcastGenerator()._generate_dialogue_transcript(
        [make_item(title="AI chip demand surges", summary="Demand is up.")],
        "ai",
    )

    assert transcript == [
        ("host", "Welcome back."),
        ("analyst", "The chip story leads today."),
        ("host", "Markets reacted quickly."),
        ("analyst", "Watch guidance next."),
    ]


def test_tts_openai_dialogue_uses_two_voices_and_stitches(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "podcast_host_voice", "alloy")
    monkeypatch.setattr(settings, "podcast_analyst_voice", "onyx")

    calls: list[tuple[str, str]] = []

    def fake_segment(script: str, voice: str, output_path: Path) -> None:
        calls.append((voice, script))
        output_path.write_bytes(voice.encode())

    gen = PodcastGenerator()
    monkeypatch.setattr(gen, "_tts_openai_segment", fake_segment)

    out = tmp_path / "dialogue.mp3"
    gen._tts_openai_dialogue(
        [("host", "First line."), ("analyst", "Second line.")],
        out,
    )

    assert calls == [
        ("alloy", "First line. ..."),
        ("onyx", "Second line."),
    ]
    assert out.read_bytes() == b"alloyonyx"


def test_generate_preserves_monologue_and_dialogue_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_api_key", "fake")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "fake-openai")

    items = [make_item(title="Story", summary="Summary")]
    gen = PodcastGenerator()

    seen: dict[str, object] = {}

    monkeypatch.setattr(gen, "_write_script", lambda *_: "mono script")
    monkeypatch.setattr(gen, "_generate_dialogue_transcript", lambda *_: [("host", "hello")])
    monkeypatch.setattr(
        gen,
        "_tts_openai",
        lambda script, output_path: (seen.setdefault("monologue", script), output_path.write_bytes(b"mono")),
    )
    monkeypatch.setattr(
        gen,
        "_tts_openai_dialogue",
        lambda transcript, output_path: (seen.setdefault("dialogue", transcript), output_path.write_bytes(b"dialogue")),
    )

    mono_path = tmp_path / "mono.mp3"
    dialogue_path = tmp_path / "dialogue.mp3"

    gen.generate(items, "ai", mono_path, format="monologue")
    gen.generate(items, "ai", dialogue_path, format="dialogue")

    assert seen["monologue"] == "mono script"
    assert seen["dialogue"] == [("host", "hello")]
    assert mono_path.read_bytes() == b"mono"
    assert dialogue_path.read_bytes() == b"dialogue"


def test_generate_dialogue_gtts_fallback_still_writes_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_api_key", "fake")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")

    gen = PodcastGenerator()
    monkeypatch.setattr(gen, "_generate_dialogue_transcript", lambda *_: [("host", "hello"), ("analyst", "hi")])
    monkeypatch.setattr(
        gen,
        "_tts_gtts_dialogue",
        lambda transcript, output_path: output_path.write_bytes(b"fallback-dialogue"),
    )

    out = tmp_path / "fallback.mp3"
    result = gen.generate([make_item(title="Story", summary="Summary")], "ai", out, format="dialogue")

    assert result == out
    assert out.read_bytes() == b"fallback-dialogue"
