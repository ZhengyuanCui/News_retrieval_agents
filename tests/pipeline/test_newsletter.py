"""Tests for the daily newsletter pipeline."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from news_agent.emailer import EmailError
from news_agent.pipeline import newsletter as nl
from news_agent.storage import NewsRepository, get_session
from tests.conftest import make_item


# ── Topic resolution ────────────────────────────────────────────────────────

async def test_get_default_topics_env_override(monkeypatch):
    """NEWSLETTER_TOPICS setting takes priority over UI-saved setting."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_topics", ["ai", "stocks"])

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.set_setting("default_topics", "ignored|ignored2")

    topics = await nl.get_default_topics()
    assert topics == ["ai", "stocks"]


async def test_get_default_topics_falls_back_to_ui_saved(monkeypatch):
    """When NEWSLETTER_TOPICS is empty, use the UI-saved default_topics setting."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_topics", [])

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.set_setting("default_topics", "ai|stocks")

    topics = await nl.get_default_topics()
    assert topics == ["ai", "stocks"]


async def test_get_default_topics_skips_empty_slots(monkeypatch):
    """UI panel 2 left blank → only panel 1 returned."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_topics", [])

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.set_setting("default_topics", "ai|")

    topics = await nl.get_default_topics()
    assert topics == ["ai"]


async def test_get_default_topics_empty_returns_empty(monkeypatch):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_topics", [])
    topics = await nl.get_default_topics()
    assert topics == []


# ── Setting persistence (UserSettingORM round-trip) ─────────────────────────

async def test_user_setting_roundtrip():
    async with get_session() as session:
        repo = NewsRepository(session)
        assert await repo.get_setting("missing", "default") == "default"
        await repo.set_setting("default_topics", "ai|markets")

    async with get_session() as session:
        repo = NewsRepository(session)
        assert await repo.get_setting("default_topics") == "ai|markets"

        await repo.set_setting("default_topics", "crypto|")
        assert await repo.get_setting("default_topics") == "crypto|"


# ── HTML rendering ──────────────────────────────────────────────────────────

def test_render_topic_section_includes_title_and_items():
    items = [
        make_item(title="Story One", source="rss", url="https://a.com",
                  summary="First story summary.", tags=["alpha", "beta"]),
        make_item(title="Story Two", source="github", url="https://b.com/abc",
                  summary="Second summary.", sentiment="positive"),
    ]
    digest = {"headline": "Today in **AI**", "bullets": ["Point **one**", "Point two"]}
    html_out = nl._render_topic_section("ai", digest, items)

    assert "Ai" in html_out or "AI" in html_out
    assert "Story One" in html_out
    assert "Story Two" in html_out
    assert "https://a.com" in html_out
    assert "<strong>AI</strong>" in html_out
    assert "<strong>one</strong>" in html_out
    assert "positive" in html_out
    assert "alpha" in html_out


def test_render_topic_section_escapes_html_in_item_fields():
    items = [make_item(title="<script>alert(1)</script>",
                       content="<b>raw</b>",
                       url="https://x.com/?q=<>&")]
    html_out = nl._render_topic_section("ai", None, items)
    assert "<script>alert" not in html_out
    assert "&lt;script&gt;" in html_out


def test_render_email_html_has_no_top_audio_note():
    """The top-of-email audio note was removed — each topic now carries its
    own inline audio player, so the email body should not advertise the
    attachments at the top."""
    sections = [nl._render_topic_section("ai", None, [])]
    with_audio = nl._render_email_html(date_str="Jan 1, 2026", sections=sections, has_audio=True)
    without_audio = nl._render_email_html(date_str="Jan 1, 2026", sections=sections, has_audio=False)
    assert "audio briefing is attached" not in with_audio
    assert "audio briefings are attached" not in with_audio
    assert "audio briefing is attached" not in without_audio


def test_render_email_html_ignores_audio_topic_count():
    """audio_topic_count is accepted for backwards compat but no longer
    rendered as a header note."""
    sections = [nl._render_topic_section("ai", None, []),
                nl._render_topic_section("stocks", None, [])]
    out = nl._render_email_html(
        date_str="Jan 1, 2026", sections=sections,
        has_audio=True, audio_topic_count=2,
    )
    assert "2 audio briefings are attached" not in out
    assert "one per topic" not in out


def test_render_topic_section_embeds_inline_audio_player():
    """Each topic section should embed an inline <audio> element pointing
    at its own MP3 attachment (via cid:) so the user can play the briefing
    on the page rather than downloading the file."""
    html_out = nl._render_topic_section(
        "ai",
        None,
        [make_item(title="Item", url="https://a.com/1")],
        audio_filename="ai-briefing-2026-01-01.mp3",
        audio_content_id="audio-ai-2026-01-01",
    )
    assert "cid:audio-ai-2026-01-01" in html_out
    assert "<audio" in html_out and "controls" in html_out
    assert "Listen to this topic" in html_out
    # Instructions for opening the attachment in a mail client should be gone.
    assert "open the attachment" not in html_out
    assert "in your mail client" not in html_out


def test_render_topic_section_has_no_audio_block_when_absent():
    html_out = nl._render_topic_section(
        "ai", None, [make_item(title="Item", url="https://a.com/1")],
    )
    assert "topic-audio" not in html_out
    assert "cid:" not in html_out


def test_topic_slug_is_filesystem_and_cid_safe():
    assert nl._topic_slug("AI") == "ai"
    assert nl._topic_slug("AI & Robotics") == "ai-robotics"
    assert nl._topic_slug("  stocks/bonds  ") == "stocks-bonds"
    assert nl._topic_slug("") == "topic"
    assert nl._topic_slug("!!!") == "topic"


def test_parse_digest_handles_plain_and_legacy_formats():
    assert nl._parse_digest(None) is None
    assert nl._parse_digest("") is None
    legacy = nl._parse_digest("Headline|||Bullet 1|||Bullet 2")
    assert legacy == {"headline": "Headline", "bullets": ["Bullet 1", "Bullet 2"]}
    plain = nl._parse_digest("Headline line\nBullet a\nBullet b")
    assert plain == {"headline": "Headline line", "bullets": ["Bullet a", "Bullet b"]}


# ── build_and_send_newsletter ───────────────────────────────────────────────

async def test_build_and_send_requires_recipient(monkeypatch):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai"])

    with pytest.raises(EmailError, match="No recipient"):
        await nl.build_and_send_newsletter(refresh=False)


async def test_build_and_send_requires_topics(monkeypatch):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", [])

    with pytest.raises(EmailError, match="No topics"):
        await nl.build_and_send_newsletter(refresh=False)


async def test_build_and_send_happy_path(monkeypatch):
    """End-to-end: seed DB items, call build_and_send, verify send_email called."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai"])
    monkeypatch.setattr(settings, "newsletter_include_audio", False)
    monkeypatch.setattr(settings, "llm_api_key", "")  # skip digest generation
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    items = [
        make_item(title="AI item 1", url="https://a.com/1", topic="ai"),
        make_item(title="AI item 2", url="https://a.com/2", topic="ai"),
    ]
    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many(items)

    captured: dict = {}

    def fake_send_email(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(nl, "send_email", fake_send_email)

    result = await nl.build_and_send_newsletter(refresh=False)

    assert result["recipient"] == "me@example.com"
    assert result["topics"] == ["ai"]
    assert result["items"] >= 2
    assert result["audio_included"] is False
    assert result["refreshed"] is False
    assert result["fetch_results"] == {}

    assert captured["to"] == "me@example.com"
    assert "AI item 1" in captured["html_body"]
    assert "AI item 2" in captured["html_body"]
    assert captured["attachments"] == []
    assert "News Digest" in captured["subject"]


async def test_build_and_send_uses_ui_saved_topics_when_env_empty(monkeypatch):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", [])
    monkeypatch.setattr(settings, "newsletter_include_audio", False)
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.set_setting("default_topics", "stocks|")
        await repo.upsert_many([
            make_item(title="Stocks item", topic="stocks", url="https://s.com/1"),
        ])

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)
    assert result["topics"] == ["stocks"]
    assert "Stocks item" in captured["html_body"]


async def test_build_and_send_generates_audio_per_topic(monkeypatch, tmp_path):
    """When include_audio is on and multiple topics have items, we generate
    one MP3 per topic and attach each one separately.  Each topic section
    in the HTML body references its own audio filename."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai", "stocks"])
    monkeypatch.setattr(settings, "newsletter_include_audio", True)
    monkeypatch.setattr(settings, "newsletter_audio_dir", str(tmp_path / "audio"))
    monkeypatch.setattr(settings, "public_base_url", "")
    monkeypatch.setattr(settings, "newsletter_attach_audio", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many([
            make_item(title="AI item", topic="ai", url="https://a.com/ai"),
            make_item(title="Stocks item", topic="stocks", url="https://s.com/st"),
        ])

    # Fake out the (synchronous) podcast generator so we don't call any TTS.
    generated_for: list[str] = []

    def fake_topic_audio(topic, items, output_path):
        generated_for.append(topic)
        output_path.write_bytes(f"MP3-{topic}".encode())
        return output_path

    monkeypatch.setattr(nl, "_generate_topic_audio", fake_topic_audio)

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)

    # One MP3 per topic, both invoked with the correct topic label
    assert sorted(generated_for) == ["ai", "stocks"]
    assert result["audio_included"] is True
    assert sorted(result["audio_topics"]) == ["ai", "stocks"]
    assert len(result["audio_files"]) == 2
    assert all(f.endswith(".mp3") for f in result["audio_files"])
    assert any("ai-briefing-" in f for f in result["audio_files"])
    assert any("stocks-briefing-" in f for f in result["audio_files"])

    # Each attachment is its own (filename, bytes, mime, cid) tuple
    attachments = captured["attachments"]
    assert len(attachments) == 2
    by_topic = {a[0].split("-")[0]: a for a in attachments}
    assert set(by_topic.keys()) == {"ai", "stocks"}
    for topic, (fname, data, mime, cid) in by_topic.items():
        assert mime == "audio/mpeg"
        assert data == f"MP3-{topic}".encode()
        assert cid.startswith(f"audio-{topic}-")

    # Each topic section in the HTML body references its own MP3 via cid:
    html_body = captured["html_body"]
    assert "cid:audio-ai-" in html_body
    assert "cid:audio-stocks-" in html_body
    # Inline audio players are embedded (not download links)
    assert html_body.count("<audio") == 2
    # The old top-of-email attachment note has been removed
    assert "audio briefings are attached" not in html_body
    assert "open the attachment" not in html_body


async def test_build_and_send_skips_audio_for_topics_without_items(monkeypatch, tmp_path):
    """A topic with zero items should not get an audio attachment, even
    when other topics in the same newsletter do."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai", "empty-topic"])
    monkeypatch.setattr(settings, "newsletter_include_audio", True)
    monkeypatch.setattr(settings, "newsletter_audio_dir", str(tmp_path / "audio"))
    monkeypatch.setattr(settings, "public_base_url", "")
    monkeypatch.setattr(settings, "newsletter_attach_audio", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many([
            make_item(title="AI item", topic="ai", url="https://a.com/ai"),
        ])

    generated_for: list[str] = []

    def fake_topic_audio(topic, items, output_path):
        generated_for.append(topic)
        output_path.write_bytes(b"MP3")
        return output_path

    monkeypatch.setattr(nl, "_generate_topic_audio", fake_topic_audio)

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)
    assert generated_for == ["ai"]
    assert result["audio_topics"] == ["ai"]
    assert len(captured["attachments"]) == 1


async def test_build_and_send_skips_audio_when_no_items(monkeypatch, tmp_path):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["nonexistent"])
    monkeypatch.setattr(settings, "newsletter_include_audio", True)
    monkeypatch.setattr(settings, "newsletter_audio_dir", str(tmp_path / "audio"))
    monkeypatch.setattr(settings, "public_base_url", "")
    monkeypatch.setattr(settings, "newsletter_attach_audio", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake")

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)
    assert result["items"] == 0
    assert result["audio_included"] is False
    assert captured["attachments"] == []


async def test_build_and_send_embeds_public_audio_url(monkeypatch, tmp_path):
    """When PUBLIC_BASE_URL is configured the email embeds a streamable
    https:// <audio> player instead of (or in addition to) the cid:
    attachment.  This is the path that actually works in Gmail / Outlook —
    those clients strip cid: audio, but will open the https URL in a
    browser tab that streams the MP3 inline."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai"])
    monkeypatch.setattr(settings, "newsletter_include_audio", True)
    monkeypatch.setattr(settings, "newsletter_audio_dir", str(tmp_path / "audio"))
    monkeypatch.setattr(settings, "public_base_url", "https://news.example.com")
    monkeypatch.setattr(settings, "newsletter_attach_audio", False)
    monkeypatch.setattr(settings, "llm_api_key", "fake")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many([
            make_item(title="AI item", topic="ai", url="https://a.com/ai"),
        ])

    def fake_topic_audio(topic, items, output_path):
        output_path.write_bytes(b"MP3-bytes")
        return output_path

    monkeypatch.setattr(nl, "_generate_topic_audio", fake_topic_audio)

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)

    # The MP3 was persisted under newsletter_audio_dir so the web app can
    # stream it.
    audio_dir = tmp_path / "audio"
    mp3s = list(audio_dir.glob("*.mp3"))
    assert len(mp3s) == 1, f"expected one mp3, got {mp3s}"
    assert mp3s[0].read_bytes() == b"MP3-bytes"

    # The email body embeds the streamable https URL — no cid: needed.
    html_body = captured["html_body"]
    assert "https://news.example.com/newsletter/audio/ai-briefing-" in html_body
    assert "<audio" in html_body
    assert "cid:audio-ai-" not in html_body

    # The <audio> element's src points at the raw streaming endpoint
    # (/newsletter/audio/...mp3); the "Play briefing in browser" link
    # points at the /newsletter/player/... page (no .mp3 suffix) so
    # clicking it never triggers a download.
    import re as _re
    listen_hrefs = _re.findall(r'a class="listen"[^>]*href="([^"]+)"', html_body)
    audio_srcs = _re.findall(r'<audio[^>]*src="([^"]+)"', html_body)
    assert listen_hrefs, "email should contain a listen link"
    for href in listen_hrefs:
        assert "/newsletter/player/" in href, f"listen link must use player page, got {href}"
        assert not href.endswith(".mp3"), f"listen link must not end in .mp3, got {href}"
    for src in audio_srcs:
        assert "/newsletter/audio/" in src
        assert src.endswith(".mp3")

    # With newsletter_attach_audio=False we send the email without attaching
    # the MP3 — playback relies entirely on the public URL.
    assert captured["attachments"] == []

    # The summary exposes the public URL for logging / UI display.
    assert len(result["audio_urls"]) == 1
    assert result["audio_urls"][0].startswith(
        "https://news.example.com/newsletter/audio/ai-briefing-"
    )


async def test_build_and_send_without_public_url_falls_back_to_cid(monkeypatch, tmp_path):
    """If PUBLIC_BASE_URL isn't set we still attach the MP3 and fall back
    to the cid: player so the email isn't completely silent."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai"])
    monkeypatch.setattr(settings, "newsletter_include_audio", True)
    monkeypatch.setattr(settings, "newsletter_audio_dir", str(tmp_path / "audio"))
    monkeypatch.setattr(settings, "public_base_url", "")
    monkeypatch.setattr(settings, "newsletter_attach_audio", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many([
            make_item(title="AI item", topic="ai", url="https://a.com/ai"),
        ])

    monkeypatch.setattr(
        nl, "_generate_topic_audio",
        lambda topic, items, output_path: (output_path.write_bytes(b"MP3"), output_path)[1],
    )
    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)
    assert result["audio_urls"] == []
    html_body = captured["html_body"]
    assert "cid:audio-ai-" in html_body
    assert "https://" not in html_body.split("<audio", 1)[1].split("</audio>", 1)[0]
    assert len(captured["attachments"]) == 1


# ── Fetch-before-send flow ──────────────────────────────────────────────────

async def test_build_and_send_refreshes_before_sending(monkeypatch):
    """With refresh=True (default) the newsletter fetches each topic first
    and the fetch results are included in the response."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai", "stocks"])
    monkeypatch.setattr(settings, "newsletter_include_audio", False)
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    called_with: list[list[str]] = []

    async def fake_fetch_and_analyze_topics(topics):
        called_with.append(list(topics))
        # Simulate the fetch by seeding the DB with fresh items
        async with get_session() as session:
            repo = NewsRepository(session)
            await repo.upsert_many([
                make_item(title=f"Fresh {t}", topic=t, url=f"https://x.com/{t}",
                          summary=f"Fresh {t} summary")
                for t in topics
            ])
        return {t: {"items_stored": 1, "analyzed": 1} for t in topics}

    monkeypatch.setattr(nl, "fetch_and_analyze_topics", fake_fetch_and_analyze_topics)

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter()

    # The fetch helper received the resolved topics, in order
    assert called_with == [["ai", "stocks"]]
    assert result["refreshed"] is True
    assert set(result["fetch_results"].keys()) == {"ai", "stocks"}
    assert result["fetch_results"]["ai"]["analyzed"] == 1

    # The freshly-fetched items made it into the email
    assert "Fresh ai" in captured["html_body"]
    assert "Fresh stocks" in captured["html_body"]


async def test_build_and_send_continues_when_fetch_fails(monkeypatch):
    """If the pre-fetch raises, the newsletter still sends with whatever
    is already in the DB so a transient network outage doesn't silence the
    daily email entirely."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai"])
    monkeypatch.setattr(settings, "newsletter_include_audio", False)
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many([
            make_item(title="Cached AI item", topic="ai", url="https://a.com/cached"),
        ])

    async def broken_fetch(topics):
        raise RuntimeError("network down")

    monkeypatch.setattr(nl, "fetch_and_analyze_topics", broken_fetch)

    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter()

    assert result["refreshed"] is True
    assert result["fetch_results"] == {}  # fetch failed → no per-topic results
    assert "Cached AI item" in captured["html_body"]
    assert captured["to"] == "me@example.com"


async def test_build_and_send_refresh_false_skips_fetch(monkeypatch):
    """refresh=False must not call fetch_and_analyze_topics at all."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "newsletter_email_to", "me@example.com")
    monkeypatch.setattr(settings, "newsletter_topics", ["ai"])
    monkeypatch.setattr(settings, "newsletter_include_audio", False)
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    async with get_session() as session:
        repo = NewsRepository(session)
        await repo.upsert_many([
            make_item(title="Only item", topic="ai", url="https://a.com/only"),
        ])

    async def should_not_run(_topics):
        raise AssertionError("fetch should not be called when refresh=False")

    monkeypatch.setattr(nl, "fetch_and_analyze_topics", should_not_run)
    captured: dict = {}
    monkeypatch.setattr(nl, "send_email", lambda **kw: captured.update(kw))

    result = await nl.build_and_send_newsletter(refresh=False)
    assert result["refreshed"] is False
    assert "Only item" in captured["html_body"]
