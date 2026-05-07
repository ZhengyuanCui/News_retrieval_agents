from __future__ import annotations

import pytest
from click.testing import CliRunner

from news_agent.cli import main


@pytest.fixture
async def isolated_db():
    """Override the global autouse DB fixture; this test does not hit the DB."""
    yield


def test_newsletter_cli_passes_format(monkeypatch):
    called: dict = {}

    async def fake_init_db() -> None:
        called["init_db"] = True

    async def fake_build_and_send_newsletter(**kwargs):
        called["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr("news_agent.storage.init_db", fake_init_db)
    monkeypatch.setattr(
        "news_agent.pipeline.newsletter.build_and_send_newsletter",
        fake_build_and_send_newsletter,
    )

    result = CliRunner().invoke(
        main,
        ["newsletter", "--topics", "ai,stocks", "--format", "monologue", "--no-refresh"],
    )

    assert result.exit_code == 0
    assert called["init_db"] is True
    assert called["kwargs"]["topics"] == ["ai", "stocks"]
    assert called["kwargs"]["podcast_format"] == "monologue"
    assert called["kwargs"]["refresh"] is False
