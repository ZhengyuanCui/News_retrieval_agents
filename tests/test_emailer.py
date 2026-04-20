"""Tests for the SMTP emailer module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from news_agent.emailer import EmailError, send_email


def test_send_email_raises_when_smtp_unconfigured(monkeypatch):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "smtp_user", "")
    monkeypatch.setattr(settings, "smtp_password", "")
    with pytest.raises(EmailError, match="SMTP not configured"):
        send_email(to="x@y.com", subject="hi", html_body="<p>hi</p>")


def test_send_email_requires_recipient(monkeypatch):
    from news_agent.config import settings
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_user", "bot@example.com")
    monkeypatch.setattr(settings, "smtp_password", "secret")
    with pytest.raises(EmailError, match="No recipient"):
        send_email(to="", subject="hi", html_body="<p>hi</p>")


def test_send_email_starttls_path(monkeypatch):
    """Port 587 + use_tls=True → uses smtplib.SMTP + starttls."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "smtp_host", "smtp.gmail.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_user", "bot@gmail.com")
    monkeypatch.setattr(settings, "smtp_password", "app-pw")
    monkeypatch.setattr(settings, "smtp_use_tls", True)
    monkeypatch.setattr(settings, "newsletter_email_from", "")

    smtp_mock = MagicMock()
    smtp_cm = MagicMock()
    smtp_cm.__enter__.return_value = smtp_mock
    smtp_cm.__exit__.return_value = False

    with patch("news_agent.emailer.smtplib.SMTP", return_value=smtp_cm) as smtp_ctor:
        send_email(
            to="me@example.com",
            subject="Test",
            html_body="<p>body</p>",
            attachments=[("a.mp3", b"\x00\x01audio", "audio/mpeg")],
        )

    smtp_ctor.assert_called_once()
    smtp_mock.starttls.assert_called_once()
    smtp_mock.login.assert_called_once_with("bot@gmail.com", "app-pw")
    smtp_mock.send_message.assert_called_once()
    sent_msg = smtp_mock.send_message.call_args[0][0]
    assert sent_msg["To"] == "me@example.com"
    assert sent_msg["From"] == "bot@gmail.com"
    assert sent_msg["Subject"] == "Test"
    # Has an attachment
    payloads = list(sent_msg.iter_attachments())
    assert len(payloads) == 1
    assert payloads[0].get_filename() == "a.mp3"


def test_send_email_ssl_path(monkeypatch):
    """Port 465 → uses smtplib.SMTP_SSL (no starttls)."""
    from news_agent.config import settings
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 465)
    monkeypatch.setattr(settings, "smtp_user", "bot@x.com")
    monkeypatch.setattr(settings, "smtp_password", "pw")
    monkeypatch.setattr(settings, "smtp_use_tls", True)
    monkeypatch.setattr(settings, "newsletter_email_from", "")

    smtp_mock = MagicMock()
    smtp_cm = MagicMock()
    smtp_cm.__enter__.return_value = smtp_mock
    smtp_cm.__exit__.return_value = False

    with patch("news_agent.emailer.smtplib.SMTP_SSL", return_value=smtp_cm) as ssl_ctor:
        send_email(to="me@example.com", subject="Test", html_body="<p>hi</p>")

    ssl_ctor.assert_called_once()
    smtp_mock.login.assert_called_once()
    smtp_mock.send_message.assert_called_once()


def test_send_email_auth_error_wrapped(monkeypatch):
    """SMTPAuthenticationError is converted to a friendly EmailError."""
    import smtplib
    from news_agent.config import settings
    monkeypatch.setattr(settings, "smtp_host", "smtp.gmail.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_user", "bot@gmail.com")
    monkeypatch.setattr(settings, "smtp_password", "wrong")
    monkeypatch.setattr(settings, "smtp_use_tls", True)

    smtp_mock = MagicMock()
    smtp_mock.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad creds")
    smtp_cm = MagicMock()
    smtp_cm.__enter__.return_value = smtp_mock
    smtp_cm.__exit__.return_value = False

    with patch("news_agent.emailer.smtplib.SMTP", return_value=smtp_cm):
        with pytest.raises(EmailError, match="App Password"):
            send_email(to="me@example.com", subject="x", html_body="<p>x</p>")
