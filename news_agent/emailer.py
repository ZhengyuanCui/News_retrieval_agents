"""SMTP email sender with optional attachments.

Gmail setup: enable 2-Step Verification, then create an App Password at
https://myaccount.google.com/apppasswords and use it as SMTP_PASSWORD.
The regular login password will NOT work.
"""
from __future__ import annotations

import logging
import mimetypes
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from news_agent.config import settings

logger = logging.getLogger(__name__)


class EmailError(RuntimeError):
    """Raised when the SMTP server rejects the message or config is missing."""


def _assert_configured() -> None:
    missing = [
        name for name, val in (
            ("SMTP_HOST", settings.smtp_host),
            ("SMTP_USER", settings.smtp_user),
            ("SMTP_PASSWORD", settings.smtp_password),
        )
        if not val
    ]
    if missing:
        raise EmailError(
            f"SMTP not configured — missing: {', '.join(missing)}. "
            "See .env.example for Gmail setup instructions."
        )


def send_email(
    *,
    to: str | list[str],
    subject: str,
    html_body: str,
    text_body: str | None = None,
    attachments: list[tuple] | None = None,
    sender: str | None = None,
) -> None:
    """Send a multipart HTML email with optional attachments.

    attachments: list of either
      - (filename, data, mime_type), or
      - (filename, data, mime_type, content_id)

    Pass mime_type="" to auto-detect from the filename.  When a content_id
    is supplied the HTML body can reference the attachment via cid:<id>
    (supported by Apple Mail, most webmail clients).
    """
    _assert_configured()

    recipients = [to] if isinstance(to, str) else list(to)
    recipients = [addr.strip() for addr in (
        recipients if isinstance(recipients, list) else [recipients]
    ) for addr in (addr.split(",") if "," in addr else [addr]) if addr.strip()]
    if not recipients:
        raise EmailError("No recipient address provided")

    from_addr = sender or settings.newsletter_email_from or settings.smtp_user

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(text_body or "This email requires an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")

    for attachment in attachments or []:
        if len(attachment) == 4:
            filename, data, mime, content_id = attachment
        else:
            filename, data, mime = attachment
            content_id = None
        if not mime:
            mime, _ = mimetypes.guess_type(filename)
            mime = mime or "application/octet-stream"
        maintype, _, subtype = mime.partition("/")
        kwargs: dict = {
            "maintype": maintype,
            "subtype": subtype or "octet-stream",
            "filename": filename,
        }
        if content_id:
            # RFC 2392: cid: URLs in HTML match the <angle-bracketed> Content-ID.
            kwargs["cid"] = f"<{content_id}>"
        msg.add_attachment(data, **kwargs)

    host = settings.smtp_host
    port = settings.smtp_port
    user = settings.smtp_user
    password = settings.smtp_password
    use_tls = settings.smtp_use_tls

    logger.info("Sending email to %s via %s:%d", recipients, host, port)
    try:
        if port == 465 or not use_tls:
            # Implicit SSL (port 465) or plain — SMTP_SSL
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            # STARTTLS (port 587 default for Gmail)
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
                smtp.login(user, password)
                smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        raise EmailError(
            f"SMTP auth failed ({e.smtp_code}). For Gmail you must use an App "
            f"Password, not your regular password. See .env.example."
        ) from e
    except smtplib.SMTPException as e:
        raise EmailError(f"SMTP send failed: {e}") from e


def load_attachment(path: Path, filename: str | None = None) -> tuple[str, bytes, str]:
    """Read a file from disk into a tuple suitable for send_email(attachments=)."""
    data = path.read_bytes()
    mime, _ = mimetypes.guess_type(path.name)
    return (filename or path.name, data, mime or "application/octet-stream")
