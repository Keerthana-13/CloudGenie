"""Ingest: normalise tickets from the four channels into one `Ticket`.

Requirements: preserve original text and channel; never raise on missing fields, odd characters
or empty bodies - degrade and attach a warning instead.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .schemas import Ticket

VALID_CHANNELS = {"email", "chat", "docs_comment", "forum"}
CHANNEL_ALIASES = {
    "live_chat": "chat", "livechat": "chat", "web_chat": "chat",
    "docs": "docs_comment", "doc_comment": "docs_comment", "documentation": "docs_comment", "comment": "docs_comment",
    "community": "forum", "community_forum": "forum",
    "mail": "email", "e-mail": "email",
}
MAX_CHARS = 4000

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"[ \t\r\f\v]+")
_SIGNATURE = re.compile(r"^(--\s*$|sent from my |kind regards,?$|best regards,?$|regards,?$|thanks,?$)", re.I)
_REPLY_HEADER = re.compile(r"^on .{5,80} wrote:\s*$", re.I)


def clean_text(value: Any) -> str:
    """Unicode-normalise, drop control characters, collapse whitespace. Non-strings become ''."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = _CONTROL.sub(" ", value)
    lines = [_WS.sub(" ", ln).strip() for ln in value.split("\n")]
    return "\n".join(ln for ln in lines).strip()


def _strip_quoted_and_signature(text: str) -> str:
    """Email and forum posts often carry quoted history and signatures that would mislead retrieval."""
    kept: list[str] = []
    for ln in text.split("\n"):
        if ln.startswith(">"):
            continue
        if _REPLY_HEADER.match(ln) or _SIGNATURE.match(ln):
            break
        kept.append(ln)
    return "\n".join(kept).strip()


def normalise_channel(value: Any) -> tuple[str, list[str]]:
    channel = clean_text(value).lower().replace(" ", "_")
    channel = CHANNEL_ALIASES.get(channel, channel)
    if channel in VALID_CHANNELS:
        return channel, []
    return "unknown", [f"unrecognised channel {value!r}"]


def normalise(raw: Any, index: int = 0) -> Ticket:
    """Turn one raw record into a Ticket. Never raises."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return Ticket(ticket_id=f"UNKNOWN-{index:04d}", channel="unknown", subject="", body="",
                      warnings=["record is not an object"], original_text=clean_text(raw)[:MAX_CHARS])

    channel, ch_warn = normalise_channel(raw.get("channel"))
    warnings += ch_warn

    ticket_id = clean_text(raw.get("ticket_id")) or f"UNKNOWN-{index:04d}"
    if not raw.get("ticket_id"):
        warnings.append("missing ticket_id; generated one")

    subject = clean_text(raw.get("subject"))
    body_raw = clean_text(raw.get("body"))
    original = (subject + "\n" + body_raw).strip()

    body = body_raw
    if channel in {"email", "forum"}:
        body = _strip_quoted_and_signature(body)
    body = " ".join(body.split())
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS]
        warnings.append("body truncated")
    if not body:
        warnings.append("empty body")

    labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else None
    history = raw.get("history") if isinstance(raw.get("history"), dict) else None

    return Ticket(
        ticket_id=ticket_id,
        channel=channel,
        subject=" ".join(subject.split()),
        body=body,
        received_at=clean_text(raw.get("received_at")),
        customer_id=clean_text(raw.get("customer_id")),
        customer_name=clean_text(raw.get("customer_name")),
        customer_tier=clean_text(raw.get("customer_tier")).lower() or "unknown",
        customer_region=clean_text(raw.get("customer_region")).lower() or "unknown",
        language_fluency=clean_text(raw.get("language_fluency")).lower() or "unknown",
        warnings=warnings,
        original_text=original[:MAX_CHARS],
        labels=labels,
        history=history,
    )
