from src.ingest import normalise


def test_all_four_channels_normalise_to_one_shape(dev_tickets):
    seen = set()
    for ch in ("email", "chat", "docs_comment", "forum"):
        raw = next(t for t in dev_tickets if t["channel"] == ch)
        t = normalise(raw)
        assert t.channel == ch and t.body and t.ticket_id == raw["ticket_id"]
        assert t.original_text            # original preserved
        seen.add(ch)
    assert len(seen) == 4


def test_chat_ticket_with_empty_subject_is_fine():
    t = normalise({"ticket_id": "X", "channel": "chat", "subject": "", "body": "my deployment keeps rolling back"})
    assert t.subject == "" and t.model_text.startswith("my deployment")


def test_missing_fields_do_not_raise():
    t = normalise({})
    assert t.ticket_id.startswith("UNKNOWN") and t.channel == "unknown" and "empty body" in t.warnings


def test_non_dict_input_does_not_raise():
    for bad in (None, 5, "text", ["a"]):
        assert normalise(bad, 3).ticket_id == "UNKNOWN-0003"


def test_odd_characters_and_control_codes():
    t = normalise({"ticket_id": "X", "channel": "email", "body": "pa\x00ss​word   reset\x07 \U0001F600 ünïcode"})
    assert "\x00" not in t.body and "\x07" not in t.body and "ünïcode" in t.body


def test_email_quotes_and_signature_stripped():
    body = "How do I rotate a key?\n> old quoted text\n--\nJohn\nCEO"
    t = normalise({"ticket_id": "X", "channel": "email", "body": body})
    assert "quoted" not in t.body and "CEO" not in t.body and "rotate a key" in t.body


def test_channel_aliases():
    assert normalise({"channel": "Live Chat", "body": "x"}).channel == "chat"
    assert normalise({"channel": "carrier pigeon", "body": "x"}).channel == "unknown"


def test_very_long_body_truncated():
    assert len(normalise({"channel": "email", "body": "a " * 10000}).body) <= 4000
