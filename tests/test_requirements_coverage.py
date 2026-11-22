"""Tests that tie the build back to the PRD: FR-05, FR-08, FR-13, FR-14, NFR-04 and the requirement ids in the log."""
import csv
import json
import re

from evaluation import harness
from evaluation import metrics as M
from src.generate import FOOTER
from src.prompts import load_prompt

from .conftest import FakeLLM

BRIEF_KEYS = ("summary", "likely_intent", "intent_confidence", "relevant_articles", "unsure_about", "why_escalated")


def test_every_escalation_carries_a_complete_brief(pipe, val_tickets):
    escalated = [o for o in (pipe.process(t, i) for i, t in enumerate(val_tickets)) if o.disposition != "auto_responded"]
    assert escalated, "expected some escalations in the validation set"
    for o in escalated:
        brief = o.escalation_brief
        assert brief and all(k in brief for k in BRIEF_KEYS), o.ticket_id
        assert brief["summary"] and brief["unsure_about"] and brief["why_escalated"]
        if brief.get("suggested_draft"):                                  # a draft is only ever offered, never sent
            assert "NOT sent" in brief["draft_note"] and o.reply_text == ""


def test_automatic_reply_discloses_automation_and_lists_sources(pipe, auto_ticket):
    o = pipe.process(auto_ticket)
    assert o.disposition == "auto_responded"
    assert FOOTER in o.reply_text and "reply to this message" in o.reply_text      # automated, and how to reach a person
    sources_block = o.reply_text.split("Sources:")[1].split(FOOTER)[0]
    listed = set(re.findall(r"DOC-[A-Z]+-\d+", sources_block))
    assert listed == set(o.cited_doc_ids) and listed


def _pairs(groups: dict[str, tuple[int, int]]):
    """groups: name -> (n tickets, n answered automatically). Every ticket is labelled auto_respond."""
    pairs = []
    for name, (n, auto) in groups.items():
        for i in range(n):
            t = {"customer_tier": name, "labels": {"intent": "a", "expected_route": "auto_respond", "expected_doc_ids": []}}
            o = {"disposition": "auto_responded" if i < auto else "escalated", "intent": "a", "sources": []}
            pairs.append((t, o))
    return pairs


def test_metrics_report_contains_fairness_slices():
    pairs = _pairs({"big_good": (20, 20), "big_bad": (20, 10), "tiny": (4, 0)})
    rep = M.fairness_report(pairs, {"customer_tier": lambda t: t["customer_tier"]})["customer_tier"]
    assert {g: r["n"] for g, r in rep["groups"].items()} == {"big_good": 20, "big_bad": 20, "tiny": 4}
    assert rep["groups"]["tiny"]["too_small_to_judge"] and not rep["groups"]["big_good"]["too_small_to_judge"]
    assert rep["spread_pts"]["auto_response_rate"] == 50.0            # tiny group is left out of the spread
    assert "auto_response_rate" in rep["flagged_over_target"]         # a gap above 5 points is flagged
    assert "intent_accuracy" not in rep["flagged_over_target"]


def test_fairness_slices_are_written_by_the_harness(tmp_path, val_tickets):
    out = tmp_path / "o"
    assert harness.main(["--input", _write(tmp_path, val_tickets), "--output", str(out), "--no-llm", "--no-ledger"]) == 0
    report = json.loads((out / "metrics.json").read_text())
    for dim in ("customer_tier", "customer_region", "language_fluency", "channel"):
        assert report["governance"]["fairness"][dim]["groups"]
    text = (out / "metrics.md").read_text()
    assert "Availability (tickets handled)" in text and "Cross-group variation" in text
    assert report["governance"]["availability"] == 1.0


def _write(tmp_path, tickets):
    p = tmp_path / "tickets_for_test.json"
    p.write_text(json.dumps(tickets))
    return str(p)


def test_review_sample_is_exported_for_human_assessors(tmp_path, val_tickets):
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert harness.main(["--input", _write(tmp_path, val_tickets), "--output", str(out), "--no-llm", "--no-ledger"]) == 0
        with (out / "review_sample.csv").open(newline="", encoding="utf-8") as fh:
            outs.append(list(csv.DictReader(fh)))
    rows = outs[0]
    assert 0 < len(rows) <= 50
    assert all(r["system_reply"] and r["cited_docs"] for r in rows)
    blank = [c for c in rows[0] if c.startswith("assessor") or c in ("helpfulness_1to5", "notes")]
    assert blank and all(r[c] == "" for r in rows for c in blank)     # left empty for the two human assessors
    assert [r["ticket_id"] for r in outs[0]] == [r["ticket_id"] for r in outs[1]]     # same sample for the same input


def test_customer_name_and_id_never_reach_the_model(make_pipe, auto_ticket):
    seen = []

    class Spy(FakeLLM):
        def chat(self, messages, temperature=0.0, max_tokens=700):
            seen.append(json.dumps(messages))
            return super().chat(messages, temperature, max_tokens)

    p = make_pipe("good")
    p.llm = Spy("good")
    p.generator.llm = p.llm
    raw = dict(auto_ticket, ticket_id="PRIV-1", customer_name="Zed Quillfeather", customer_id="CUST-987654")
    o = p.process(raw)
    assert o.disposition == "auto_responded" and seen
    blob = " ".join(seen)
    assert "Quillfeather" not in blob and "CUST-987654" not in blob


def test_decision_log_stamps_the_prd_requirement_ids(pipe, auto_ticket):
    o = pipe.process(dict(auto_ticket, ticket_id="REQ-1"))
    stages = {d["stage"]: d["requirement_ids"] for d in pipe.decisions.decisions_for(o.ticket_id)}
    assert stages["classification"] == ["FR-02"] and stages["retrieval"] == ["FR-03"]
    assert stages["routing"] == ["FR-04", "FR-05"] and stages["validation"] == ["FR-07"]


def test_all_three_prompt_files_load_and_match_the_register():
    for filename, pid in (("PR-01_specify_requirement.v1.md", "PR-01"), ("PR-02_generate_answer.v1.md", "PR-02"),
                          ("PR-03_judge_reply_support.v1.md", "PR-03")):
        p = load_prompt(filename)
        assert p.id == pid and p.version == "1.0" and p.system and p.user_template
