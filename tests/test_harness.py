"""A9 / A10 - one command, any input file, unattended, metrics report at the end."""
import json
import shutil

from evaluation import harness


def test_harness_runs_on_an_unseen_filename_and_writes_metrics(tmp_path, val_tickets):
    hidden_like = tmp_path / "some_file_i_have_never_seen.json"      # path is an argument, never hardcoded
    tickets = val_tickets[:30] + [{"ticket_id": "BAD-1", "channel": "email", "body": None}, {"nope": True}]
    hidden_like.write_text(json.dumps(tickets))
    out = tmp_path / "out"
    code = harness.main(["--input", str(hidden_like), "--output", str(out), "--no-llm", "--no-ledger"])
    assert code == 0
    report = json.loads((out / "metrics.json").read_text())
    assert report["volume"]["tickets_processed"] == 32                # nothing skipped, including the broken ones
    assert report["governance"]["decision_log_reconciliation"]["ok"]
    assert report["governance"]["private_data_in_sent_replies"] == 0
    assert (out / "metrics.md").exists() and (out / "results.jsonl").exists() and (out / "decisions.db").exists()
    lines = (out / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 32
    for key in ("volume", "business", "technical", "governance", "calibration"):
        assert key in report
    assert report["business"]["system_reply_time"]["all_tickets"]["p95_s"] is not None


def test_harness_reports_missing_input_cleanly(tmp_path):
    assert harness.main(["--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "o")]) == 2


def test_metric_functions_on_known_values():
    from evaluation import metrics as M
    pairs = [({"labels": {"intent": "a"}}, {"intent": "a"}), ({"labels": {"intent": "a"}}, {"intent": "b"}),
             ({"labels": {"intent": "b"}}, {"intent": "b"})]
    rep = M.classification_report(pairs)
    assert rep["accuracy"] == 2 / 3 and rep["per_class"]["b"]["precision"] == 0.5 and rep["per_class"]["a"]["recall"] == 0.5
    bands = M.calibration_bands([(0.9, True), (0.9, True), (0.9, False), (0.1, False)])
    assert bands[-1]["observed"] == round(2 / 3, 3)
