"""Unattended evaluation harness  (acceptance criteria A9 and A10).

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/validation/

* Takes ANY file with the ticket schema as --input (the hidden set is pointed at here), and an output directory.
* Processes every ticket without intervention: one bad ticket is logged and escalated, never skipped or fatal.
* Every ticket yields a sent answer or a logged escalation; every decision is written to the decision log.
* Ends by writing the metrics report (metrics.json + metrics.md) with no manual step.

Outputs in --output:  results.jsonl  decisions.db  metrics.json  metrics.md  classification_per_class.csv
                      confusion_matrix.csv  review_sample.csv  escalation_briefs.jsonl
It also appends one line per run to evaluation/run_ledger.jsonl - use it to state in the report when, and
how many times, you ran the hidden set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT, SYSTEM_VERSION, get_settings
from src.ingest import normalise
from src.pipeline import SupportPipeline
from src.schemas import Outcome

from . import metrics as M

log = logging.getLogger("harness")


def load_tickets(path: Path) -> list:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except ValueError:                                   # JSON Lines
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("tickets", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError("input must be a JSON list of tickets (or {'tickets': [...]})")
    return data


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def _length_bin(t: dict) -> str:
    words = len((t.get("body") or "").split())
    return "short (<=20 words)" if words <= 20 else ("medium (21-30)" if words <= 30 else "long (>30)")


def run(input_path: Path, output_dir: Path, limit: int | None = None, reference_path: Path | None = None,
        write_ledger: bool = True, pipeline: SupportPipeline | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw_tickets = load_tickets(input_path)
    if limit:
        raw_tickets = raw_tickets[:limit]
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()

    db_path = output_dir / "decisions.db"
    for suffix in ("", "-wal", "-shm"):                  # a fresh log per run so reconciliation is exact
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    pipe = pipeline or SupportPipeline(log_path=db_path)
    pipe.guards.register_names([t.get("customer_name", "") for t in raw_tickets if isinstance(t, dict)])

    pairs: list[tuple[dict, dict]] = []
    ids: list[str] = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for i, raw in enumerate(raw_tickets):
            try:
                outcome = pipe.process(raw, i)
            except Exception as exc:                      # belt and braces: process() should never raise
                log.exception("ticket %s crashed the pipeline", i)
                tid = raw.get("ticket_id", f"UNKNOWN-{i:04d}") if isinstance(raw, dict) else f"UNKNOWN-{i:04d}"
                outcome = Outcome(tid, "unknown", "escalated", f"harness caught an unexpected error: {exc}",
                                  rule="system_error", errors=[str(exc)])
                pipe.decisions.record(tid, "routing", "escalate", reason=outcome.reason)
                pipe.decisions.finish_ticket(tid, "escalated")
            od = outcome.to_dict()
            ids.append(outcome.ticket_id)
            t_norm = raw if isinstance(raw, dict) else {}
            pairs.append((t_norm, od))
            fh.write(json.dumps(od, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(raw_tickets):
                print(f"  processed {i + 1}/{len(raw_tickets)}", flush=True)

    # ---------------------------------------------------------------- reporting
    reference = None
    if reference_path and reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    dev_path = get_settings().data_dir / "development_tickets.json"
    dev_bodies = {" ".join(t["body"].lower().split()) for t in json.loads(dev_path.read_text(encoding="utf-8"))} if dev_path.exists() else set()

    recon = pipe.decisions.reconcile(ids)
    dims = {"customer_tier": lambda t: t.get("customer_tier", "unknown"),
            "customer_region": lambda t: t.get("customer_region", "unknown"),
            "language_fluency": lambda t: t.get("language_fluency", "unknown"),
            "channel": lambda t: t.get("channel", "unknown"),
            "ticket_length": _length_bin}
    extra = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(input_path), "input_sha256": input_sha[:16], "tickets": len(raw_tickets),
        "system_version": SYSTEM_VERSION, "git_commit": _git_rev(), "python": platform.python_version(),
        "wall_clock_seconds": round(time.time() - started, 1),
        "components": pipe.describe(),
        "llm_stats": pipe.llm.stats,
    }
    report = M.build_report(pairs, pipe.kb.resolve, reference, recon, dims, extra)

    # independent post-hoc scan of every reply that was actually sent (belt and braces on the guardrail)
    leaked = 0
    for t, o in pairs:
        if o["disposition"] == "auto_responded":
            if pipe.guards.check_pii(normalise(t), o["reply_text"])["status"] == "fail":
                leaked += 1
    report["governance"]["private_data_in_sent_replies"] = leaked

    cls = report["technical"]["classification"]
    if cls.get("available") and dev_bodies:
        seen = [o["intent"] == t["labels"]["intent"] for t, o in pairs if t.get("labels") and " ".join(t["body"].lower().split()) in dev_bodies]
        unseen = [o["intent"] == t["labels"]["intent"] for t, o in pairs if t.get("labels") and " ".join(t["body"].lower().split()) not in dev_bodies]
        cls["seen_vs_unseen_text"] = {
            "accuracy_on_text_also_in_dev": M._mean(seen), "n_seen": len(seen),
            "accuracy_on_text_not_in_dev": M._mean(unseen), "n_unseen": len(unseen),
            "note": "Many tickets repeat the development set's wording verbatim; accuracy on unseen wording is the more honest estimate."}

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (output_dir / "metrics.md").write_text(render_markdown(report), encoding="utf-8")
    _write_csvs(output_dir, report, pairs)
    with (output_dir / "escalation_briefs.jsonl").open("w", encoding="utf-8") as fh:
        for _, o in pairs:
            if o.get("escalation_brief"):
                fh.write(json.dumps(o["escalation_brief"], ensure_ascii=False) + "\n")

    if write_ledger:
        with (ROOT / "evaluation" / "run_ledger.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"timestamp_utc": extra["timestamp_utc"], "input": input_path.name,
                                 "input_sha256": extra["input_sha256"], "tickets": len(raw_tickets),
                                 "git_commit": extra["git_commit"], "reconciled": recon["ok"]}) + "\n")
    return report


def _write_csvs(out: Path, report: dict, pairs) -> None:
    cls = report["technical"]["classification"]
    if cls.get("available"):
        with (out / "classification_per_class.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["intent", "precision", "recall", "support", "predicted"])
            for c, v in cls["per_class"].items():
                w.writerow([c, v["precision"], v["recall"], v["support"], v["predicted"]])
        with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["true->predicted", "count"])
            for k, v in sorted(cls["confusion"].items()):
                w.writerow([k, v])
    sent = [(t, o) for t, o in pairs if o["disposition"] == "auto_responded"]
    random.Random(7).shuffle(sent)                         # fixed seed: the same sample every time
    with (out / "review_sample.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticket_id", "customer_message", "system_reply", "cited_docs",
                    "assessor1_unsupported_claim(Y/N)", "assessor2_unsupported_claim(Y/N)",
                    "assessor1_citation_supports_sentence(Y/N)", "assessor2_citation_supports_sentence(Y/N)",
                    "helpfulness_1to5", "notes"])
        for t, o in sent[:50]:
            w.writerow([o["ticket_id"], t.get("body", ""), o["reply_text"], ";".join(o["cited_doc_ids"]), "", "", "", "", "", ""])


def render_markdown(r: dict) -> str:
    v, b, t, g, run_ = r["volume"], r["business"], r["technical"], r["governance"], r["run"]
    base = b.get("baseline_from_ticket_history") or {}
    lat = b["system_reply_time"]
    cls, rt, ret, cit = t["classification"], t["routing"], t["retrieval"], t["citations"]
    p95 = lat["all_tickets"]["p95_s"]
    fmt = lambda x, d=1: "n/a" if x is None else f"{100 * x:.{d}f}%"
    rows = [
        ("First contact resolution (proxy)", fmt(base.get("first_contact_resolution")) if base else "42%", "60%+",
         fmt(b["first_contact_resolution"]), M.status(b["first_contact_resolution"], 0.60), "tickets answered with no human"),
        ("Escalation rate", fmt(base.get("escalation_rate")) if base else "58%", "<=30%", fmt(b["escalation_rate"]),
         M.status(b["escalation_rate"], 0.30, False), "includes guardrail-blocked tickets"),
        ("Median system reply time", "8-12 hrs", "<5 min", "n/a" if lat["all_tickets"]["median_s"] is None else f"{lat['all_tickets']['median_s']:.2f} s", "met" if (lat['all_tickets']['median_s'] or 0) < 300 else "NOT met", "machine time only; escalated tickets still wait for a person"),
        ("Satisfaction", "3.2/5", "4.0+", "not measured", "n/a", "needs human rating of review_sample.csv"),
        ("Intent precision (macro)", "-", "85%+", fmt(cls.get("macro_precision")) if cls.get("available") else "n/a",
         M.status(cls.get("macro_precision"), 0.85) if cls.get("available") else "n/a", "see per-class CSV"),
        ("Hallucination rate", "-", "<=5%", "not measured", "n/a", "needs two human assessors, >=50 replies"),
        ("Citation accuracy (proxy)", "-", "95%+", fmt(cit.get("citations_resolve_to_retrieved_corpus_passages")) if cit.get("available") else "n/a",
         M.status(cit.get("citations_resolve_to_retrieved_corpus_passages"), 0.95) if cit.get("available") else "n/a", "citations resolve to retrieved corpus passages; support needs humans"),
        ("Latency p95", "-", "<3 s", "n/a" if p95 is None else f"{p95:.2f} s", M.status(p95, 3.0, False), "excludes free-tier throttle sleeps"),
        ("Private data in sent replies", "-", "0", str(g.get("private_data_in_sent_replies")), "met" if g.get("private_data_in_sent_replies") == 0 else "NOT met", ""),
        ("Decision log reconciles", "-", "100%", "yes" if g["decision_log_reconciliation"]["ok"] else "NO", "met" if g["decision_log_reconciliation"]["ok"] else "NOT met", ""),
    ]
    avail = g.get("availability")
    rows.append(("Availability (tickets handled)", "-", "99.5%+", fmt(avail), M.status(avail, 0.995),
                 "tickets ending in a reply or a logged escalation; run with --no-llm for the provider-disconnected figure"))
    fair = g["fairness"]
    worst = max((s for d in fair.values() for s in d["spread_pts"].values() if s is not None), default=None)
    credible = sorted({f"{dim}/{m}" for dim, d in fair.items() for m in d.get("credibly_over_target", [])})
    rows.append(("Cross-group variation (max spread)", "-", "<5 pts", "n/a" if worst is None else f"{worst:.1f} pts",
                 M.status(worst, 5.0, False),
                 "point estimate; 95% bootstrap intervals in the fairness section show which gaps exceed noise" +
                 (f" (interval entirely above 5 pts: {len(credible)} of {sum(1 for d in fair.values() for v in d['spread_pts'].values() if v is not None)} measures)" if worst is not None else "")))
    lines = [f"# Evaluation report", "",
             f"- Input: `{run_['input_file']}` ({run_['tickets']} tickets, sha256 {run_['input_sha256']})",
             f"- Run at {run_['timestamp_utc']} UTC, commit `{run_['git_commit']}`, wall-clock {run_['wall_clock_seconds']} s",
             f"- Components: retrieval `{run_['components']['embedding_and_index']}`, answers `{run_['components']['llm_mode']}`, "
             f"threshold {run_['components']['confidence_threshold']}", "",
             "## Results against targets", "", "| Measure | Baseline | Target | Achieved | Status | Notes |", "|---|---|---|---|---|---|"]
    lines += [f"| {a} | {b_} | {c} | {d} | {e} | {f} |" for a, b_, c, d, e, f in rows]
    lines += ["", "## Volume", "",
              f"Processed {v['tickets_processed']}: answered automatically {v['answered_automatically']}, escalated {v['escalated']}, "
              f"blocked by guardrails then escalated {v['blocked_by_guardrails_then_escalated']}, tickets with internal errors {v['tickets_with_errors']}.",
              "", f"Escalation reasons: {v['escalation_reasons']}", ""]
    if rt.get("available"):
        lines += ["## Routing against labels", "",
                  f"- Route accuracy {fmt(rt['route_accuracy'])}; auto-answer precision {fmt(rt['auto_precision'])}; auto-answer recall {fmt(rt['auto_recall'])}",
                  f"- Must-not-auto-respond violations: **{rt['must_not_auto_respond_violations']}** (must be 0)",
                  f"- Auto-answered but labelled escalate: {rt['auto_answered_but_labelled_escalate']}; escalated but labelled auto: {rt['escalated_but_labelled_auto']}", ""]
    if cls.get("available"):
        sv = cls.get("seen_vs_unseen_text", {})
        lines += ["## Classification", "", f"- Intent accuracy {fmt(cls['accuracy'])} on {cls['n']} labelled tickets; macro precision {fmt(cls['macro_precision'])}, macro recall {fmt(cls['macro_recall'])}; urgency accuracy {fmt(cls['urgency_accuracy'])}"]
        if sv:
            lines.append(f"- Accuracy on wording also present in the development set: {fmt(sv['accuracy_on_text_also_in_dev'])} (n={sv['n_seen']}); on unseen wording: {fmt(sv['accuracy_on_text_not_in_dev'])} (n={sv['n_unseen']})")
        lines.append("")
    if ret.get("available"):
        lines += ["## Retrieval", "", f"- Expected article retrieved (any of top-k): {fmt(ret['hit_rate_any'])}; as top result: {fmt(ret['hit_rate_top1'])} (n={ret['n_with_expected_docs']})",
                  f"- Empty result when no article is expected: {fmt(ret['empty_result_when_no_expected_docs'])} (n={ret['n_without_expected_docs']})", ""]
    lines += ["## Calibration (stated confidence vs observed accuracy)", "", "| Confidence | Band | n | Stated | Observed | Gap (pts) |", "|---|---|---|---|---|---|"]
    for name, key in (("intent", "intent_confidence"), ("routing", "routing_confidence")):
        for row in r["calibration"][key]:
            lines.append(f"| {name} | {row['band']} | {row['n']} | {row['stated']} | {row['observed']} | {row['gap_pts']} |")
    lines += ["", "## Governance", "", f"- Decision log: {g['decision_log_reconciliation']['decisions_logged']} decisions for "
              f"{g['decision_log_reconciliation']['tickets_processed']} tickets; reconciles: **{g['decision_log_reconciliation']['ok']}**",
              f"- Guardrail activations: {g['guardrail_activations'] or 'none'}; private-data blocks {g['private_data_detections_blocked']}; private data in sent replies {g['private_data_in_sent_replies']}", "",
              "### Fairness (spread = best minus worst group, groups with n>=10; [lo-hi] and the spread intervals are 95% bootstrap intervals, in % points)", ""]
    ci = lambda x, m: (lambda c: "" if not c else f" [{c[0]:.0f}-{c[1]:.0f}]")((x.get("ci95_pts") or {}).get(m))
    for dim, d in fair.items():
        lines.append(f"**{dim}**: " + ", ".join(
            f"{k} (n={x['n']}{', too small to judge' if x.get('too_small_to_judge') else ''}, "
            f"auto {fmt(x['auto_response_rate'])}{ci(x, 'auto_response_rate')}, "
            f"route acc {fmt(x['route_accuracy'])}{ci(x, 'route_accuracy')}, "
            f"retrieval hit {fmt(x['retrieval_hit_rate'])})" for k, x in d["groups"].items()))
        lines.append(f"  spread (pts): {d['spread_pts']}")
        for m, c in (d.get("spread_ci95_pts") or {}).items():
            if c is not None and d["spread_pts"].get(m) is not None:
                lines.append(f"  - {m}: {d['spread_pts'][m]} pts, 95% bootstrap interval {c[0]:.1f}-{c[1]:.1f} -> {d['spread_verdict'][m]}")
        if d.get("credibly_over_target"):
            lines.append(f"  **credibly over the 5-point target (interval entirely above it):** {', '.join(d['credibly_over_target'])}")
        elif d.get("flagged_over_target"):
            lines.append(f"  over the 5-point target on the point estimate only (not conclusive): {', '.join(d['flagged_over_target'])}")
        lines.append("")
    lines += ["## Caveats", "",
              "- The figures above should be treated with caution: FCR is a proxy (answered without a human is not the same as solved), "
              "satisfaction and hallucination need human assessment, and many tickets repeat development-set wording.",
              "- In the labelled data, identical ticket text appears with both `auto_respond` and `escalate` labels, so no text-based system can reach 100% route accuracy."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the full evaluation unattended and write the metrics report.")
    ap.add_argument("--input", required=True, type=Path, help="ticket file (JSON list) - any file with the standard schema")
    ap.add_argument("--output", required=True, type=Path, help="output directory (created if missing)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tickets (debugging)")
    ap.add_argument("--reference", type=Path, default=ROOT / "data" / "ground_truth_responses.json",
                    help="optional reference answers for must_mention / must_not_claim comparison")
    ap.add_argument("--no-llm", action="store_true", help="force extractive answers (no model calls)")
    ap.add_argument("--no-ledger", action="store_true", help="do not append to evaluation/run_ledger.jsonl")
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")
    if args.no_llm:
        os.environ["LLM_PROVIDER"] = "none"
    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2
    try:
        report = run(args.input, args.output, args.limit, args.reference, write_ledger=not args.no_ledger)
    except (ValueError, OSError) as exc:
        print(f"cannot run: {exc}", file=sys.stderr)
        return 2
    v = report["volume"]
    print(f"\nDone. {v['tickets_processed']} tickets: {v['answered_automatically']} answered, {v['escalated']} escalated, "
          f"{v['blocked_by_guardrails_then_escalated']} blocked. Decision log reconciles: "
          f"{report['governance']['decision_log_reconciliation']['ok']}")
    print(f"Report: {args.output / 'metrics.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
