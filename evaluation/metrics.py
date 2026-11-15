"""Metric calculations for the evaluation harness (all computed by code, none by hand).

Pure functions over (ticket, outcome) pairs so they are easy to unit test. Anything that cannot honestly
be measured automatically (customer satisfaction, hallucination rate) is reported as "needs human
review" and a review sample is exported, rather than being approximated with a number that looks real.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Callable

import numpy as np

TARGETS = {
    "first_contact_resolution": 0.60, "escalation_rate": 0.30, "classification_precision": 0.85,
    "citation_accuracy": 0.95, "latency_p95_s": 3.0, "cross_group_variation_pts": 5.0,
}


def pct(a: float | None) -> float | None:
    return None if a is None else round(100 * a, 1)


def _mean(xs) -> float | None:
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def percentile(xs, q) -> float | None:
    xs = list(xs)
    return float(np.percentile(xs, q)) if xs else None


def _lab(t: dict) -> dict:
    return t.get("labels") or {}


def classification_report(pairs) -> dict:
    labelled = [(t, o) for t, o in pairs if _lab(t).get("intent")]
    if not labelled:
        return {"available": False}
    truth = [_lab(t)["intent"] for t, _ in labelled]
    pred = [o["intent"] for _, o in labelled]
    classes = sorted(set(truth) | set(pred))
    per = {}
    for c in classes:
        tp = sum(1 for a, b in zip(truth, pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(truth, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(truth, pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        per[c] = {"precision": p, "recall": r, "support": truth.count(c), "predicted": pred.count(c)}
    precs = [v["precision"] for v in per.values() if v["precision"] is not None and v["support"] > 0]
    recs = [v["recall"] for v in per.values() if v["recall"] is not None and v["support"] > 0]
    acc = float(np.mean([a == b for a, b in zip(truth, pred)]))
    urg = [(_lab(t).get("urgency"), o["urgency"]) for t, o in labelled if _lab(t).get("urgency")]
    confusion = Counter(zip(truth, pred))
    return {"available": True, "n": len(labelled), "accuracy": acc, "macro_precision": _mean(precs),
            "macro_recall": _mean(recs), "per_class": per,
            "urgency_accuracy": _mean(a == b for a, b in urg), "confusion": {f"{a}->{b}": n for (a, b), n in confusion.items()}}


def calibration_bands(points: list[tuple[float, bool]], edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        grp = [(c, ok) for c, ok in points if lo <= c < hi]
        if grp:
            stated, observed = float(np.mean([c for c, _ in grp])), float(np.mean([ok for _, ok in grp]))
            rows.append({"band": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(grp), "stated": round(stated, 3),
                         "observed": round(observed, 3), "gap_pts": round(100 * (stated - observed), 1)})
    return rows


def routing_report(pairs) -> dict:
    lab = [(t, o) for t, o in pairs if _lab(t).get("expected_route")]
    if not lab:
        return {"available": False}
    auto = [(t, o) for t, o in lab if o["disposition"] == "auto_responded"]
    exp_auto = [(t, o) for t, o in lab if _lab(t)["expected_route"] == "auto_respond"]
    correct = [(t, o) for t, o in lab if (o["disposition"] == "auto_responded") == (_lab(t)["expected_route"] == "auto_respond")]
    return {
        "available": True, "n": len(lab),
        "route_accuracy": len(correct) / len(lab),
        "auto_precision": _mean(_lab(t)["expected_route"] == "auto_respond" for t, _ in auto),
        "auto_recall": (sum(1 for t, o in exp_auto if o["disposition"] == "auto_responded") / len(exp_auto)) if exp_auto else None,
        "must_not_auto_respond_violations": sum(1 for t, _ in auto if _lab(t).get("must_not_auto_respond")),
        "auto_answered_but_labelled_escalate": sum(1 for t, _ in auto if _lab(t)["expected_route"] != "auto_respond"),
        "escalated_but_labelled_auto": sum(1 for t, o in lab if o["disposition"] != "auto_responded"
                                           and _lab(t)["expected_route"] == "auto_respond"),
    }


def retrieval_report(pairs) -> dict:
    with_docs = [(t, o) for t, o in pairs if _lab(t).get("expected_doc_ids")]
    without = [(t, o) for t, o in pairs if _lab(t) and not _lab(t).get("expected_doc_ids")]
    if not with_docs and not without:
        return {"available": False}
    def ids(o): return [s["doc_id"] for s in o["sources"]]
    return {
        "available": True, "n_with_expected_docs": len(with_docs),
        "hit_rate_any": _mean(any(d in ids(o) for d in _lab(t)["expected_doc_ids"]) for t, o in with_docs),
        "hit_rate_top1": _mean(bool(ids(o)) and ids(o)[0] in _lab(t)["expected_doc_ids"] for t, o in with_docs),
        "empty_result_when_no_expected_docs": _mean(not ids(o) for t, o in without),
        "n_without_expected_docs": len(without),
    }


def citation_report(pairs, resolve: Callable[[str], object]) -> dict:
    sent = [(t, o) for t, o in pairs if o["disposition"] == "auto_responded"]
    if not sent:
        return {"available": False, "n_sent": 0}
    resolved = [bool(o["cited_doc_ids"]) and all(resolve(d) is not None and d in {s["doc_id"] for s in o["sources"]}
                                                 for d in o["cited_doc_ids"]) for _, o in sent]
    exp = [(t, o) for t, o in sent if _lab(t).get("expected_doc_ids")]
    return {"available": True, "n_sent": len(sent),
            "citations_resolve_to_retrieved_corpus_passages": _mean(resolved),
            "cites_an_expected_document": _mean(any(d in _lab(t)["expected_doc_ids"] for d in o["cited_doc_ids"]) for t, o in exp),
            "n_with_expected_docs": len(exp),
            "note": ("Automated proxies. True citation accuracy (does the passage support the sentence?) needs the "
                     "human review sample in review_sample.csv; guardrail 'grounding' enforces a lexical version at runtime.")}


def reference_report(pairs, reference: list[dict] | None) -> dict:
    """Compare sent replies with senior-agent reference answers where ticket ids match."""
    if not reference:
        return {"available": False}
    ref = {r["ticket_id"]: r for r in reference}
    covs, claim_hits, n = [], 0, 0
    for t, o in pairs:
        r = ref.get(t.get("ticket_id"))
        if not r or o["disposition"] != "auto_responded":
            continue
        n += 1
        text = o["reply_text"].lower()
        must = [m.lower() for m in r.get("must_mention", [])]
        if must:
            covs.append(np.mean([m in text or all(w in text for w in re.findall(r"[a-z0-9]+", m) if len(w) > 3) for m in must]))
        for bad in r.get("must_not_claim", []):
            if bad.lower() in text:
                claim_hits += 1
    return {"available": n > 0, "n_matched_sent_replies": n,
            "must_mention_coverage": _mean(covs), "n_with_must_mention": len(covs),
            "must_not_claim_literal_hits": claim_hits}


def latency_report(outcomes: list[dict]) -> dict:
    lat = [o["latency_s"] for o in outcomes]
    auto = [o["latency_s"] for o in outcomes if o["disposition"] == "auto_responded"]
    return {"all_tickets": {"mean_s": _mean(lat), "median_s": percentile(lat, 50), "p95_s": percentile(lat, 95), "max_s": max(lat, default=None)},
            "auto_responded": {"mean_s": _mean(auto), "median_s": percentile(auto, 50), "p95_s": percentile(auto, 95)},
            "client_side_throttle_wait_total_s": round(sum(o.get("throttle_wait_s", 0.0) for o in outcomes), 2),
            "note": "Latency excludes our own free-tier throttle sleeps (reported separately); it includes retrieval, classification, generation and validation."}


def fairness_report(pairs, dims: dict[str, Callable[[dict], str]], min_n: int = 10) -> dict:
    out = {}
    for dim, keyfn in dims.items():
        groups = defaultdict(list)
        for t, o in pairs:
            groups[keyfn(t)].append((t, o))
        rows = {}
        for g, items in sorted(groups.items()):
            lab = [(t, o) for t, o in items if _lab(t)]
            exp = [(t, o) for t, o in lab if _lab(t).get("expected_doc_ids")]
            rows[g] = {
                "n": len(items),
                "too_small_to_judge": len(items) < min_n,
                "auto_response_rate": _mean(o["disposition"] == "auto_responded" for _, o in items),
                "route_accuracy": _mean((o["disposition"] == "auto_responded") == (_lab(t)["expected_route"] == "auto_respond")
                                        for t, o in lab if _lab(t).get("expected_route")),
                "intent_accuracy": _mean(o["intent"] == _lab(t)["intent"] for t, o in lab if _lab(t).get("intent")),
                "retrieval_hit_rate": _mean(any(s["doc_id"] in _lab(t)["expected_doc_ids"] for s in o["sources"]) for t, o in exp),
            }
        metrics = ["auto_response_rate", "route_accuracy", "intent_accuracy", "retrieval_hit_rate"]
        spread = {}
        for m in metrics:
            vals = [r[m] for r in rows.values() if r[m] is not None and r["n"] >= min_n]
            spread[m] = round(100 * (max(vals) - min(vals)), 1) if len(vals) >= 2 else None
        flagged = [m for m, v in spread.items() if v is not None and v > TARGETS["cross_group_variation_pts"]]
        out[dim] = {"groups": rows, "spread_pts": spread,
                    "flagged_over_target": flagged,
                    "note": f"spread = best minus worst group among groups with n >= {min_n}; groups below {min_n} are "
                            f"marked too_small_to_judge and left out of the spread; a spread above "
                            f"{TARGETS['cross_group_variation_pts']:.0f} points is listed in flagged_over_target"}
    return out


def build_report(pairs, resolve, reference, recon: dict, dims, extra: dict) -> dict:
    outcomes = [o for _, o in pairs]
    n = len(outcomes)
    disp = Counter(o["disposition"] for o in outcomes)
    auto, blocked = disp["auto_responded"], disp["blocked_escalated"]
    esc_total = n - auto
    cls = classification_report(pairs)
    rt = routing_report(pairs)
    hist = [t["history"] for t, _ in pairs if t.get("history")]
    baseline = ({"first_contact_resolution": _mean(h["first_contact_resolution"] for h in hist),
                 "escalation_rate": _mean(h["escalated"] for h in hist),
                 "mean_resolution_minutes": _mean(h["resolution_time_minutes"] for h in hist),
                 "mean_csat": _mean(h["csat_rating"] for h in hist if h.get("csat_rating")),
                 "repeat_contact_rate": _mean(h["repeat_contact"] for h in hist)} if hist else None)

    ic = [(o["intent_confidence"], o["intent"] == _lab(t)["intent"]) for t, o in pairs if _lab(t).get("intent") and not o["rule"] == "classification_fallback"]
    gated = {"safety_keyword", "must_escalate_intent", "kill_switch", "classification_fallback", "no_relevant_documentation", "system_error"}
    rc = [(o["route_confidence"], _lab(t)["expected_route"] == "auto_respond") for t, o in pairs
          if _lab(t).get("expected_route") and o["rule"] not in gated]
    guard_fail = Counter()
    for o in outcomes:
        for name, status in (o.get("guardrails") or {}).items():
            if status == "fail":
                guard_fail[name] += 1

    report = {
        "run": extra,
        "volume": {"tickets_processed": n, "answered_automatically": auto,
                   "escalated": disp["escalated"], "blocked_by_guardrails_then_escalated": blocked,
                   "tickets_with_errors": sum(1 for o in outcomes if o["errors"]),
                   "by_channel": dict(Counter(o["channel"] for o in outcomes)),
                   "escalation_reasons": dict(Counter(o["rule"] for o in outcomes if o["disposition"] != "auto_responded")),
                   "generation_modes": dict(Counter(o["generation_mode"] for o in outcomes if o["disposition"] == "auto_responded"))},
        "business": {"first_contact_resolution": auto / n if n else None, "escalation_rate": esc_total / n if n else None,
                     "system_reply_time": latency_report(outcomes),
                     "baseline_from_ticket_history": baseline,
                     "customer_satisfaction": "not measurable automatically - rate review_sample.csv against a rubric (see Evaluation Framework)",
                     "repeat_contacts": "not measurable without live follow-up data",
                     "note": ("FCR here = tickets the system answered without a human, a PROXY for true first-contact resolution: "
                              "we cannot observe whether the customer's problem was actually solved.")},
        "technical": {"classification": cls, "routing": rt, "retrieval": retrieval_report(pairs),
                      "citations": citation_report(pairs, resolve), "reference_comparison": reference_report(pairs, reference),
                      "hallucination_rate": "needs human review of >= 50 replies by two assessors (review_sample.csv)"},
        "calibration": {"intent_confidence": calibration_bands(ic), "routing_confidence": calibration_bands(rc, (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001))},
        "governance": {"decision_log_reconciliation": recon,
                       "availability": (recon.get("tickets_with_final_disposition", 0) / n) if n else None,
                       "guardrail_activations": dict(guard_fail),
                       "tickets_blocked_by_guardrails": blocked,
                       "private_data_detections_blocked": guard_fail.get("pii", 0),
                       "fairness": fairness_report(pairs, dims)},
    }
    return report


def status(value, target, higher_is_better=True):
    if value is None:
        return "n/a"
    return "met" if (value >= target if higher_is_better else value <= target) else "NOT met"
