"""Choose the routing threshold (and calibrate the routing confidence) from data.

The brief's 0.80 is illustrative. Here the number is *measured*:

1. Grouped 5-fold cross-validation over the development tickets (identical ticket texts never straddle
   train/test - see src/classifier.py). For each held-out ticket we run the real classifier, retrieval
   and router.
2. The routing confidence is mapped to the observed rate at which auto-answering was right. After this, a
   stated confidence of 0.8 means roughly 80% of such tickets really were answerable - which is what makes a
   threshold meaningful (governance condition: confidence within 5 points of observed accuracy).
3. The threshold maximises   correct_auto_answers - W * wrong_auto_answers   (W = --wrong-cost, default 2:
   a wrong automatic answer costs twice what a correct one saves, reflecting the head of support's
   "I would rather it said nothing than said something wrong"). W is a business judgement, not a
   statistical one: the script prints how the best threshold moves as W changes. Must-not-auto-respond tickets can never be
   auto-answered, by rule.

Two confidence scores compete, both evaluated out-of-fold on the SAME folds (no score is ever computed by
parameters that saw the ticket's text):
  legacy  - P(intent right) x P(answerable | intent), isotonic-calibrated
  model   - src/answerability.py: a per-ticket logistic regression that adds retrieval evidence
The model is written to config/policy.json (and so used by the router) only if it beats the legacy score on
net benefit; otherwise the legacy score is kept. The script prints both so the choice is visible.

The validation and hidden sets are never used here.

    python -m evaluation.tune_thresholds                  # print the trade-off curve
    python -m evaluation.tune_thresholds --write          # also update config/policy.json
    python -m evaluation.tune_thresholds --wrong-cost 5   # a more cautious business
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src import answerability
from src.classifier import TicketClassifier, body_groups
from src.config import POLICY_PATH, ROOT, get_settings, load_policy
from src.ingest import normalise
from src.knowledge_base import KnowledgeBase
from src.router import raw_route_confidence, route, route_confidence


def collect_oof(tickets: list[dict], policy: dict, embedder_backend: str = "tfidf", n_splits: int = 5):
    settings = get_settings()
    kb = KnowledgeBase.from_files(settings.data_dir / "documentation.json", settings.data_dir / "development_tickets.json",
                                  embedder_backend=embedder_backend, vector_backend="memory")
    groups = body_groups([t["body"] for t in tickets])
    rows = []
    for tr, te in GroupKFold(n_splits=n_splits).split(tickets, groups=groups):
        clf = TicketClassifier().fit([tickets[i] for i in tr])
        for i in te:
            ticket = normalise(tickets[i], i)
            cls = clf.predict(ticket.model_text)
            prefer = set(clf.intent_docs.get(cls.intent, []))
            ret = kb.search(ticket.model_text, top_k=3, min_score=float(policy["retrieval_min_score"]),
                            prefer_docs=prefer, boost=float(policy.get("intent_boost", 0.1)))
            rows.append((ticket, cls, ret, tickets[i]["labels"]))
    return rows


def _reaches_confidence_rule(cls, ret, policy: dict) -> bool:
    return not (cls.intent in set(policy["must_escalate_intents"]) or cls.fallback or not ret.passages)


def fit_calibration(rows, policy: dict) -> dict:
    """Isotonic map raw -> observed, fitted on tickets that actually reach the confidence rule."""
    xs, ys = [], []
    for ticket, cls, ret, lab in rows:
        if not _reaches_confidence_rule(cls, ret, policy):
            continue
        xs.append(raw_route_confidence(cls))
        ys.append(float(lab["expected_route"] == "auto_respond"))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(xs, ys)
    return {"x": [round(float(v), 4) for v in iso.X_thresholds_], "y": [round(float(v), 4) for v in iso.y_thresholds_]}


def fit_cv(rows, policy: dict, n_splits: int = 5):
    """Fit both confidence scores on grouped folds of the out-of-fold rows.

    Returns (answerability model on all rows, per-row policy overrides for the model, per-row overrides for the
    legacy score). Row i is only ever scored by parameters fitted on folds that exclude its text."""
    groups = body_groups([r[0].body for r in rows])
    X = np.array([answerability.features(c, r) for _, c, r, _ in rows])
    y = np.array([lab["expected_route"] == "auto_respond" for _, _, _, lab in rows])
    gate = np.array([_reaches_confidence_rule(c, r, policy) for _, c, r, _ in rows])
    model_ov: list[dict] = [{}] * len(rows)
    legacy_ov: list[dict] = [{}] * len(rows)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        fit_idx = tr[gate[tr]]
        m = {"answerability": answerability.fit(X[fit_idx], y[fit_idx])}
        cal = {"route_calibration": fit_calibration([rows[i] for i in tr], policy)}
        for i in te:
            model_ov[i], legacy_ov[i] = m, cal
    return answerability.fit(X[gate], y[gate]), model_ov, legacy_ov


def evaluate(rows, policy: dict, tau: float, wrong_cost: float, overrides=None) -> dict:
    """overrides: optional per-row policy fragments carrying that row's out-of-fold score parameters."""
    auto = right = must_viol = 0
    for i, (ticket, cls, ret, lab) in enumerate(rows):
        p = {**policy, "confidence_threshold": float(tau), **(overrides[i] if overrides else {})}
        if route(ticket, cls, ret, p).action == "auto_respond":
            auto += 1
            right += lab["expected_route"] == "auto_respond"
            must_viol += bool(lab["must_not_auto_respond"])
    n = len(rows)
    wrong = auto - right
    return {"threshold": round(float(tau), 2), "automation_rate": auto / n,
            "auto_precision": right / auto if auto else 1.0, "wrong_auto_answers": wrong,
            "must_not_violations": must_viol, "auto_answered": auto, "n": n,
            "net_benefit": right - wrong_cost * wrong}


def _confidences(rows, policy: dict, overrides=None) -> list[tuple[float, bool]]:
    """(confidence, auto-answer was right) for every ticket that reaches the confidence rule."""
    pts = []
    for i, (t, c, r, lab) in enumerate(rows):
        if _reaches_confidence_rule(c, r, policy):
            p = {**policy, **(overrides[i] if overrides else {})}
            pts.append((route_confidence(c, r, p), lab["expected_route"] == "auto_respond"))
    return pts


def calibration_table(rows, policy: dict, bins=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01), overrides=None) -> list[dict]:
    pts = _confidences(rows, policy, overrides)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        grp = [(s, ok) for s, ok in pts if lo <= s < hi]
        if grp:
            out.append({"band": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(grp),
                        "stated": round(float(np.mean([s for s, _ in grp])), 3),
                        "observed": round(float(np.mean([ok for _, ok in grp])), 3)})
    return out


def _auc(pts) -> float:
    y = [ok for _, ok in pts]
    return float(roc_auc_score(y, [s for s, _ in pts])) if len(set(y)) == 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrong-cost", type=float, default=2.0,
                    help="cost of one wrong auto-answer, in units of the benefit of one correct auto-answer")
    ap.add_argument("--write", action="store_true", help="write the result into config/policy.json")
    ap.add_argument("--embedding-backend", default="tfidf")
    args = ap.parse_args()

    policy = load_policy()
    policy.pop("route_calibration", None)
    policy.pop("answerability", None)
    tickets = json.loads((get_settings().data_dir / "development_tickets.json").read_text(encoding="utf-8"))
    print(f"Collecting grouped out-of-fold predictions for {len(tickets)} development tickets ...")

    # retrieval relevance floor: reject only clearly irrelevant matches (keep ~98% of answerable tickets)
    probe = collect_oof(tickets, {**policy, "retrieval_min_score": 0.0}, args.embedding_backend)
    ans_scores = np.array([r[2].top_score for r in probe if r[3].get("expected_doc_ids")])
    rho = float(max(0.05, round(np.percentile(ans_scores, 2) * 0.9, 3)))
    policy["retrieval_min_score"] = rho
    rows = [(t, c, type(r)([p for p in r.passages if p.score >= rho], r.top_score, r.backend, rho), lab)
            for t, c, r, lab in probe]

    # intent / urgency quality on text-grouped out-of-fold predictions (the honest figure; see docs/architecture.md)
    from . import metrics as M
    cls_rep = M.classification_report([({"labels": lab}, {"intent": c.intent, "urgency": c.urgency}) for _, c, _, lab in rows])
    urg_majority = max(np.bincount([["high", "medium", "low"].index(lab["urgency"]) for *_, lab in rows])) / len(rows)
    print(f"\nGrouped out-of-fold classification: intent accuracy {cls_rep['accuracy']:.1%}, macro precision "
          f"{cls_rep['macro_precision']:.1%}, macro recall {cls_rep['macro_recall']:.1%}; urgency accuracy "
          f"{cls_rep['urgency_accuracy']:.1%} (majority-class baseline {urg_majority:.1%})")

    taus = np.arange(0.40, 0.951, 0.01)
    pick = lambda grid: max((g for g in grid if g["must_not_violations"] == 0 and g["auto_answered"] > 0),
                            key=lambda g: (g["net_benefit"], -g["threshold"]))

    ans_model, model_ov, legacy_ov = fit_cv(rows, policy)
    full_calibration = fit_calibration(rows, policy)
    grids = {"legacy": [evaluate(rows, policy, t, args.wrong_cost, legacy_ov) for t in taus],
             "model": [evaluate(rows, policy, t, args.wrong_cost, model_ov) for t in taus]}
    ovs = {"legacy": legacy_ov, "model": model_ov}
    best = {k: pick(g) for k, g in grids.items()}
    use_model = best["model"]["net_benefit"] > best["legacy"]["net_benefit"]
    winner = "model" if use_model else "legacy"
    grid, chosen = grids[winner], best[winner]
    cand = [g for g in grid if g["must_not_violations"] == 0 and g["auto_answered"] > 0]
    policy["confidence_threshold"] = chosen["threshold"]
    table = calibration_table(rows, policy, overrides=ovs[winner])

    print("\nConfidence scores compared (grouped out-of-fold, same folds; tickets that reach the confidence rule):")
    print(f"  {'score':<24}{'AUC':>7}   best threshold  automation  precision  net benefit (W={args.wrong_cost:g})")
    for key, label in (("legacy", "legacy: intent x prior"), ("model", "answerability model")):
        b = best[key]
        print(f"  {label:<24}{_auc(_confidences(rows, policy, ovs[key])):>7.3f}"
              f"   {b['threshold']:>14.2f}  {b['automation_rate']:>9.1%}  {b['auto_precision']:>9.1%}  {b['net_benefit']:>10.1f}")
    print("  -> " + ("answerability model beats the legacy score and will be used" if use_model
                     else "answerability model does NOT beat the legacy score; legacy score kept"))
    print("  model weights (standardised): " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(ans_model["features"], ans_model["coef"])))

    print(f"\nretrieval_min_score = {rho}  (keeps ~98% of answerable dev tickets)")
    print(f"wrong-answer cost W = {args.wrong_cost:g}  (net benefit = correct - W x wrong); table below is the {winner} score\n")
    print("threshold  automation  auto-precision  wrong auto-answers  net benefit")
    for g in sorted({id(x): x for x in grid[::5] + [chosen]}.values(), key=lambda g: g["threshold"]):
        mark = "  <-- chosen" if g is chosen else ""
        print(f"  {g['threshold']:.2f}      {g['automation_rate']:6.1%}      {g['auto_precision']:6.1%}"
              f"          {g['wrong_auto_answers']:3d}               {g['net_benefit']:7.1f}{mark}")
    print("\nHow the best threshold moves with the cost of a wrong answer (W):")
    for w in (1, 2, 3, 5):
        top = max(({**g, "nb": g["auto_answered"] - (1 + w) * g["wrong_auto_answers"]} for g in cand),
                  key=lambda g: (g["nb"], -g["threshold"]))
        print(f"  W={w}: threshold {top['threshold']:.2f} -> automation {top['automation_rate']:.1%}, "
              f"precision {top['auto_precision']:.1%}")
    print("\nCalibration of the routing confidence (grouped out-of-fold):")
    for r in table:
        print(f"  band {r['band']}: n={r['n']:3d}  stated {r['stated']:.2f}  observed {r['observed']:.2f}")

    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "threshold_frontier.json").write_text(json.dumps(
        {"wrong_cost": args.wrong_cost, "retrieval_min_score": rho, "confidence_score_used": winner, "chosen": chosen,
         "frontier": grid, "calibration": table, "legacy_chosen": best["legacy"], "model_chosen": best["model"],
         "answerability_model": ans_model,
         "grouped_oof_classification": {"intent_accuracy": cls_rep["accuracy"], "macro_precision": cls_rep["macro_precision"],
                                        "macro_recall": cls_rep["macro_recall"], "urgency_accuracy": cls_rep["urgency_accuracy"],
                                        "urgency_majority_baseline": float(urg_majority), "backend": args.embedding_backend},
         "method": "grouped 5-fold CV on development tickets; real classifier, retrieval, router"},
        indent=2))

    if args.write:
        pol = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        pol["confidence_threshold"] = chosen["threshold"]
        pol["retrieval_min_score"] = rho
        pol["route_calibration"] = full_calibration
        if use_model:
            pol["answerability"] = ans_model
        else:
            pol.pop("answerability", None)           # never leave a stale model behind
        pol["tuning"] = {
            "method": "grouped 5-fold CV on development_tickets.json (identical texts never split across folds)",
            "confidence_score": "answerability logistic regression" if use_model else "intent x per-intent prior (isotonic)",
            "wrong_answer_cost": args.wrong_cost,
            "oof_automation_rate": round(chosen["automation_rate"], 4),
            "oof_auto_precision": round(chosen["auto_precision"], 4),
            "tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_backend_used": args.embedding_backend,
        }
        POLICY_PATH.write_text(json.dumps(pol, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote thresholds and calibration to {POLICY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
