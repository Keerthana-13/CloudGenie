"""Route: decide between answering automatically and escalating, and say why.

A pure function of its inputs: the same ticket, classification, retrieval and policy always give the
same decision (A5). Rules are evaluated in priority order and the first that fires wins, so the
`rule` field is also a compact audit trail and a metrics dimension.

Priority:  kill switch > safety keywords > must-escalate intents > classification fallback
           > no relevant documentation > confidence below threshold > auto-respond.
"""
from __future__ import annotations

import re

import numpy as np

from . import answerability
from .schemas import ClassResult, RetrievalResult, RouteDecision, Ticket


def raw_route_confidence(cls: ClassResult) -> float:
    """P(intent right) x P(answerable | intent). Uncalibrated."""
    return 0.0 if cls.fallback else float(cls.confidence * cls.answer_prior)


def calibrated_confidence(raw: float, policy: dict) -> float:
    """Map the raw score to the observed rate at which auto-answering was right (isotonic map fitted by
    evaluation/tune_thresholds.py on grouped out-of-fold predictions). Without a map, the raw value is used."""
    cal = policy.get("route_calibration")
    if not cal:
        return round(raw, 4)
    return round(float(np.interp(raw, cal["x"], cal["y"])), 4)


def route_confidence(cls: ClassResult, retrieval: RetrievalResult, policy: dict) -> float:
    """Confidence that *auto-answering is right*. Uses the per-ticket answerability model when the policy
    carries one (fitted by evaluation/tune_thresholds.py); otherwise the older intent x prior score."""
    model = policy.get("answerability")
    if model:
        return answerability.confidence(model, cls, retrieval)
    return calibrated_confidence(raw_route_confidence(cls), policy)


def route(ticket: Ticket, cls: ClassResult, retrieval: RetrievalResult, policy: dict,
          kill_switch: bool = False) -> RouteDecision:
    threshold = float(policy["confidence_threshold"])
    must = set(policy["must_escalate_intents"])
    flags: list[str] = []
    confidence = route_confidence(cls, retrieval, policy)
    modelled = bool(policy.get("answerability"))

    def esc(rule: str, reason: str) -> RouteDecision:
        return RouteDecision("escalate", reason, rule, confidence, threshold, flags)

    if kill_switch:
        return esc("kill_switch", "Automatic answering is switched off; every ticket goes to a human.")

    text = ticket.model_text.lower()
    hits = [k for k in policy.get("security_keywords", []) if re.search(k, text)]
    if hits:
        flags.append("security_keyword")
        return esc("safety_keyword",
                   f"Ticket mentions a possible security problem ('{hits[0]}'); security matters always go to a person.")

    if cls.intent in must:
        return esc("must_escalate_intent",
                   f"Classified as '{cls.intent}', a category that is never answered automatically "
                   f"(security, compliance, feature requests and unclear requests).")

    if cls.fallback:
        return esc("classification_fallback", "The ticket could not be classified, so it was not answered automatically.")

    if not retrieval.passages:
        return esc("no_relevant_documentation",
                   f"No documentation article was relevant enough (best match {retrieval.top_score:.2f}, "
                   f"minimum {retrieval.min_score:.2f}); there is nothing to ground an answer in.")

    if confidence < threshold:
        if modelled:
            return esc("low_confidence",
                       f"Confidence {confidence:.2f} is below the {threshold:.2f} threshold: classified as "
                       f"'{cls.intent}' ({cls.confidence:.0%} sure); {cls.answer_prior:.0%} of past '{cls.intent}' tickets "
                       f"were answerable, and the best article ({retrieval.passages[0].doc_id}, score "
                       f"{retrieval.passages[0].score:.2f}) is not a convincing enough match.")
        return esc("low_confidence",
                   f"Confidence {confidence:.2f} is below the {threshold:.2f} threshold: classified as "
                   f"'{cls.intent}' ({cls.confidence:.0%} sure), and only {cls.answer_prior:.0%} of past '{cls.intent}' "
                   f"tickets could be answered from the documentation.")

    return RouteDecision(
        "auto_respond",
        f"Classified as '{cls.intent}' ({cls.confidence:.0%} sure); {cls.answer_prior:.0%} of past '{cls.intent}' tickets were "
        f"answerable from documentation; best matching article {retrieval.passages[0].doc_id} "
        f"(score {retrieval.passages[0].score:.2f}). Confidence {confidence:.2f} >= threshold {threshold:.2f}.",
        "confident_and_grounded", confidence, threshold, flags)
