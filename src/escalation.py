"""Escalation brief: what a human agent receives instead of a raw forwarded ticket.

Interviews (Daniel, Sofia) asked for exactly this: a short summary, what the system thinks it is,
the documentation that seemed relevant, a draft where one is safe to offer, and - above all - what
the system was unsure about. Built deterministically (no model call) so it exists even in an outage.
"""
from __future__ import annotations

from .schemas import ClassResult, Generation, RetrievalResult, RouteDecision, Ticket, ValidationResult


def build_brief(ticket: Ticket, cls: ClassResult, retrieval: RetrievalResult, route: RouteDecision,
                draft: Generation | None = None, validation: ValidationResult | None = None,
                blocked_by: list[str] | None = None) -> dict:
    uncertain: list[str] = []
    if cls.fallback:
        uncertain.append("could not classify the ticket")
    elif cls.confidence < 0.8:
        uncertain.append(f"intent is uncertain ({cls.confidence:.0%}); alternatives: "
                         + ", ".join(f"{a['value']} ({a['confidence']:.0%})" for a in cls.alternatives))
    if not retrieval.passages:
        uncertain.append("no documentation article matched this ticket")
    if route.rule == "low_confidence":
        uncertain.append(f"only {cls.answer_prior:.0%} of past '{cls.intent}' tickets could be answered from documentation")
    if blocked_by:
        uncertain.append("an automatic reply was drafted but blocked by guardrail(s): " + ", ".join(blocked_by))
    if route.rule in {"safety_keyword", "must_escalate_intent"}:
        uncertain.append("category that policy reserves for a human, whatever the model thinks")

    brief = {
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel,
        "customer_tier": ticket.customer_tier,
        "summary": ticket.summary(240),
        "likely_intent": cls.intent,
        "intent_confidence": round(cls.confidence, 3),
        "urgency": cls.urgency,
        "why_escalated": route.reason,
        "rule": route.rule,
        "unsure_about": uncertain or ["nothing specific; routed by threshold"],
        "relevant_articles": [{"doc_id": p.doc_id, "title": p.title, "score": round(p.score, 3)}
                              for p in retrieval.passages],
        "suggested_draft": None,
        "draft_note": None,
    }
    if draft is not None and draft.can_answer and draft.reply_text:
        brief["suggested_draft"] = draft.reply_text
        brief["draft_note"] = ("Draft for the agent to check and edit - NOT sent to the customer"
                               + (f" (blocked by: {', '.join(blocked_by)})" if blocked_by else ""))
    return brief
