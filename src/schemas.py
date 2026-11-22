"""Plain data containers shared across components. Kept dependency-free on purpose."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Ticket:
    """The single internal representation every channel is normalised into."""
    ticket_id: str
    channel: str
    subject: str
    body: str
    received_at: str = ""
    customer_id: str = ""
    customer_name: str = ""
    customer_tier: str = "unknown"
    customer_region: str = "unknown"
    language_fluency: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    original_text: str = ""          # untouched subject + body, preserved for audit
    labels: dict | None = None       # present on dev/validation/hidden sets; never used for decisions
    history: dict | None = None

    @property
    def model_text(self) -> str:
        """Text used for classification and retrieval.

        Body only: in the development data the same body appears under several different subjects
        (69 of 216 distinct bodies), so the subject is noise. It is a fallback when the body is empty.
        """
        return self.body or self.subject

    def summary(self, limit: int = 160) -> str:
        text = " ".join(self.model_text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ClassResult:
    intent: str
    confidence: float
    alternatives: list[dict]
    urgency: str
    urgency_confidence: float
    answer_prior: float              # P(ticket is answerable from docs | intent), from past tickets
    fallback: bool = False
    model_version: str = ""
    preferred_docs: list[str] = field(default_factory=list)   # articles that usually answer this intent

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Passage:
    doc_id: str
    chunk_id: str
    title: str
    section: str
    text: str                        # full parent article, used for generation and grounding checks
    score: float

    def to_source(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id, "score": round(float(self.score), 4)}


@dataclass
class RetrievalResult:
    passages: list[Passage]
    top_score: float                 # best score before the relevance threshold was applied
    backend: str
    min_score: float


@dataclass
class RouteDecision:
    action: str                      # auto_respond | escalate
    reason: str
    rule: str                        # which rule fired, for audit and metrics
    confidence: float                # confidence that auto-responding is right (0 for escalations by rule)
    threshold: float
    flags: list[str] = field(default_factory=list)


@dataclass
class Generation:
    mode: str                        # llm | extractive | none
    can_answer: bool
    sentences: list[dict]            # [{"text": str, "cite": [doc_id, ...]}]
    reply_text: str = ""
    unsure: str = ""
    prompt_version: str = ""
    model: str = ""
    error: str = ""

    def cited_doc_ids(self) -> list[str]:
        seen: list[str] = []
        for s in self.sentences:
            for d in s.get("cite", []):
                if d not in seen:
                    seen.append(d)
        return seen


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, dict[str, Any]]
    blocked_by: list[str]

    def results_summary(self) -> dict[str, str]:
        return {name: info["status"] for name, info in self.checks.items()}


@dataclass
class Outcome:
    """What the pipeline returns for one ticket. Always produced, even when a stage fails."""
    ticket_id: str
    channel: str
    disposition: str                 # auto_responded | escalated | blocked_escalated
    reason: str
    intent: str = ""
    intent_confidence: float = 0.0
    urgency: str = ""
    route_confidence: float = 0.0
    threshold: float = 0.0
    rule: str = ""
    sources: list[dict] = field(default_factory=list)
    reply_text: str = ""
    cited_doc_ids: list[str] = field(default_factory=list)
    escalation_brief: dict | None = None
    guardrails: dict = field(default_factory=dict)
    blocked_by: list[str] = field(default_factory=list)
    generation_mode: str = "none"
    latency_s: float = 0.0
    throttle_wait_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    top_retrieval_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
