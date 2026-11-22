"""The pipeline: Ingest -> Classify -> Retrieve -> Route -> Generate -> Validate, with a decision log.

Design rules enforced here:
  * `process()` never raises. Every stage is wrapped; a failure degrades (classification fallback,
    empty retrieval, extractive answer) or fails *closed* (validation error => block and escalate).
  * Every ticket ends in exactly one of: auto_responded, escalated, blocked_escalated - never dropped.
  * Every stage writes a decision-log record, including for escalations and blocks.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from . import telemetry
from .classifier import TicketClassifier
from .config import Settings, get_settings, kill_switch_engaged, load_policy
from .decision_log import DecisionLog
from .escalation import build_brief
from .generate import Generator
from .ingest import normalise
from .knowledge_base import KnowledgeBase
from .llm import LLMClient
from .router import route as route_ticket
from .schemas import (ClassResult, Generation, Outcome, RetrievalResult, RouteDecision, Ticket,
                      ValidationResult)
from .validate import Guardrails

log = logging.getLogger(__name__)


def _load_names(*paths: Path) -> list[str]:
    names: list[str] = []
    for p in paths:
        try:
            if Path(p).exists():
                names += [t.get("customer_name", "") for t in json.loads(Path(p).read_text(encoding="utf-8"))]
        except Exception:
            pass
    return names


class SupportPipeline:
    def __init__(self, settings: Settings | None = None, policy: dict | None = None,
                 kb: KnowledgeBase | None = None, classifier: TicketClassifier | None = None,
                 llm: LLMClient | None = None, decision_log: DecisionLog | None = None,
                 run_id: str | None = None, log_path: Path | None = None):
        self.settings = settings or get_settings()
        self.policy = policy or load_policy()
        data = self.settings.data_dir
        storage = self.settings.storage_dir
        dev = data / "development_tickets.json"
        self.kb = kb or KnowledgeBase.from_files(
            data / "documentation.json", dev, embedder_backend=self.settings.embedding_backend,
            embedding_model=self.settings.embedding_model, chroma_path=storage / "chroma")
        self.classifier = classifier or TicketClassifier.load_or_train(dev, storage)
        self.llm = llm if llm is not None else LLMClient(self.settings, cache_dir=storage / "llm_cache")
        self.generator = Generator(self.llm)
        self.guards = Guardrails(self.policy, _load_names(dev, data / "validation_tickets.json"))
        self.decisions = decision_log or DecisionLog(log_path or storage / "decisions.db", run_id)
        self.req = self.policy.get("requirement_ids", {})

    # ------------------------------------------------------------------ helpers
    def describe(self) -> dict:
        return {
            "embedding_and_index": self.kb.backend,
            "classifier": self.classifier.meta,
            "llm_mode": (f"{self.settings.llm_provider}:{self.settings.llm_model}" if self.llm.available
                         else "none (extractive answers)"),
            "confidence_threshold": self.policy["confidence_threshold"],
            "confidence_model": "answerability-lr" if self.policy.get("answerability") else "intent x prior (legacy)",
            "retrieval_min_score": self.policy["retrieval_min_score"],
            "must_escalate_intents": self.policy["must_escalate_intents"],
        }

    def _log(self, ticket: Ticket, stage: str, action: str, **fields) -> None:
        fields.setdefault("input_summary", ticket.summary())
        fields.setdefault("requirement_ids", self.req.get(stage, []))
        self.decisions.record(ticket.ticket_id, stage, action, **fields)

    # -------------------------------------------------------------------- main
    def process(self, raw: dict, index: int = 0) -> Outcome:
        t0 = time.perf_counter()
        wait0 = self.llm.stats["throttle_wait_s"]
        errors: list[str] = []
        ticket = normalise(raw, index)
        outcome = Outcome(ticket_id=ticket.ticket_id, channel=ticket.channel, disposition="escalated", reason="")
        cls = ClassResult("unclear_request", 0.0, [], "medium", 0.0, 0.0, True, "n/a")
        retrieval = RetrievalResult([], 0.0, self.kb.backend, float(self.policy["retrieval_min_score"]))
        decision = RouteDecision("escalate", "system error before routing", "system_error", 0.0,
                                 float(self.policy["confidence_threshold"]))
        draft: Generation | None = None
        validation: ValidationResult | None = None
        try:
            # ---- classify
            try:
                cls = self.classifier.predict(ticket.model_text)
            except Exception as exc:                                  # pragma: no cover - predict() already guards
                errors.append(f"classify: {exc}")
            self._log(ticket, "classification", "classify",
                      model={"name": "tfidf-logreg", "version": cls.model_version},
                      prediction={"value": cls.intent, "confidence": round(cls.confidence, 4)},
                      alternatives=cls.alternatives,
                      reason=("fallback: could not classify" if cls.fallback else
                              f"intent '{cls.intent}', urgency '{cls.urgency}'"),
                      threshold_applied=None, guardrail_results={}, prompt_version=None, sources_used=[])

            # ---- retrieve
            try:
                prefer = set(self.classifier.intent_docs.get(cls.intent, [])) if not cls.fallback else set()
                retrieval = self.kb.search(ticket.model_text, top_k=self.settings.retrieval_top_k,
                                           min_score=float(self.policy["retrieval_min_score"]),
                                           prefer_docs=prefer, boost=float(self.policy.get("intent_boost", 0.1)))
            except Exception as exc:
                errors.append(f"retrieve: {exc}")
            self._log(ticket, "retrieval", "retrieve",
                      model={"name": self.kb.embedder.name, "version": self.kb.backend},
                      prediction={"value": retrieval.passages[0].doc_id if retrieval.passages else None,
                                  "confidence": round(retrieval.top_score, 4)},
                      alternatives=[], sources_used=[p.to_source() for p in retrieval.passages],
                      threshold_applied=retrieval.min_score,
                      reason=(f"{len(retrieval.passages)} article(s) above relevance threshold"
                              if retrieval.passages else f"nothing relevant (best {retrieval.top_score:.2f})"),
                      guardrail_results={}, prompt_version=None)

            # ---- route
            try:
                decision = route_ticket(ticket, cls, retrieval, self.policy, kill_switch_engaged(self.settings))
            except Exception as exc:
                errors.append(f"route: {exc}")
                decision = RouteDecision("escalate", f"routing failed ({exc}); escalated to be safe", "system_error",
                                         0.0, float(self.policy["confidence_threshold"]))
            self._log(ticket, "routing", decision.action,
                      model={"name": "rule-router", "version": "1.0"},
                      prediction={"value": decision.action, "confidence": decision.confidence},
                      alternatives=[], sources_used=[p.to_source() for p in retrieval.passages],
                      threshold_applied=decision.threshold, reason=decision.reason,
                      guardrail_results={}, prompt_version=None, rule=decision.rule, flags=decision.flags)

            blocked_by: list[str] = []
            if decision.action == "auto_respond":
                # ---- pre-generation injection scan, then generate, then validate (fail closed)
                scan = self.guards.input_scan(ticket)
                if not scan.passed:
                    blocked_by = scan.blocked_by
                    validation = scan
                    decision = RouteDecision("escalate", "Reply blocked by guardrail(s): instruction_integrity: "
                                             + scan.checks["instruction_integrity"]["detail"] + " (input never reached the model)",
                                             "guardrail_block", decision.confidence, decision.threshold)
                    self._log(ticket, "validation", "block", model={"name": "guardrails", "version": "1.0"},
                              prediction={"value": "block", "confidence": 1.0}, alternatives=[], sources_used=[],
                              threshold_applied=None, guardrail_results=scan.results_summary(),
                              reason=scan.checks["instruction_integrity"]["detail"] + " - input never reached the model",
                              prompt_version=None)
                else:
                    gen = Generation("none", False, [], error="generation not run")
                    try:
                        gen = self.generator.generate(ticket, retrieval.passages)
                    except Exception as exc:
                        errors.append(f"generate: {exc}")
                        gen = self.generator.extractive_draft(retrieval.passages)
                    outcome.generation_mode = gen.mode
                    self._log(ticket, "generation", "generate",
                              model={"name": gen.model or "none", "version": gen.mode},
                              prediction={"value": "answer" if gen.can_answer else "cannot_answer", "confidence": None},
                              alternatives=[], sources_used=[p.to_source() for p in retrieval.passages],
                              threshold_applied=None, prompt_version=gen.prompt_version,
                              reason=(f"mode={gen.mode}" + (f"; {gen.error}" if gen.error else "")
                                      + (f"; unsure: {gen.unsure}" if gen.unsure else "")),
                              guardrail_results={})
                    if not gen.can_answer:
                        decision = RouteDecision("escalate", f"Model declined to answer from the sources ({gen.unsure or 'insufficient sources'}).",
                                                 "generation_declined", decision.confidence, decision.threshold)
                    else:
                        try:
                            validation = self.guards.validate(ticket, gen, retrieval.passages, decision,
                                                              self.kb.resolve, self.generator.prompt.system)
                        except Exception as exc:                       # fail closed
                            errors.append(f"validate: {exc}")
                            validation = ValidationResult(False, {"validator_error": {"status": "fail", "detail": str(exc)}},
                                                          ["validator_error"])
                        failed = [f"{n}: {validation.checks[n]['detail']}" for n in validation.blocked_by]
                        self._log(ticket, "validation", "auto_respond" if validation.passed else "block",
                                  model={"name": "guardrails", "version": "1.0"},
                                  prediction={"value": "pass" if validation.passed else "block", "confidence": 1.0},
                                  alternatives=[], sources_used=[{"doc_id": d} for d in gen.cited_doc_ids()],
                                  threshold_applied=None, guardrail_results=validation.results_summary(),
                                  reason=("all guardrails passed" if validation.passed else "; ".join(failed)),
                                  prompt_version=gen.prompt_version)
                        if validation.passed:
                            outcome.disposition = "auto_responded"
                            outcome.reply_text = gen.reply_text
                            outcome.cited_doc_ids = gen.cited_doc_ids()
                            outcome.reason = decision.reason
                        else:
                            blocked_by = validation.blocked_by
                            draft = gen
                            decision = RouteDecision("escalate", "Reply blocked by guardrail(s): " + "; ".join(failed),
                                                     "guardrail_block", decision.confidence, decision.threshold)

            if outcome.disposition != "auto_responded":
                outcome.disposition = "blocked_escalated" if blocked_by else "escalated"
                outcome.reason = decision.reason
                if draft is None and decision.rule not in {"safety_keyword", "must_escalate_intent", "kill_switch"} \
                        and retrieval.passages and not blocked_by:
                    draft = self.generator.extractive_draft(retrieval.passages)
                if blocked_by and "instruction_integrity" in blocked_by:
                    draft = None
                outcome.escalation_brief = build_brief(ticket, cls, retrieval, decision, draft, validation, blocked_by)
        except Exception as exc:     # last line of defence: nothing may escape and nothing may be dropped
            errors.append(f"pipeline: {exc}")
            log.exception("unexpected pipeline failure for %s", ticket.ticket_id)
            outcome.disposition = "escalated"
            outcome.reason = f"Unexpected system error ({exc}); escalated to a human."
            outcome.reply_text = ""
            outcome.escalation_brief = {"ticket_id": ticket.ticket_id, "summary": ticket.summary(240),
                                        "why_escalated": outcome.reason, "unsure_about": ["system error"]}
            self._log(ticket, "routing", "escalate", model={"name": "pipeline", "version": "1.0"},
                      prediction={"value": "escalate", "confidence": 0.0}, alternatives=[], sources_used=[],
                      threshold_applied=None, reason=outcome.reason, guardrail_results={}, prompt_version=None)
        finally:
            self.decisions.finish_ticket(ticket.ticket_id, outcome.disposition)

        outcome.intent, outcome.intent_confidence, outcome.urgency = cls.intent, round(cls.confidence, 4), cls.urgency
        outcome.alternatives = cls.alternatives
        outcome.route_confidence, outcome.threshold, outcome.rule = decision.confidence, decision.threshold, decision.rule
        outcome.sources = [p.to_source() for p in retrieval.passages]
        outcome.top_retrieval_score = round(retrieval.top_score, 4)
        outcome.guardrails = validation.results_summary() if validation else {}
        outcome.blocked_by = validation.blocked_by if validation else []
        outcome.errors = errors
        wait = self.llm.stats["throttle_wait_s"] - wait0
        outcome.latency_s = round(max(time.perf_counter() - t0 - wait, 0.0), 4)   # excludes our own free-tier throttle sleeps
        outcome.throttle_wait_s = round(wait, 4)
        telemetry.observe(outcome.disposition, outcome.latency_s, outcome.blocked_by,
                          outcome.generation_mode == "extractive" and self.llm.available)
        return outcome
