"""Prometheus metrics, optional: if prometheus_client is missing every call is a harmless no-op."""
from __future__ import annotations

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _OK = True
except Exception:          # pragma: no cover - library is in requirements.txt, but never hard-fail on it
    _OK = False

REGISTRY = None
_tickets = _blocks = _llm_fail = _latency = None
if _OK:
    REGISTRY = CollectorRegistry()
    _tickets = Counter("cloudserve_tickets_total", "Tickets processed", ["disposition"], registry=REGISTRY)
    _blocks = Counter("cloudserve_guardrail_blocks_total", "Replies blocked by guardrail", ["guardrail"], registry=REGISTRY)
    _llm_fail = Counter("cloudserve_generation_fallbacks_total", "Replies produced without the LLM", registry=REGISTRY)
    _latency = Histogram("cloudserve_ticket_latency_seconds", "End-to-end latency per ticket",
                         buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30), registry=REGISTRY)


def observe(disposition: str, latency_s: float, blocked_by: list[str], used_fallback: bool) -> None:
    if not _OK:
        return
    _tickets.labels(disposition).inc()
    _latency.observe(latency_s)
    for g in blocked_by:
        _blocks.labels(g).inc()
    if used_fallback:
        _llm_fail.inc()


def render() -> tuple[bytes, str]:
    if not _OK:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
