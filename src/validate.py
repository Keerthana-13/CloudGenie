"""Validate: guardrails that run on every response before release, and can block it.

Five checks (names match the governance framework): pii, grounding, instruction_integrity,
tone_scope, confidence_floor. Any failure blocks the reply and the ticket is escalated - we never
"redact and send". All results are recorded whether or not anything was blocked.

These are deliberately simple, inspectable rules, not a second model: a support manager can read
exactly why a reply was stopped, and the checks cannot be talked out of firing.
"""
from __future__ import annotations

import re
from typing import Iterable

from .schemas import Generation, Passage, RouteDecision, Ticket, ValidationResult

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().\-]{8,}\d(?![\w])")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,}|"
                     r"gh[pousr]_[A-Za-z0-9]{30,}|xox[abp]-[A-Za-z0-9\-]{10,}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})")
_LONG_TOKEN = re.compile(r"\b(?=[A-Za-z0-9_\-]{32,}\b)(?=\S*\d)(?=\S*[A-Za-z])[A-Za-z0-9_\-]{32,}\b")
_CUST_ID = re.compile(r"\bCUST-\d+\b", re.I)
_URL = re.compile(r"https?://[^\s)\]>]+")
_BACKTICK = re.compile(r"`([^`]+)`")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")

_STOP = set("""this that with from your have will been were into then than them they their there which when what
would should could about after before also being does done each only over such some very more most other these those
while where whether using used uses make made like just into onto upon within without please thanks thank hello""".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _content_tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) >= 4 and w not in _STOP}


class Guardrails:
    def __init__(self, policy: dict, known_names: Iterable[str] = ()):
        self.commit = [re.compile(p, re.I) for p in policy.get("commitment_patterns", [])]
        self.inject = [re.compile(p, re.I) for p in policy.get("injection_patterns", [])]
        self.known_names: set[str] = {n.strip().lower() for n in known_names if n and len(n.strip()) > 4}
        self.min_support = float(policy.get("grounding_min_support", 0.5))

    def register_names(self, names: Iterable[str]) -> None:
        self.known_names |= {n.strip().lower() for n in names if n and len(n.strip()) > 4}

    # -------------------------------------------------------------- individual checks
    def check_pii(self, ticket: Ticket, text: str) -> dict:
        found = []
        if _EMAIL.search(text):
            found.append("email address")
        if _SECRET.search(text) or _LONG_TOKEN.search(text):
            found.append("key/token-like string")
        if _CARD.search(text):
            found.append("card-number-like digits")
        if _PHONE.search(text):
            found.append("phone-number-like digits")
        others = [c for c in _CUST_ID.findall(text) if c.upper() != (ticket.customer_id or "").upper()]
        if others:
            found.append(f"another customer id ({others[0]})")
        lowered = text.lower()
        me = (ticket.customer_name or "").strip().lower()
        for name in self.known_names:
            if name != me and name in lowered:
                found.append("another customer's name")
                break
        if found:
            return {"status": "fail", "detail": "private data in reply: " + ", ".join(found)}
        return {"status": "pass", "detail": "no private data patterns found"}

    def check_grounding(self, gen: Generation, passages: list[Passage], resolve) -> dict:
        retrieved = {p.doc_id: p for p in passages}
        problems: list[str] = []
        claims = [s for s in gen.sentences if not s.get("framing")]
        if not claims:
            return {"status": "fail", "detail": "reply has no cited claims"}
        for i, s in enumerate(gen.sentences, 1):
            if s.get("framing"):
                continue
            text, cites = s["text"], s.get("cite", [])
            if not cites:
                problems.append(f"sentence {i} has no citation")
                continue
            bad = [c for c in cites if c not in retrieved or resolve(c) is None]
            if bad:
                problems.append(f"sentence {i} cites {bad[0]}, which was not retrieved/does not exist")
                continue
            source = " ".join(retrieved[c].text for c in cites)
            source_l = source.lower()
            tokens = _content_tokens(text)
            if tokens:
                support = len(tokens & _content_tokens(source)) / len(tokens)
                if support < self.min_support:
                    problems.append(f"sentence {i} only {support:.0%} supported by its cited article")
                    continue
            for num in _NUMBER.findall(text):
                if not re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", source):
                    problems.append(f"sentence {i} states '{num}', which is not in the cited article")
                    break
            else:
                for code in _BACKTICK.findall(text):
                    if code.lower() not in source_l:
                        problems.append(f"sentence {i} shows `{code}`, which is not in the cited article")
                        break
                else:
                    for url in _URL.findall(text):
                        if url.lower().rstrip(".,") not in source_l:
                            problems.append(f"sentence {i} links {url}, which is not in the cited article")
                            break
        if problems:
            return {"status": "fail", "detail": "; ".join(problems[:3])}
        return {"status": "pass", "detail": f"{len(claims)} claim sentences each supported by a retrieved article"}

    def check_instruction_integrity(self, ticket: Ticket, reply: str = "", system_prompt: str = "") -> dict:
        text = ticket.model_text
        for rx in self.inject:
            m = rx.search(text)
            if m:
                return {"status": "fail", "detail": f"ticket text contains instruction-like content ('{m.group(0)[:40]}')"}
        if reply and system_prompt:
            marker = system_prompt[:60]
            if marker and marker in reply:
                return {"status": "fail", "detail": "reply leaks the system prompt"}
        if re.search(r"</?(ticket|sources|system)>", reply or "", re.I):
            return {"status": "fail", "detail": "reply contains prompt markup"}
        return {"status": "pass", "detail": "no instruction-like content in ticket or reply"}

    def check_tone_scope(self, text: str) -> dict:
        for rx in self.commit:
            m = rx.search(text)
            if m:
                return {"status": "fail", "detail": f"reply makes a commitment ('{m.group(0)}')"}
        return {"status": "pass", "detail": "no refund/timeline/fix commitments"}

    def check_confidence_floor(self, route: RouteDecision) -> dict:
        c = route.confidence
        ok = isinstance(c, (int, float)) and c == c and 0.0 <= c <= 1.0 and route.threshold > 0
        if ok and c >= route.threshold:
            return {"status": "pass", "detail": f"confidence {c:.2f} >= threshold {route.threshold:.2f} was applied"}
        return {"status": "fail", "detail": f"missing or below-threshold confidence ({c})"}

    # ------------------------------------------------------------------- aggregate
    def input_scan(self, ticket: Ticket) -> ValidationResult:
        """Run before generation so an obvious injection never reaches the model."""
        chk = self.check_instruction_integrity(ticket)
        return ValidationResult(chk["status"] == "pass", {"instruction_integrity": chk},
                                [] if chk["status"] == "pass" else ["instruction_integrity"])

    def validate(self, ticket: Ticket, gen: Generation, passages: list[Passage], route: RouteDecision,
                 resolve, system_prompt: str = "") -> ValidationResult:
        """Check the claims *and* the full reply text (footer included) before release."""
        checks = {
            "pii": self.check_pii(ticket, gen.reply_text),
            "grounding": self.check_grounding(gen, passages, resolve),
            "instruction_integrity": self.check_instruction_integrity(ticket, gen.reply_text, system_prompt),
            "tone_scope": self.check_tone_scope(gen.reply_text),
            "confidence_floor": self.check_confidence_floor(route),
        }
        blocked = [n for n, c in checks.items() if c["status"] == "fail"]
        return ValidationResult(not blocked, checks, blocked)
