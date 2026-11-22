"""Generate: a cited answer grounded in the retrieved articles.

Two modes, chosen automatically and recorded on every result:
  * llm        - the configured model drafts the answer from the retrieved articles (prompt PR-02);
  * extractive - no model needed: the article's own resolution steps, each cited to its article.
The extractive mode is both the offline fallback (provider outage, rate limiting, no key: A11) and a
safe baseline - it cannot hallucinate because it only repeats corpus text.
"""
from __future__ import annotations

import json
import logging
import re

from .llm import LLMClient, LLMUnavailable
from .prompts import load_prompt
from .schemas import Generation, Passage, Ticket

log = logging.getLogger(__name__)
GENERATE_PROMPT_FILE = "PR-02_generate_answer.v1.md"

FOOTER = ("This reply was drafted automatically from Cloud Genie's support documentation. "
          "If it does not resolve your issue, reply to this message and our support team will take over.")


def _sanitise_ticket_text(text: str) -> str:
    """Stop customer text from closing our <ticket> tag or opening fake ones."""
    return text.replace("<", "‹").replace(">", "›")


def _sources_block(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.doc_id}] {p.title}\n{p.text}" for p in passages)


def parse_llm_json(raw: str) -> dict | None:
    """Pull the JSON object out of a model reply (tolerates code fences and chatter). None if invalid."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("can_answer"), bool):
        return None
    sentences = []
    for s in data.get("sentences") or []:
        if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"].strip():
            cites = s.get("cite") or []
            cites = [c for c in cites if isinstance(c, str)] if isinstance(cites, list) else []
            sentences.append({"text": s["text"].strip(), "cite": cites})
    return {"can_answer": data["can_answer"], "sentences": sentences, "unsure": str(data.get("unsure") or "")}


def compose_reply(sentences: list[dict], passages: list[Passage]) -> str:
    body = " ".join(f"{s['text']} [{', '.join(s['cite'])}]" if s.get("cite") else s["text"] for s in sentences)
    titles = {p.doc_id: p.title for p in passages}
    cited = []
    for s in sentences:
        for d in s.get("cite", []):
            if d not in cited:
                cited.append(d)
    sources = "\n".join(f"- {d}: {titles.get(d, '(unknown)')}" for d in cited)
    return f"Hello,\n\n{body}\n\nSources:\n{sources}\n\n{FOOTER}"


def extractive_sentences(passage: Passage, max_steps: int = 4) -> list[dict]:
    """Resolution steps lifted from the article, each cited to it."""
    m = re.search(r"##\s*Resolution\s*\n(.*?)(?:\n##\s|\Z)", passage.text, re.S)
    steps = re.findall(r"^\s*\d+\.\s+(.*\S)", m.group(1), re.M) if m else []
    # The opening line is framing, not a factual claim, so it carries no citation and is exempt from the
    # grounding check. Only code in this module can set "framing": parse_llm_json never copies it from a model.
    sentences = [{"text": f'Our article "{passage.title}" describes this situation.', "cite": [], "framing": True}]
    sentences += [{"text": s, "cite": [passage.doc_id]} for s in steps[:max_steps]]
    return sentences


class Generator:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm
        self.prompt = load_prompt(GENERATE_PROMPT_FILE)

    def _extractive(self, passages: list[Passage], error: str = "") -> Generation:
        top = passages[0]
        sentences = extractive_sentences(top)
        return Generation("extractive", True, sentences, compose_reply(sentences, passages[:1]),
                          prompt_version="extractive-1.0", model="none", error=error)

    def extractive_draft(self, passages: list[Passage]) -> Generation:
        """Model-free draft for human agents (no quota spent, available during an outage)."""
        if not passages:
            return Generation("none", False, [], unsure="no passages retrieved")
        return self._extractive(passages)

    def generate(self, ticket: Ticket, passages: list[Passage]) -> Generation:
        if not passages:
            return Generation("none", False, [], unsure="no passages retrieved", error="no passages")
        if self.llm is None or not self.llm.available:
            return self._extractive(passages, "LLM not configured")
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.render_user(
                channel=ticket.channel, ticket_text=_sanitise_ticket_text(ticket.model_text),
                sources=_sources_block(passages))},
        ]
        try:
            raw = self.llm.chat(messages, temperature=0.0)
        except LLMUnavailable as exc:
            log.warning("LLM unavailable, using extractive answer: %s", exc)
            return self._extractive(passages, f"LLM unavailable: {exc}")
        parsed = parse_llm_json(raw)
        if parsed is None:
            log.warning("unparseable LLM output, using extractive answer")
            return self._extractive(passages, "LLM output was not valid JSON")
        if not parsed["can_answer"] or not parsed["sentences"]:
            return Generation("llm", False, [], unsure=parsed["unsure"] or "model could not answer from sources",
                              prompt_version=self.prompt.label, model=self.llm.s.llm_model)
        return Generation("llm", True, parsed["sentences"], compose_reply(parsed["sentences"], passages),
                          unsure=parsed["unsure"], prompt_version=self.prompt.label, model=self.llm.s.llm_model)
