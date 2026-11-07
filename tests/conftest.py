"""Shared fixtures. Tests run fully offline: no LLM, no model download, TF-IDF retrieval."""
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_TMP = tempfile.mkdtemp(prefix="cloudserve-tests-")
os.environ.update({"LLM_PROVIDER": "none", "EMBEDDING_BACKEND": "tfidf", "STORAGE_DIR": _TMP,
                   "AUTO_RESPOND_ENABLED": "true"})

from src.config import ROOT  # noqa: E402
from src.llm import LLMUnavailable  # noqa: E402
from src.pipeline import SupportPipeline  # noqa: E402

DATA = ROOT / "data"


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class FakeLLM:
    """Stands in for the model provider. `mode` scripts what the 'model' does."""
    available = True

    def __init__(self, mode="good"):
        self.mode = mode
        self.s = SimpleNamespace(llm_model="fake-model", llm_provider="fake", llm_rpm=0)
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "breaker_skips": 0, "throttle_wait_s": 0.0}

    def chat(self, messages, temperature=0.0, max_tokens=700):
        self.stats["calls"] += 1
        if self.mode == "outage":
            raise LLMUnavailable("simulated provider outage")
        user = messages[-1]["content"]
        ids = re.findall(r"^\[(DOC-[A-Z]+-\d+)\]", user, re.M)
        step = re.search(r"^\d+\.\s+(.*\S)", user, re.M)
        text, doc = step.group(1), ids[0]
        if self.mode == "good":
            sent = [{"text": text, "cite": [doc]}]
        elif self.mode == "pii":
            sent = [{"text": text, "cite": [doc]}, {"text": "Write to jane.doe@example.com for help.", "cite": [doc]}]
        elif self.mode == "fabricated_citation":
            sent = [{"text": text, "cite": ["DOC-FAKE-999"]}]
        elif self.mode == "unsupported_claim":
            sent = [{"text": "Our platform guarantees 99.999% uptime across 47 regions worldwide.", "cite": [doc]}]
        elif self.mode == "commitment":
            sent = [{"text": text, "cite": [doc]}, {"text": "We will refund the difference to your account.", "cite": [doc]}]
        elif self.mode == "decline":
            return json.dumps({"can_answer": False, "sentences": [], "unsure": "sources do not cover this"})
        elif self.mode == "garbage":
            return "Sure! Here is my answer, no JSON though."
        else:
            raise AssertionError(self.mode)
        return json.dumps({"can_answer": True, "sentences": sent, "unsure": ""})


@pytest.fixture(scope="session")
def dev_tickets():
    return load("development_tickets.json")


@pytest.fixture(scope="session")
def val_tickets():
    return load("validation_tickets.json")


@pytest.fixture(scope="session")
def pipe(tmp_path_factory):
    return SupportPipeline(log_path=tmp_path_factory.mktemp("log") / "decisions.db")


@pytest.fixture(scope="session")
def auto_ticket(pipe, val_tickets):
    """A real ticket that the system answers automatically (so guardrail tests have something to block)."""
    for t in val_tickets:
        if pipe.process(t).disposition == "auto_responded":
            return t
    pytest.fail("no auto-respondable validation ticket found")


@pytest.fixture
def make_pipe(tmp_path):
    def _make(mode="good"):
        return SupportPipeline(llm=FakeLLM(mode), log_path=tmp_path / f"{mode}.db")
    return _make
