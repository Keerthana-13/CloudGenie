"""Classify: intent + urgency with a calibrated confidence, plus a learned "answerability" prior.

Why a classical model rather than an LLM call:
  * deterministic (A5) and free of rate limits - the full run cannot stall on a provider;
  * its probabilities can be calibrated and *checked* (see evaluation/), which an LLM's verbalised
    confidence cannot;
  * ~milliseconds per ticket, which keeps p95 latency inside the 3 second target.

Confidence is temperature-scaled on *grouped* out-of-fold predictions. Grouping by ticket body matters:
the development set contains only 216 distinct bodies for 500 tickets, so ordinary cross-validation
would put copies of the same text in train and test folds and report near-perfect, uncalibrated numbers.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from .schemas import ClassResult

log = logging.getLogger(__name__)
MODEL_VERSION = "tfidf-lr-1.0"
DEFAULT_C = 30.0
FALLBACK_INTENT = "unclear_request"      # always escalates; the safe default when we cannot classify


def make_pipeline(C: float = DEFAULT_C) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("feat", feats), ("lr", LogisticRegression(C=C, max_iter=3000))])


def body_groups(texts: list[str]) -> np.ndarray:
    index: dict[str, int] = {}
    return np.array([index.setdefault(" ".join(t.lower().split()), len(index)) for t in texts])


def grouped_oof_proba(texts: list[str], y: list[str], C: float = DEFAULT_C, n_splits: int = 5):
    """Out-of-fold class probabilities where identical texts never straddle train and test."""
    classes = sorted(set(y))
    col = {c: i for i, c in enumerate(classes)}
    groups = body_groups(texts)
    n_splits = max(2, min(n_splits, len(set(groups))))
    P = np.zeros((len(texts), len(classes)))
    X = np.array(texts, dtype=object)
    yy = np.array(y, dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, yy, groups):
        pipe = make_pipeline(C).fit(list(X[tr]), list(yy[tr]))
        proba = pipe.predict_proba(list(X[te]))
        for j, c in enumerate(pipe.classes_):
            P[te, col[c]] = proba[:, j]
    return classes, P


def _apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / T
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def fit_temperature(P: np.ndarray, y_idx: np.ndarray) -> float:
    """Single scalar T minimising negative log-likelihood of the out-of-fold predictions."""
    def nll(T: float) -> float:
        q = _apply_temperature(P, T)
        return float(-np.log(np.clip(q[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)).mean())
    return float(minimize_scalar(nll, bounds=(0.3, 6.0), method="bounded").x)


class TicketClassifier:
    def __init__(self):
        self.intent_pipe: Pipeline | None = None
        self.urgency_pipe: Pipeline | None = None
        self.temperature: float = 1.0
        self.prior: dict = {}
        self.intent_docs: dict[str, list[str]] = {}
        self.meta: dict = {}

    # ---------------------------------------------------------------- training
    def fit(self, tickets: list[dict], C: float = DEFAULT_C) -> "TicketClassifier":
        usable = [t for t in tickets if t.get("labels") and (t.get("body") or "").strip()]
        if not usable:
            raise ValueError("no labelled tickets to train on")
        texts = [t["body"] for t in usable]
        y_intent = [t["labels"]["intent"] for t in usable]
        y_urg = [t["labels"]["urgency"] for t in usable]

        T = 1.0
        if len(set(body_groups(texts))) >= 4 and len(set(y_intent)) >= 2:
            classes, P = grouped_oof_proba(texts, y_intent, C)
            T = fit_temperature(P, np.array([classes.index(c) for c in y_intent]))
        self.temperature = T
        self.intent_pipe = make_pipeline(C).fit(texts, y_intent)
        self.urgency_pipe = make_pipeline(C).fit(texts, y_urg) if len(set(y_urg)) >= 2 else None
        self._fit_prior(usable)
        self._fit_intent_docs(usable)
        self.meta = {"version": MODEL_VERSION, "n_train": len(usable), "C": C, "temperature": round(T, 4),
                     "sklearn": sklearn.__version__}
        return self

    def _fit_prior(self, tickets: list[dict]) -> None:
        """P(answerable | intent): share of past tickets of this intent that were routed to auto-respond.

        In the development data the text of a ticket does not fully determine whether it was answerable:
        identical bodies appear with both outcomes. The honest confidence that auto-answering is right is
        therefore (confidence in the intent) x (how often tickets of this intent are answerable), smoothed
        toward the global rate so thin classes do not dominate.

        Predicted urgency is deliberately NOT used: out-of-fold urgency accuracy is 49% against a 45% majority
        baseline, i.e. no usable signal, and adding it to the prior made grouped-CV routing slightly worse.
        """
        n_i, a_i = defaultdict(int), defaultdict(int)
        n_g = a_g = 0
        for t in tickets:
            lab = t["labels"]
            auto = int(lab.get("expected_route") == "auto_respond")
            n_i[lab["intent"]] += 1
            a_i[lab["intent"]] += auto
            n_g += 1
            a_g += auto
        p_g = a_g / max(n_g, 1)
        self.prior = {"global": p_g, "intent": {i: (a_i[i] + 2 * p_g) / (n_i[i] + 2) for i in n_i}}

    def _fit_intent_docs(self, tickets: list[dict]) -> None:
        """Which articles answer which intent (>=30% of that intent's answerable tickets). Used only to
        nudge retrieval ranking; it never creates a citation on its own."""
        seen, answerable = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for t in tickets:
            lab = t["labels"]
            if lab.get("expected_doc_ids"):
                answerable[lab["intent"]] += 1
                for d in lab["expected_doc_ids"]:
                    seen[lab["intent"]][d] += 1
        self.intent_docs = {i: sorted(d for d, c in seen[i].items() if c / answerable[i] >= 0.3) for i in answerable}

    def answer_prior(self, intent: str, urgency: str = "") -> float:
        return float(self.prior.get("intent", {}).get(intent, self.prior.get("global", 0.0)))

    # --------------------------------------------------------------- inference
    def predict(self, text: str) -> ClassResult:
        """Never raises: any failure returns the safe fallback (an intent that always escalates)."""
        try:
            if not text or not text.strip() or self.intent_pipe is None:
                raise ValueError("empty text or untrained model")
            proba = self.intent_pipe.predict_proba([text])[0]
            proba = _apply_temperature(proba, self.temperature)
            order = np.argsort(-proba, kind="stable")[:3]
            classes = self.intent_pipe.classes_
            intent, conf = str(classes[order[0]]), float(proba[order[0]])
            alts = [{"value": str(classes[i]), "confidence": round(float(proba[i]), 4)} for i in order[1:]]
            urgency, u_conf = "medium", 0.0
            if self.urgency_pipe is not None:
                up = self.urgency_pipe.predict_proba([text])[0]
                ui = int(np.argmax(up))
                urgency, u_conf = str(self.urgency_pipe.classes_[ui]), float(up[ui])
            return ClassResult(intent, conf, alts, urgency, u_conf, self.answer_prior(intent, urgency),
                               False, MODEL_VERSION)
        except Exception as exc:
            log.warning("classification fallback: %s", exc)
            return ClassResult(FALLBACK_INTENT, 0.0, [], "medium", 0.0, 0.0, True, MODEL_VERSION)

    # ------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"intent": self.intent_pipe, "urgency": self.urgency_pipe, "T": self.temperature,
                     "prior": self.prior, "intent_docs": self.intent_docs, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "TicketClassifier":
        blob = joblib.load(path)
        obj = cls()
        obj.intent_pipe, obj.urgency_pipe = blob["intent"], blob["urgency"]
        obj.temperature, obj.prior, obj.meta = blob["T"], blob["prior"], blob["meta"]
        obj.intent_docs = blob.get("intent_docs", {})
        return obj

    @classmethod
    def load_or_train(cls, dev_path: Path, storage_dir: Path) -> "TicketClassifier":
        """Reuse a cached model only if it was trained on exactly this file with this library version."""
        dev_path = Path(dev_path)
        digest = hashlib.sha256(dev_path.read_bytes()).hexdigest()[:16]
        cache = Path(storage_dir) / "models" / f"classifier-{digest}-{sklearn.__version__}.joblib"
        if cache.exists():
            try:
                return cls.load(cache)
            except Exception as exc:
                log.warning("discarding unreadable cached model: %s", exc)
        tickets = json.loads(dev_path.read_text(encoding="utf-8"))
        model = cls().fit(tickets)
        model.meta["trained_on"] = dev_path.name
        try:
            model.save(cache)
        except Exception as exc:        # read-only filesystem etc. - training in memory is fine
            log.warning("could not cache model: %s", exc)
        return model
