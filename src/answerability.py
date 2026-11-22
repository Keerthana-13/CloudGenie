"""Answerability: a per-ticket estimate that automatically answering this ticket is the right call.

The older routing confidence was  P(intent right) x P(answerable | intent), where the second factor is one
fixed number per intent, so two tickets of the same intent differed only through the intent probability and
nothing said whether the retrieved article really fits *this* ticket. This model keeps those two signals but
adds evidence about the retrieval itself:

    intent_conf          calibrated probability of the predicted intent
    answer_prior         historical auto-answer rate for that intent
    intent_x_prior       their product (the previous confidence)
    top_score            best article relevance score (before the relevance floor)
    score_margin         gap between the best and second-best article (a clear winner vs. a toss-up)
    top_in_intent_docs   1 if the best article is one that historically answers this intent
    n_passages           how many articles cleared the relevance floor

It is a plain logistic regression, so its output is already a probability and its parameters are stored in
config/policy.json as numbers (no pickle): the router stays a pure function of (ticket, classification,
retrieval, policy). Fitting happens in evaluation/tune_thresholds.py on grouped out-of-fold predictions only.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from .schemas import ClassResult, RetrievalResult

FEATURES = ["intent_conf", "answer_prior", "intent_x_prior", "top_score", "score_margin",
            "top_in_intent_docs", "n_passages"]


def features(cls: ClassResult, retrieval: RetrievalResult) -> list[float]:
    scores = sorted((p.score for p in retrieval.passages), reverse=True)
    best = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else 0.0
    in_docs = float(bool(retrieval.passages) and retrieval.passages[0].doc_id in set(cls.preferred_docs))
    return [float(cls.confidence), float(cls.answer_prior), float(cls.confidence * cls.answer_prior),
            float(retrieval.top_score), float(best - second), in_docs, float(len(retrieval.passages))]


def fit(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> dict:
    """Standardise, fit a regularised logistic regression, return JSON-serialisable parameters."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    lr = LogisticRegression(C=C, max_iter=2000).fit((X - mean) / scale, np.asarray(y, dtype=int))
    return {"features": FEATURES, "mean": [round(float(v), 6) for v in mean], "scale": [round(float(v), 6) for v in scale],
            "coef": [round(float(v), 6) for v in lr.coef_[0]], "intercept": round(float(lr.intercept_[0]), 6), "C": C}


def predict(model: dict, x: list[float]) -> float:
    z = float(model["intercept"])
    for v, m, s, w in zip(x, model["mean"], model["scale"], model["coef"]):
        z += w * (v - m) / s
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def confidence(model: dict, cls: ClassResult, retrieval: RetrievalResult) -> float:
    if cls.fallback:
        return 0.0
    return round(predict(model, features(cls, retrieval)), 4)
