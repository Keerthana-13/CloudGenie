import json

import numpy as np

from evaluation import metrics as M
from src import answerability
from src.router import route, route_confidence
from src.schemas import ClassResult, Passage, RetrievalResult, Ticket


def _cls(conf=0.9, prior=0.8, docs=("D1",)):
    return ClassResult("api_usage_question", conf, [], "medium", 0.0, prior, False, "t", list(docs))


def _ret(*scores, doc_ids=("D1", "D2", "D3")):
    ps = [Passage(d, d + "#0", "t", "s", "text", s) for d, s in zip(doc_ids, scores)]
    return RetrievalResult(ps, max(scores, default=0.0), "tfidf", 0.1)


def _ticket():
    return Ticket("T1", "email", "", "how do I paginate results")


def test_features_match_declared_names_and_reflect_retrieval():
    f = answerability.features(_cls(), _ret(0.8, 0.3))
    assert len(f) == len(answerability.FEATURES)
    d = dict(zip(answerability.FEATURES, f))
    assert d["top_in_intent_docs"] == 1.0 and abs(d["score_margin"] - 0.5) < 1e-9 and d["n_passages"] == 2.0
    # top article is not one that usually answers this intent -> flag off
    assert dict(zip(answerability.FEATURES, answerability.features(_cls(docs=("D9",)), _ret(0.8))))["top_in_intent_docs"] == 0.0


def test_model_round_trips_through_json_and_learns_the_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(answerability.FEATURES)))
    y = (X[:, 3] + 0.3 * rng.normal(size=400)) > 0            # top_score decides
    model = json.loads(json.dumps(answerability.fit(X, y)))   # policy.json stores it as plain numbers
    hi, lo = X[0].copy(), X[0].copy()
    hi[3], lo[3] = 3.0, -3.0
    p_hi, p_lo = answerability.predict(model, list(hi)), answerability.predict(model, list(lo))
    assert 0.0 <= p_lo < 0.5 < p_hi <= 1.0


def test_router_uses_the_answerability_model_when_policy_has_one():
    policy = {"confidence_threshold": 0.5, "retrieval_min_score": 0.1, "must_escalate_intents": [],
              "security_keywords": []}
    n = len(answerability.FEATURES)
    zero = {"features": answerability.FEATURES, "mean": [0.0] * n, "scale": [1.0] * n, "coef": [0.0] * n}
    yes, no = {**zero, "intercept": 4.0}, {**zero, "intercept": -4.0}
    cls, ret = _cls(), _ret(0.8, 0.3)
    assert route(_ticket(), cls, ret, {**policy, "answerability": yes}).action == "auto_respond"
    d = route(_ticket(), cls, ret, {**policy, "answerability": no})
    assert d.action == "escalate" and d.rule == "low_confidence"
    assert route_confidence(cls, ret, {**policy, "answerability": yes}) > 0.9


def test_router_falls_back_to_legacy_score_without_a_model():
    policy = {"confidence_threshold": 0.5, "retrieval_min_score": 0.1, "must_escalate_intents": [], "security_keywords": []}
    cls = _cls(conf=0.9, prior=0.8)
    assert route_confidence(cls, _ret(0.8), policy) == round(0.9 * 0.8, 4)      # no calibration map -> raw product
    assert route_confidence(ClassResult("x", 0.9, [], "low", 0.0, 0.8, True), _ret(0.8), policy) == 0.0


# ------------------------------------------------------------ bootstrap intervals for fairness
def _pairs(groups):
    """groups: name -> (n tickets, n answered automatically); every ticket is labelled auto_respond."""
    out = []
    for name, (n, auto) in groups.items():
        for i in range(n):
            t = {"g": name, "labels": {"intent": "a", "expected_route": "auto_respond", "expected_doc_ids": []}}
            o = {"disposition": "auto_responded" if i < auto else "escalated", "intent": "a", "sources": []}
            out.append((t, o))
    return out


def _fair(groups, **kw):
    return M.fairness_report(_pairs(groups), {"g": lambda t: t["g"]}, **kw)["g"]


def test_interval_brackets_the_point_estimate_and_is_reproducible():
    a = _fair({"x": (40, 30), "y": (40, 20)})
    b = _fair({"x": (40, 30), "y": (40, 20)})
    assert a == b                                              # fixed seed -> identical report
    lo, hi = a["groups"]["x"]["ci95_pts"]["auto_response_rate"]
    assert lo <= 75.0 <= hi and 0 <= lo < hi <= 100


def test_large_clear_gap_is_credibly_over_target():
    rep = _fair({"x": (200, 180), "y": (200, 100)})            # 90% vs 50%
    assert "auto_response_rate" in rep["credibly_over_target"]
    assert rep["spread_verdict"]["auto_response_rate"].startswith("over target")
    assert rep["spread_ci95_pts"]["auto_response_rate"][0] > 5


def test_small_groups_give_wide_inconclusive_intervals():
    rep = _fair({"x": (60, 45), "y": (12, 8)})                 # 75% vs 66.7%, one group has only 12 tickets
    lo, hi = rep["groups"]["y"]["ci95_pts"]["auto_response_rate"]
    assert hi - lo > 30                                        # a 12-ticket rate is very uncertain
    assert rep["spread_verdict"]["auto_response_rate"].startswith("inconclusive")
    assert "auto_response_rate" not in rep["credibly_over_target"]


def test_groups_below_minimum_are_excluded_and_metrics_without_data_have_no_interval():
    rep = _fair({"x": (30, 20), "tiny": (3, 0)})
    assert rep["spread_ci95_pts"]["auto_response_rate"] is None          # only one eligible group -> no spread
    assert rep["groups"]["tiny"]["ci95_pts"]["retrieval_hit_rate"] is None   # no expected docs -> nothing to resample
