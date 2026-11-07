import copy

import pytest

from src.classifier import TicketClassifier
from src.config import get_settings, load_policy
from src.ingest import normalise
from src.router import route
from src.schemas import RetrievalResult


@pytest.fixture(scope="module")
def clf():
    s = get_settings()
    return TicketClassifier.load_or_train(s.data_dir / "development_tickets.json", s.storage_dir)


def test_every_prediction_has_class_and_confidence_in_unit_interval(clf, dev_tickets):
    for t in dev_tickets[:40]:
        r = clf.predict(t["body"])
        assert r.intent and 0.0 <= r.confidence <= 1.0 and r.urgency in {"high", "medium", "low"}
        assert 1 <= len(r.alternatives) <= 2          # alternatives recorded, not just the winner


def test_fallback_on_empty_text_does_not_raise(clf):
    r = clf.predict("")
    assert r.fallback and r.intent == "unclear_request" and r.confidence == 0.0


def test_classification_deterministic(clf):
    a, b = clf.predict("Our API key is returning 401"), clf.predict("Our API key is returning 401")
    assert a == b


def _decision(ticket_raw, clf, kb_result, policy=None, kill=False):
    t = normalise(ticket_raw)
    return route(t, clf.predict(t.model_text), kb_result, policy or load_policy(), kill)


@pytest.fixture
def good_retrieval(pipe):
    return pipe.kb.search("how do I paginate large result sets", 3)


def test_same_input_same_routing_decision(clf, good_retrieval):
    raw = {"ticket_id": "T", "channel": "email", "body": "How do I page through very large result sets from the API?"}
    a, b = _decision(raw, clf, good_retrieval), _decision(raw, clf, good_retrieval)
    assert (a.action, a.rule, a.confidence, a.reason) == (b.action, b.rule, b.confidence, b.reason)


def test_security_keyword_always_escalates(clf, good_retrieval):
    raw = {"ticket_id": "T", "channel": "chat", "body": "I think our account was compromised and someone changed settings"}
    d = _decision(raw, clf, good_retrieval)
    assert d.action == "escalate" and d.rule == "safety_keyword"


def test_must_escalate_intents_never_auto_respond(clf, good_retrieval, dev_tickets):
    policy = load_policy()
    for t in [x for x in dev_tickets if x["labels"]["must_not_auto_respond"]][:60]:
        assert _decision(t, clf, good_retrieval, policy).action == "escalate"


def test_no_documentation_escalates(clf):
    empty = RetrievalResult([], 0.01, "tfidf+memory", 0.077)
    d = _decision({"ticket_id": "T", "channel": "email", "body": "How do I page through large result sets?"}, clf, empty)
    assert d.action == "escalate" and d.rule == "no_relevant_documentation"


def test_kill_switch_escalates_everything(clf, good_retrieval):
    d = _decision({"ticket_id": "T", "channel": "email", "body": "How do I page through large result sets?"}, clf,
                  good_retrieval, kill=True)
    assert d.action == "escalate" and d.rule == "kill_switch"


def test_threshold_is_applied(clf, good_retrieval):
    policy = copy.deepcopy(load_policy())
    raw = {"ticket_id": "T", "channel": "email", "body": "How do I page through very large result sets from the API?"}
    policy["confidence_threshold"] = 0.0 + 0.01
    assert _decision(raw, clf, good_retrieval, policy).action == "auto_respond"
    policy["confidence_threshold"] = 0.999
    d = _decision(raw, clf, good_retrieval, policy)
    assert d.action == "escalate" and d.rule == "low_confidence" and "threshold" in d.reason
