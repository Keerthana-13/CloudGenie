from src.config import get_settings
from src.knowledge_base import KnowledgeBase, chunk_document


def kb():
    s = get_settings()
    return KnowledgeBase.from_files(s.data_dir / "documentation.json", s.data_dir / "development_tickets.json",
                                    embedder_backend="tfidf")


def test_known_question_finds_right_article():
    r = kb().search("Our connection pool appears to be exhausted and requests fail with no connection available", top_k=3)
    assert r.passages and r.passages[0].doc_id == "DOC-PERF-003"


def test_sources_resolve_to_real_corpus():
    k = kb()
    for p in k.search("how do I rotate an api key without downtime", top_k=3).passages:
        assert k.resolve(p.doc_id) is not None and p.text == k.resolve(p.doc_id)["content"]


def test_gibberish_returns_nothing_not_something():
    r = kb().search("xkcd qwzv plorf zzzz", top_k=3, min_score=0.077)
    assert r.passages == []


def test_empty_query():
    assert kb().search("   ").passages == []


def test_deterministic():
    k = kb()
    a = [p.doc_id for p in k.search("webhook retries failing", 3).passages]
    assert a == [p.doc_id for p in k.search("webhook retries failing", 3).passages]


def test_chunking_keeps_resolution_steps_together(dev_tickets):
    docs = kb().docs
    sections = {c.section for d in docs.values() for c in chunk_document(d)}
    assert {"Symptoms", "Common causes", "Resolution", "Notes"} <= sections
    res = [c for c in chunk_document(docs["DOC-AUTH-001"]) if c.section == "Resolution"][0]
    assert all(f"{n}." in res.text for n in (1, 2, 3, 4))
