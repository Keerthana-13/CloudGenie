"""Retrieve: the support knowledge base (29 articles) as a searchable index.

Chunking decision: every article has the same four sections (Symptoms, Common causes, Resolution,
Notes). We index each *section* (prefixed with the article title) so a customer's description of a
symptom matches the Symptoms chunk, but we return the *whole parent article* as the passage. That
keeps resolution steps intact - splitting inside a numbered sequence gives passages that retrieve
well but read as incomplete - and gives every citation a stable `doc_id` that resolves to the corpus.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import get_embedder, similarity
from .schemas import Passage, RetrievalResult

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


def chunk_document(doc: dict) -> list[Chunk]:
    content = doc.get("content", "")
    title = doc.get("title", doc["doc_id"])
    parts = re.split(r"\n(?=## )", content)
    chunks: list[Chunk] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"##\s+(.*)", part)
        section = m.group(1).strip() if m else "Overview"
        if i == 0 and not m:  # leading "# Title" + "Applies to" block
            part = re.sub(r"^#\s+.*\n?", "", part).strip()
            section = "Overview"
            if not part:
                continue
        chunks.append(Chunk(f"{doc['doc_id']}#{len(chunks)}", doc["doc_id"], title, section, f"{title}. {part}"))
    if not chunks:
        chunks.append(Chunk(f"{doc['doc_id']}#0", doc["doc_id"], title, "Overview", f"{title}. {content}"))
    return chunks


class _MemoryIndex:
    name = "memory"

    def __init__(self, embedder, chunks: list[Chunk]):
        self._embedder = embedder
        self._matrix = embedder.encode([c.text for c in chunks])

    def query(self, text: str) -> np.ndarray:
        return similarity(self._embedder.encode([text]), self._matrix)[0]


class _ChromaIndex:
    name = "chroma"

    def __init__(self, embedder, chunks: list[Chunk], path: Path | None):
        import chromadb  # optional dependency
        client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        try:
            client.delete_collection("cloudserve_kb")
        except Exception:
            pass
        self._col = client.create_collection("cloudserve_kb", metadata={"hnsw:space": "cosine"})
        self._n = len(chunks)
        self._embedder = embedder
        self._col.add(ids=[c.chunk_id for c in chunks],
                      embeddings=embedder.encode([c.text for c in chunks]).tolist(),
                      documents=[c.text for c in chunks])
        self._ids = [c.chunk_id for c in chunks]

    def query(self, text: str) -> np.ndarray:
        res = self._col.query(query_embeddings=self._embedder.encode([text]).tolist(),
                              n_results=self._n, include=["distances"])
        scores = np.zeros(self._n)
        pos = {cid: i for i, cid in enumerate(self._ids)}
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            scores[pos[cid]] = 1.0 - float(dist)          # cosine distance -> similarity
        return scores


class KnowledgeBase:
    def __init__(self, docs: list[dict], embedder_backend: str = "auto", embedding_model: str = "all-MiniLM-L6-v2",
                 extra_fit_texts: list[str] | None = None, vector_backend: str = "auto",
                 chroma_path: Path | None = None, embedder=None):
        self.docs = {d["doc_id"]: d for d in docs}
        self.chunks = [c for d in docs for c in chunk_document(d)]
        fit_corpus = [c.text for c in self.chunks] + list(extra_fit_texts or [])
        self.embedder = embedder or get_embedder(embedder_backend, fit_corpus, embedding_model)
        index = None
        if self.embedder.dense and vector_backend in {"auto", "chroma"}:
            try:
                index = _ChromaIndex(self.embedder, self.chunks, chroma_path)
            except Exception as exc:
                if vector_backend == "chroma":
                    raise
                log.warning("Chroma unavailable (%s); using in-memory index", exc)
        self.index = index or _MemoryIndex(self.embedder, self.chunks)

    @classmethod
    def from_files(cls, docs_path: Path, dev_tickets_path: Path | None = None, **kwargs) -> "KnowledgeBase":
        docs = json.loads(Path(docs_path).read_text(encoding="utf-8"))
        extra: list[str] = []
        if dev_tickets_path and Path(dev_tickets_path).exists():
            # Unsupervised vocabulary only (TF-IDF idf weights); no labels are used here.
            extra = [t.get("body", "") for t in json.loads(Path(dev_tickets_path).read_text(encoding="utf-8"))]
        return cls(docs, extra_fit_texts=extra, **kwargs)

    @property
    def backend(self) -> str:
        return f"{self.embedder.name}+{self.index.name}"

    def resolve(self, doc_id: str) -> dict | None:
        """Citation check: does this identifier exist in the real corpus?"""
        return self.docs.get(doc_id)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0,
               prefer_docs: set[str] | None = None, boost: float = 0.0) -> RetrievalResult:
        """Ranked articles (best chunk per article). Returns nothing below `min_score` - an empty result
        is a valid and often correct answer. `prefer_docs`/`boost` re-rank (never admit) articles that the
        predicted intent usually maps to; the relevance threshold is always applied to the raw score."""
        if not query or not query.strip():
            return RetrievalResult([], 0.0, self.backend, min_score)
        scores = self.index.query(query)
        best: dict[str, tuple[float, Chunk]] = {}
        for chunk, score in zip(self.chunks, scores):
            if chunk.doc_id not in best or score > best[chunk.doc_id][0]:
                best[chunk.doc_id] = (float(score), chunk)
        prefer = prefer_docs or set()
        ranked = sorted(best.values(), key=lambda x: (-(x[0] + (boost if x[1].doc_id in prefer else 0.0)), x[1].doc_id))
        top_score = max((s for s, _ in ranked), default=0.0)
        passages = [
            Passage(doc_id=c.doc_id, chunk_id=c.chunk_id, title=c.title, section=c.section,
                    text=self.docs[c.doc_id]["content"], score=s)
            for s, c in ranked[:top_k] if s >= min_score
        ]
        return RetrievalResult(passages, top_score, self.backend, min_score)
