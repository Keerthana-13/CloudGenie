"""Embedding backends.

Preferred: all-MiniLM-L6-v2 (dense, semantic, handles paraphrase and non-fluent phrasing better).
Fallback: TF-IDF (word + character n-grams) - fully offline, deterministic, no download.

The fallback is not only a convenience: it means a missing model download or a machine without
network access degrades retrieval quality instead of stopping the system (acceptance criterion A11).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

log = logging.getLogger(__name__)


class EmbedderUnavailable(RuntimeError):
    pass


class TfidfEmbedder:
    name = "tfidf"
    dense = False

    def __init__(self, fit_corpus: Sequence[str]):
        self._vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])
        self._vec.fit(list(fit_corpus))

    def encode(self, texts: Sequence[str]) -> sparse.csr_matrix:
        m = sparse.csr_matrix(self._vec.transform(list(texts)))
        # each sub-vectorizer is L2-normalised; renormalise the concatenation so cosine is in [0, 1]
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ m


class MiniLMEmbedder:
    dense = True

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # missing package, no network for first download, corrupted cache...
            raise EmbedderUnavailable(f"MiniLM unavailable: {exc}") from exc
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


def similarity(queries, matrix) -> np.ndarray:
    """Cosine similarity for either sparse (TF-IDF) or dense (MiniLM) unit vectors."""
    if sparse.issparse(queries) or sparse.issparse(matrix):
        return np.asarray((queries @ matrix.T).todense())
    return np.asarray(queries @ matrix.T)


def get_embedder(backend: str, fit_corpus: Sequence[str], model_name: str = "all-MiniLM-L6-v2"):
    """backend: auto | minilm | tfidf.  `auto` tries MiniLM and falls back to TF-IDF with a warning."""
    if backend in {"auto", "minilm"}:
        try:
            return MiniLMEmbedder(model_name)
        except EmbedderUnavailable as exc:
            if backend == "minilm":
                raise
            log.warning("%s - falling back to offline TF-IDF retrieval", exc)
    return TfidfEmbedder(fit_corpus)
