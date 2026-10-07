"""Hybrid retrieval: BM25 (sparse) + vector similarity (dense), fused with
Reciprocal Rank Fusion, then a reranking pass.

The default dense encoder is a TF-IDF + truncated SVD (LSA) embedding so the
project runs offline with no model downloads. Set EMBEDDING_BACKEND to
"sentence-transformers" to use a neural embedding model instead.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it its me my of on or should "
    "that the their this to was what when which who why with you your".split()
)


def stem(token: str) -> str:
    """Tiny suffix stripper so 'reapplied' matches 'reapply' and 'hours' matches 'hour'."""
    if len(token) <= 4:
        return token
    for suffix, repl in (("ied", "y"), ("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + repl
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sparse_rank: int | None = None
    dense_rank: int | None = None


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            for t in query:
                if t in tf:
                    out[i] += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
        return out


class LsaEncoder:
    """Offline dense encoder: TF-IDF followed by truncated SVD, L2-normalised."""

    def __init__(self, dim: int = 64):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(STOPWORDS))
        self.dim = dim
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LsaEncoder":
        x = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=7).fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.svd.transform(self.vectorizer.transform(texts))
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def _make_encoder(backend: str):
    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer  # optional dependency

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        class _ST:
            def fit(self, texts):
                return self

            def encode(self, texts):
                return model.encode(texts, normalize_embeddings=True)

        return _ST()
    return LsaEncoder()


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedding_backend: str = "lsa", rrf_k: int = 60):
        if not chunks:
            raise ValueError("HybridRetriever needs at least one chunk")
        self.chunks = chunks
        self.rrf_k = rrf_k
        texts = [f"{c.title}. {c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.encoder = _make_encoder(embedding_backend).fit(texts)
        self.matrix = self.encoder.encode(texts)
        logger.info("Indexed %d chunks (dense backend=%s)", len(chunks), embedding_backend)

    def search(self, query: str, top_k: int = 4, candidates: int = 10) -> list[Hit]:
        q_tokens = tokenize(query)
        sparse = self.bm25.scores(q_tokens)
        dense = self.matrix @ self.encoder.encode([query])[0]
        sparse_order = [i for i in np.argsort(-sparse) if sparse[i] > 0][:candidates]
        dense_order = list(np.argsort(-dense)[:candidates])

        fused: dict[int, Hit] = {}
        for rank, i in enumerate(sparse_order):
            hit = fused.setdefault(i, Hit(self.chunks[i], 0.0))
            hit.score += 1.0 / (self.rrf_k + rank + 1)
            hit.sparse_rank = rank + 1
        for rank, i in enumerate(dense_order):
            hit = fused.setdefault(i, Hit(self.chunks[i], 0.0))
            hit.score += 1.0 / (self.rrf_k + rank + 1)
            hit.dense_rank = rank + 1

        return rerank(query, sorted(fused.values(), key=lambda h: -h.score))[:top_k]


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """Lightweight reranker: blends fused score with query-term coverage.

    Swap for a cross-encoder (e.g. ms-marco-MiniLM) in production; this keeps
    the default build dependency-free and deterministic for CI.
    """
    q = set(tokenize(query))
    if not q:
        return hits
    for h in hits:
        coverage = len(q & set(tokenize(h.chunk.section + " " + h.chunk.text))) / len(q)
        h.score = h.score * (1 + coverage)
    return sorted(hits, key=lambda h: -h.score)
