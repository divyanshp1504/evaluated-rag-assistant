"""End-to-end question answering: guardrails -> retrieve -> generate -> verify."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .guardrails import DISCLAIMER, check_input
from .ingest import load_corpus
from .llm import Generator, get_generator
from .retrieval import HybridRetriever, tokenize

logger = logging.getLogger(__name__)
_CITATION = re.compile(r"\[[^\]]+#\d+\]")

REFUSAL_NO_CONTEXT = "I could not find this in the knowledge base, so I can't answer it reliably."
REFUSAL_INJECTION = (
    "This request looks like an attempt to override the assistant's instructions, so it was not processed."
)


@dataclass
class Settings:
    corpus_dir: str = os.getenv("CORPUS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "corpus"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "80"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "20"))
    min_groundedness: float = float(os.getenv("MIN_GROUNDEDNESS", "0.6"))
    min_query_coverage: float = float(os.getenv("MIN_QUERY_COVERAGE", "0.4"))
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "lsa")


@dataclass
class Citation:
    chunk_id: str
    title: str
    section: str


@dataclass
class Answer:
    answer: str
    refused: bool
    reason: str | None
    citations: list[Citation] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    groundedness: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    redactions: dict[str, int] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def groundedness(answer: str, contexts: list[str]) -> float:
    """Share of answer content words that appear in the cited context (0-1)."""
    words = tokenize(_CITATION.sub(" ", answer))
    if not words:
        return 0.0
    vocab = set(tokenize(" ".join(contexts)))
    return sum(w in vocab for w in words) / len(words)


def query_coverage(question: str, hits) -> float:
    """Best share of question words found in any single retrieved chunk (0-1)."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    return max(len(q & set(tokenize(h.chunk.section + " " + h.chunk.text))) / len(q) for h in hits)


class RagPipeline:
    def __init__(self, settings: Settings | None = None, generator: Generator | None = None):
        self.settings = settings or Settings()
        chunks = load_corpus(self.settings.corpus_dir, self.settings.chunk_size, self.settings.chunk_overlap)
        self.by_id = {c.chunk_id: c for c in chunks}
        self.retriever = HybridRetriever(chunks, self.settings.embedding_backend)
        self.generator = generator or get_generator()

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        checked = check_input(question)

        def done(ans: Answer) -> Answer:
            ans.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            ans.redactions = checked.redactions
            logger.info(
                "question answered refused=%s reason=%s grounded=%.2f latency_ms=%.1f redactions=%s",
                ans.refused,
                ans.reason,
                ans.groundedness,
                ans.latency_ms,
                checked.redactions,
            )
            return ans

        if checked.injection_detected:
            return done(Answer(REFUSAL_INJECTION, True, "prompt_injection", model=self.generator.name))

        hits = self.retriever.search(checked.clean_text, top_k=self.settings.top_k)
        retrieved = [h.chunk.chunk_id for h in hits]
        if not hits or query_coverage(checked.clean_text, hits) < self.settings.min_query_coverage:
            return done(
                Answer(REFUSAL_NO_CONTEXT, True, "no_context", retrieved_ids=retrieved, model=self.generator.name)
            )

        try:
            draft = self.generator.generate(checked.clean_text, hits)
        except Exception:  # provider/network/parse failure -> safe refusal, never a crash
            logger.exception("Generator failed")
            return done(
                Answer(REFUSAL_NO_CONTEXT, True, "generator_error", retrieved_ids=retrieved, model=self.generator.name)
            )

        cited = [self.by_id[c] for c in draft.cited_ids if c in self.by_id]
        score = groundedness(draft.answer, [c.text for c in cited])
        if not draft.answer or not cited or score < self.settings.min_groundedness:
            return done(
                Answer(
                    REFUSAL_NO_CONTEXT,
                    True,
                    "low_groundedness",
                    retrieved_ids=retrieved,
                    groundedness=round(score, 3),
                    model=draft.model,
                )
            )

        return done(
            Answer(
                answer=draft.answer,
                refused=False,
                reason=None,
                citations=[Citation(c.chunk_id, c.title, c.section) for c in cited],
                retrieved_ids=retrieved,
                groundedness=round(score, 3),
                model=draft.model,
            )
        )
