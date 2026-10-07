"""Document loading and chunking.

Markdown files with a small YAML-style front matter block are split on
section headings, then into word windows with overlap. Each chunk keeps
metadata (doc id, title, section) so answers can cite their sources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Return (section heading, section text) pairs."""
    positions = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(body)]
    if not positions:
        return [("", body.strip())]
    sections = []
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        text = body[start:end].split("\n", 1)[1] if "\n" in body[start:end] else ""
        if text.strip():
            sections.append((heading, text.strip()))
    return sections


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= overlap:
        raise ValueError("chunk_size must be larger than chunk_overlap")
    if len(words) <= size:
        return [words]
    step = size - overlap
    return [words[i : i + size] for i in range(0, len(words) - overlap, step)]


def chunk_document(doc_id: str, raw: str, chunk_size: int = 80, chunk_overlap: int = 20) -> list[Chunk]:
    meta, body = _parse_front_matter(raw)
    title = meta.get("title", doc_id)
    chunks: list[Chunk] = []
    for section, text in _split_sections(body):
        for n, window in enumerate(_windows(text.split(), chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    title=title,
                    section=section or title,
                    text=" ".join(window),
                    metadata={**meta, "window": n},
                )
            )
    return chunks


def load_corpus(corpus_dir: str | Path, chunk_size: int = 80, chunk_overlap: int = 20) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), chunk_size, chunk_overlap))
    if not chunks:
        raise ValueError(f"No markdown documents found in {corpus_dir}")
    logger.info("Loaded %d chunks from %s", len(chunks), corpus_dir)
    return chunks
