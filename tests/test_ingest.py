import pytest

from rag.ingest import chunk_document, load_corpus

DOC = """---
title: Test doc
source: synthetic-educational
---
# Test doc

Intro paragraph about testing.

## Section A

""" + " ".join(f"word{i}" for i in range(150))


def test_front_matter_and_sections():
    chunks = chunk_document("t", DOC, chunk_size=80, chunk_overlap=20)
    assert chunks[0].title == "Test doc"
    assert {c.section for c in chunks} == {"Test doc", "Section A"}
    assert all(c.chunk_id.startswith("t#") for c in chunks)


def test_windows_overlap():
    chunks = [c for c in chunk_document("t", DOC, 80, 20) if c.section == "Section A"]
    # 150 words, window 80, step 60 -> windows start at 0, 60, 120
    assert len(chunks) == 3
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == {f"word{i}" for i in range(150)}


def test_invalid_window():
    with pytest.raises(ValueError):
        chunk_document("t", DOC, chunk_size=10, chunk_overlap=10)


def test_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")
    with pytest.raises(ValueError):
        load_corpus(tmp_path)
