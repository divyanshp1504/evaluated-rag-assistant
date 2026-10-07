from rag.retrieval import BM25, tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("What is the SPF of my sunscreen?") == ["spf", "sunscreen"]
    assert tokenize("reapplied hours") == ["reapply", "hour"]


def test_paraphrase_regression(pipeline):
    # Found by the eval harness: 'reapply' vs 'reapplied' used to cause a false refusal.
    ans = pipeline.ask("How often should I reapply sunscreen?")
    assert not ans.refused and "two hours" in ans.answer


def test_bm25_prefers_matching_doc():
    bm = BM25([["sleep", "hours"], ["sunscreen", "spf"]])
    s = bm.scores(["spf"])
    assert s[1] > s[0] == 0


def test_hybrid_search_finds_expected_doc(pipeline):
    hits = pipeline.retriever.search("How often should I reapply sunscreen?")
    assert hits[0].chunk.doc_id == "sun_safety"
    assert hits[0].sparse_rank is not None and hits[0].dense_rank is not None
