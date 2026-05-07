from unittest.mock import patch

from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims


def test_decompose_simple_sentences():
    text = "This is claim one with enough words. Here is claim two with enough words! And claim three also has words?"
    claims = decompose_claims(text)
    assert len(claims) == 3
    assert claims[0].startswith("This is claim one")
    assert claims[2].startswith("And claim three")


def test_decompose_skips_short():
    text = "Yes. No. This is long enough to be a claim."
    claims = decompose_claims(text)
    assert len(claims) == 1
    assert "long enough" in claims[0].lower()


class MockBM25:
    def __init__(self):
        self.documents = []

    def add_documents(self, docs):
        self.documents = list(docs)

    def query(self, query, n_results=1):
        if not self.documents:
            return []
        # Deterministic "best match": pick the doc with max token overlap.
        q_tokens = set(query.lower().split())
        best_doc = max(
            self.documents,
            key=lambda d: len(q_tokens.intersection(set(str(d).lower().split()))),
        )
        return [best_doc]


def test_all_claims_grounded():
    claims = ["biomaterial surface modified with titanium", "obese mice had elevated IL-6 levels"]
    chunks = [
        {
            "text": "The biomaterial surface was modified with titanium. Obese mice had elevated IL-6 levels."
        },
    ]

    with patch("src.anchoring.evidence_check.BM25Index", new=MockBM25):
        score, ungrounded = compute_anchoring_score(claims, chunks, threshold=0.2)

    assert score == 1.0
    assert ungrounded == []


def test_some_claims_ungrounded():
    claims = ["X increases Y.", "Z decreases W."]
    chunks = [{"text": "X increases Y."}]

    class PartialMockBM25(MockBM25):
        def query(self, query, n_results=1):
            if "X" in query:
                return ["X increases Y."]
            return []

    with patch("src.anchoring.evidence_check.BM25Index", new=PartialMockBM25):
        score, ungrounded = compute_anchoring_score(claims, chunks, threshold=0.5)

    assert score == 0.5
    assert len(ungrounded) == 1
    assert ungrounded[0]["claim"] == "Z decreases W."
    assert ungrounded[0]["similarity"] == 0.0


def test_empty_chunks():
    claims = ["Something profound and specific happened here."]
    chunks = []
    score, ungrounded = compute_anchoring_score(claims, chunks)
    assert score == 0.0
    assert len(ungrounded) == 1
    assert ungrounded[0]["similarity"] == 0.0


def test_no_claims():
    score, ungrounded = compute_anchoring_score([], [{"text": "data"}])
    assert score == 1.0
    assert ungrounded == []

