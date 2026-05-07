"""Evidence anchoring check – programmatic verification of claims.

Implements README §5.3: decompose the synthesis into claims, retrieve best
evidence sentences via BM25, and compute cosine similarity (TF-IDF).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.bm25_index import BM25Index
from src.unicode_map import scrub_unicode


def decompose_claims(text: str) -> List[str]:
    """Split a synthesis paragraph into atomic factual statements.

    Uses simple sentence splitting and discards very short fragments.
    """

    clean_text = scrub_unicode(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)

    claims: List[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent.split()) < 5:
            continue
        claims.append(sent)
    return claims


def _split_chunks_into_sentences(chunks: Sequence[Dict[str, Any]]) -> List[str]:
    sentences: List[str] = []
    for chunk in chunks:
        text = scrub_unicode(str(chunk.get("text", "") or ""))
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def _extract_text(result: Any) -> str:
    """Best-effort extraction of evidence text from a BM25 result item."""

    if isinstance(result, str):
        return result
    if isinstance(result, tuple) and result:
        # e.g. (text, score)
        if isinstance(result[0], str):
            return result[0]
    if isinstance(result, dict):
        for k in ("text", "document", "content"):
            v = result.get(k)
            if isinstance(v, str):
                return v
    return str(result)


def compute_anchoring_score(
    claims: Sequence[str],
    chunks: Sequence[Dict[str, Any]],
    bm25_index: Optional[BM25Index] = None,
    threshold: float = 0.35,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute an anchoring score and the list of ungrounded claims.

    Each claim is grounded if its best-matching evidence sentence has TF-IDF
    cosine similarity >= threshold.

    Uses BM25 for candidate retrieval, then TF-IDF cosine for the similarity
    score (as specified in README §5.3).
    """

    claims_list = [scrub_unicode(c).strip() for c in claims if scrub_unicode(c).strip()]
    if not claims_list:
        return 1.0, []

    evidence_sentences = _split_chunks_into_sentences(chunks)
    if not evidence_sentences:
        return 0.0, [
            {"claim": c, "best_evidence_sentence": "", "similarity": 0.0} for c in claims_list
        ]

    # Build TF-IDF index over evidence sentences
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    evidence_matrix = vectorizer.fit_transform(evidence_sentences)

    bm25 = bm25_index or BM25Index()
    if bm25_index is None:
        bm25.add_documents(evidence_sentences)

    grounded = 0
    ungrounded: List[Dict[str, Any]] = []

    for claim in claims_list:
        results: Iterable[Any] = bm25.query(claim, n_results=3)
        results_list = list(results) if results is not None else []

        if not results_list:
            best_sentence = ""
            sim = 0.0
        else:
            # Compute TF-IDF cosine between the claim and each BM25 candidate,
            # take the highest score.
            claim_vec = vectorizer.transform([claim])
            sims = cosine_similarity(claim_vec, evidence_matrix)
            # Find the best match among the BM25-retrieved candidates
            best_sim = 0.0
            best_sentence = ""
            for result in results_list:
                cand_text = scrub_unicode(_extract_text(result)).strip()
                if cand_text in evidence_sentences:
                    idx = evidence_sentences.index(cand_text)
                    s = float(sims[0, idx])
                else:
                    cand_vec = vectorizer.transform([cand_text])
                    s = float(cosine_similarity(claim_vec, cand_vec)[0, 0])
                if s > best_sim:
                    best_sim = s
                    best_sentence = cand_text
            sim = best_sim

        if sim >= threshold:
            grounded += 1
        else:
            ungrounded.append(
                {
                    "claim": claim,
                    "best_evidence_sentence": best_sentence,
                    "similarity": round(sim, 4),
                }
            )

    return grounded / len(claims_list), ungrounded

