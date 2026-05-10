"""Evidence anchoring check – programmatic verification of claims.

Implements README §5.3: decompose the synthesis into claims, retrieve best
evidence sentences via hybrid retrieval (BM25 sparse + ChromaDB dense with RRF),
and compute cosine similarity (TF-IDF).

The hybrid retrieval step mirrors the main pipeline's ``HybridRetriever``:
BM25 provides exact keyword precision (gene names, alloy codes, PMIDs) while
ChromaDB dense embeddings provide semantic matching (synonyms, paraphrases).
RRF fusion naturally deduplicates the two result lists.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.bm25_index import BM25Index
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

# Module-level ChromaClient singleton for hybrid retrieval.
# Set via set_anchoring_chroma() at pipeline startup, benchmark init, etc.
# When None (default), anchoring falls back to BM25-only — safe for tests
# that don't have a running ChromaDB.
_anchoring_chroma: Any = None


def set_anchoring_chroma(client: Any) -> None:
    """Set the ChromaClient singleton used by compute_anchoring_score."""
    global _anchoring_chroma
    _anchoring_chroma = client


def clear_anchoring_chroma() -> None:
    """Clear the ChromaClient singleton (for testing)."""
    global _anchoring_chroma
    _anchoring_chroma = None


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
    chroma_client: Any = None,  # Optional[ChromaClient] — lazy import to avoid circular dep
    threshold: float = 0.35,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute an anchoring score and the list of ungrounded claims.

    Each claim is grounded if its best-matching evidence sentence has TF-IDF
    cosine similarity >= threshold.

    Uses hybrid retrieval for candidate selection when a ChromaClient is
    available (via *chroma_client* parameter or the module-level singleton
    set by ``set_anchoring_chroma()``).  BM25 sparse (exact keyword) +
    ChromaDB dense (semantic) results are fused via simple candidate pooling.
    Falls back to BM25-only when no ChromaClient is available (e.g. in tests
    that don't have a running ChromaDB).
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

    # Pre-build a sentence→chunk index for fast chunk→sentence lookup.
    # Maps each sentence back to its source chunk index.
    sentence_chunk_map: List[int] = []
    for ci, chunk in enumerate(chunks):
        text = scrub_unicode(str(chunk.get("text", "") or ""))
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        sentence_chunk_map.extend([ci] * len(sents))

    grounded = 0
    ungrounded: List[Dict[str, Any]] = []

    for claim in claims_list:
        # ── BM25 sparse retrieval ──
        results: Iterable[Any] = bm25.query(claim, n_results=5)
        results_list = list(results) if results is not None else []

        # ── ChromaDB dense retrieval (if available) ──
        _chroma = chroma_client or _anchoring_chroma
        if _chroma is not None and len(evidence_sentences) > 10:
            try:
                dense_results = _chroma.query(claim, n_results=3)
                # ChromaDB returns {"ids": [[...]], "documents": [[...]], ...}
                dense_docs = (dense_results or {}).get("documents", [[]])
                dense_texts = dense_docs[0] if dense_docs else []
                for dr_text in dense_texts:
                    dr_text = scrub_unicode(str(dr_text))
                    if not dr_text:
                        continue
                    # Find the sentence within this chunk that best matches the claim
                    chunk_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", dr_text) if s.strip()]
                    if not chunk_sents:
                        continue
                    try:
                        chunk_sent_vecs = vectorizer.transform(chunk_sents)
                        claim_vec = vectorizer.transform([claim])
                        best_sent_sim = cosine_similarity(claim_vec, chunk_sent_vecs)[0]
                        best_sent_idx = int(best_sent_sim.argmax())
                        best_sent = chunk_sents[best_sent_idx]
                        # Avoid duplicates — only add if not already in BM25 results
                        existing = [scrub_unicode(_extract_text(r)) for r in results_list]
                        if best_sent not in existing:
                            results_list.append(best_sent)
                    except Exception:
                        continue
            except Exception:
                logger.debug("ChromaDB dense retrieval unavailable for anchoring", exc_info=True)

        if not results_list:
            best_sentence = ""
            sim = 0.0
        else:
            claim_vec = vectorizer.transform([claim])
            sims = cosine_similarity(claim_vec, evidence_matrix)
            best_sim = 0.0
            best_sentence = ""

            # Score each candidate (BM25 sentences + dense-chunk best sentences)
            candidates = []
            for result in results_list:
                cand_text = scrub_unicode(_extract_text(result)).strip()
                if not cand_text:
                    continue
                candidates.append(cand_text)

            # Deduplicate
            seen: set = set()
            unique_candidates = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            for cand_text in unique_candidates:
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

