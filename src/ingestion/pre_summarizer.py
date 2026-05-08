"""Pre-summarization at ingest time — extracts key sentences from each chunk
using TF-IDF scoring and stores the summary in chunk metadata.

At query time, pre‑written summaries are concatenated instead of running a
separate LLM call, eliminating the query‑time Summarize node.

Uses TF-IDF extractive summarization (no LLM) for factual fidelity and speed.
"""

import logging
import re
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    raw = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _extractive_summary(text: str, max_sentences: int = 2) -> str:
    """Extract top sentences by TF-IDF importance score.

    Returns the top `max_sentences` most important sentences,
    preserving their original order in the text.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text[:200]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = tfidf_matrix.sum(axis=1).A1  # sum of TF-IDF values per sentence
    except ValueError:
        # Fallback if too few unique words
        return " ".join(sentences[:max_sentences])

    # Get indices of top-scoring sentences, then sort by original position
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:max_sentences]
    top_indices.sort()  # preserve original order

    return " ".join(sentences[i] for i in top_indices)


class PreSummarizer:
    """Summarizes individual chunks at ingest time using TF-IDF extraction.

    Extractive methods are more faithful to source text than LLM-generated
    summaries and eliminate hallucination risk. Summaries are stored in
    chunk metadata for use by downstream extraction and synthesis agents.
    """

    def __init__(self) -> None:
        pass

    def summarize_chunk(self, chunk: Dict[str, Any]) -> str:
        """Produce a short extractive summary of a single chunk."""
        text = scrub_unicode(str(chunk.get("text", "") or ""))
        if len(text) < 100:
            return text  # Too short to summarize; use verbatim
        return _extractive_summary(text, max_sentences=2)

    def summarize_all(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add 'chunk_summary' field to every chunk's metadata."""
        enriched = []
        for ch in chunks:
            summary = self.summarize_chunk(ch)
            meta = dict(ch.get("metadata", {}))
            meta["chunk_summary"] = summary
            enriched.append({
                **ch,
                "metadata": meta,
                "text": scrub_unicode(ch.get("text", "") or ""),
            })
        logger.info("Pre-summarized %d chunks (TF-IDF extractive).", len(enriched))
        return enriched
