"""SciSpaCy NER – deterministic biomedical entity extraction.

Runs before the LLM extraction pass to identify entities computationally,
providing grounding hints for the LLM extraction agent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import spacy

from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

# Lazy-loaded model
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_sci_sm")
    return _nlp


def extract_ner_entities(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run SciSpaCy NER on each chunk, return deduplicated entity list.

    Returns:
        List of dicts with keys: ``text``, ``label``, ``source_chunk``.
    """
    nlp = _get_nlp()
    entities: List[Dict[str, Any]] = []
    seen = set()

    for i, ch in enumerate(chunks):
        text = scrub_unicode(str(ch.get("text", "") or ""))
        if not text.strip():
            continue
        doc = nlp(text)
        for ent in doc.ents:
            ent_text = scrub_unicode(ent.text).strip()
            if not ent_text or len(ent_text) < 2:
                continue
            key = (ent_text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append(
                    {
                        "text": ent_text,
                        "label": ent.label_,
                        "source_chunk": i,
                    }
                )

    logger.info("SciSpaCy NER extracted %d unique entities from %d chunks", len(entities), len(chunks))
    return entities
