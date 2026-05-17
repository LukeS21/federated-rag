"""Relevance router — gates community access based on query relevance.

Uses a cheap model (OLLAMA_SMALL_MODEL) to determine which research
communities are relevant to a user query. Supports embedding-based
similarity as a fast primary path with LLM fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.llm import get_chat_model, resolve_model
from src.cache.llm_cache import get_cache
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

_EMBEDDER: Optional[SentenceTransformer] = None
_EMBEDDER_NAME = "all-MiniLM-L6-v2"

_ROUTING_SYSTEM = """You are a biomedical research router. Given a user query and a list of research \
community summaries, determine which communities are relevant to the query.

Rules:
- A community is relevant if it discusses concepts, methods, or findings mentioned in the query.
- Assign a relevance score from 0.0 (completely unrelated) to 1.0 (directly answers the query).
- Score >= 0.5 means the community should be included.
- Return a JSON object mapping community IDs to scores.
- Output ONLY the JSON object, no other text.

Example output:
{"0": 0.9, "1": 0.3, "2": 0.8}"""


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        logger.info("Loading embedding model: %s", _EMBEDDER_NAME)
        _EMBEDDER = SentenceTransformer(_EMBEDDER_NAME)
    return _EMBEDDER


class RelevanceRouter:
    """Routes queries to relevant KG communities.

    Primary path: embedding-based cosine similarity (fast, deterministic).
    Fallback path: LLM-based relevance scoring (when use_llm=True or
    embedding similarity is ambiguous).
    """

    EMBEDDING_THRESHOLD = 0.35
    LLM_CONFIRMATION_THRESHOLD = 0.2  # If max embedding sim is below this, use LLM

    def __init__(
        self,
        model: Optional[str] = None,
        use_llm: bool = False,
    ):
        self.model = resolve_model(model or "small")
        self.use_llm = use_llm

    def route(
        self,
        query: str,
        community_summaries: Dict[int, Dict[str, Any]],
        *,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Determine which communities are relevant to the query.

        Args:
            query: The user's research question.
            community_summaries: Output from ``CommunitySummarizer.summarize()``.
            threshold: Minimum similarity score to include a community.
                Defaults to ``EMBEDDING_THRESHOLD``.

        Returns:
            {
                "relevant_communities": [0, 2, 5],
                "scores": {0: 0.85, 1: 0.12, 2: 0.61, ...},
                "method": "embedding" | "llm",
                "threshold": 0.35,
            }
        """
        if not community_summaries:
            return {
                "relevant_communities": [],
                "scores": {},
                "method": "embedding",
                "threshold": threshold or self.EMBEDDING_THRESHOLD,
            }

        threshold = threshold or self.EMBEDDING_THRESHOLD

        if not self.use_llm:
            return self._route_by_embedding(query, community_summaries, threshold)

        return self._route_by_llm(query, community_summaries, threshold)

    def _route_by_embedding(
        self,
        query: str,
        community_summaries: Dict[int, Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, Any]:
        """Route by embedding similarity between query and community summaries."""
        embedder = _get_embedder()

        cids = sorted(community_summaries.keys())
        summary_texts = [community_summaries[c]["summary"] for c in cids]

        query_emb = embedder.encode([query], show_progress_bar=False)
        summary_embs = embedder.encode(summary_texts, show_progress_bar=False)

        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        summary_norms = summary_embs / (np.linalg.norm(summary_embs, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(query_norm, summary_norms.T).flatten()

        scores: Dict[int, float] = {}
        relevant: List[int] = []

        for i, cid in enumerate(cids):
            score = float(similarities[i])
            scores[cid] = round(score, 4)
            if score >= threshold:
                relevant.append(cid)

        max_sim = float(similarities.max()) if len(similarities) > 0 else 0.0

        method = "embedding"
        if max_sim < self.LLM_CONFIRMATION_THRESHOLD and not self.use_llm:
            method = "llm_fallback"

        logger.info(
            "Embedding routing: %d/%d communities relevant (max_sim=%.3f, threshold=%.2f)",
            len(relevant), len(cids), max_sim, threshold,
        )

        return {
            "relevant_communities": relevant,
            "scores": scores,
            "method": method,
            "threshold": threshold,
        }

    def _route_by_llm(
        self,
        query: str,
        community_summaries: Dict[int, Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, Any]:
        """Route by LLM relevance scoring."""
        summary_text = "\n\n".join(
            f"Community {cid}: {info['summary']}"
            for cid, info in sorted(community_summaries.items())
        )

        user_prompt = (
            f"User query: {query}\n\n"
            f"Research communities:\n{summary_text}\n\n"
            f"Return a JSON object mapping community IDs (as string keys) to relevance scores (0.0-1.0)."
        )

        cache = get_cache()
        cached = cache.get(_ROUTING_SYSTEM, user_prompt, model=self.model)
        if cached is not None:
            raw = scrub_unicode(cached)
        else:
            try:
                llm = get_chat_model(self.model, temperature=0.0, max_tokens=500)
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=_ROUTING_SYSTEM),
                    HumanMessage(content=user_prompt),
                ]
                response = llm.invoke(messages)
                raw = scrub_unicode((response.content or "").strip())
                cache.set(_ROUTING_SYSTEM, user_prompt, raw, model=self.model)
            except Exception as e:
                logger.error("LLM routing failed: %s — falling back to embedding", e)
                return self._route_by_embedding(query, community_summaries, threshold)

        return self._parse_routing(raw, list(community_summaries.keys()), threshold)

    def _parse_routing(
        self, raw: str, expected_cids: List[int], threshold: float
    ) -> Dict[str, Any]:
        """Parse LLM routing response into scores."""
        import json as _json

        try:
            text = raw.strip()
            for prefix in ("```json", "```"):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            l, r = text.find("{"), text.rfind("}")
            if l != -1 and r != -1 and r > l:
                text = text[l:r + 1]
            parsed = _json.loads(text)
        except (_json.JSONDecodeError, ValueError) as e:
            logger.warning("LLM routing parse failed: %s — falling back to embedding", e)
            return self._route_by_embedding(
                "", {cid: {"summary": raw[:200]} for cid in expected_cids}, threshold
            )

        scores: Dict[int, float] = {}
        relevant: List[int] = []
        for key, val in parsed.items():
            try:
                cid = int(key)
                score = float(val)
                scores[cid] = round(score, 4)
                if score >= threshold:
                    relevant.append(cid)
            except (ValueError, TypeError):
                continue

        logger.info("LLM routing: %d/%d communities relevant (threshold=%.2f)",
                      len(relevant), len(expected_cids), threshold)

        return {
            "relevant_communities": relevant,
            "scores": scores,
            "method": "llm",
            "threshold": threshold,
        }
