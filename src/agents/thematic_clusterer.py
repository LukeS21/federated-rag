"""Thematic Clustering Agent — assigns papers to 1+ themes for Survey Mode.

After query decomposition produces thematic sub-queries, this agent assigns
every paper in the corpus to the themes it addresses. No paper is excluded —
a single paper may belong to multiple themes.

Uses sentence-transformer embeddings for deterministic cosine-similarity
clustering. An LLM-based fallback is preserved for edge cases.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from src.cache.llm_cache import get_cache
from src.unicode_map import sanitize_api_key, scrub_unicode

logger = logging.getLogger(__name__)

# Load embedding model once (lazy, cached at module level)
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.35


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model: %s", _EMBEDDING_MODEL_NAME)
        _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


class ThematicClusterer:
    """Assigns papers to thematic clusters based on their content summaries.

    Default method: embedding-based cosine similarity (deterministic, fast).
    Fallback method: LLM-based classification (when use_embeddings=False).
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        use_embeddings: bool = True,
    ) -> None:
        self.model = model
        self.use_embeddings = use_embeddings
        if not use_embeddings:
            self._llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                api_key=sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")),
                base_url="https://api.deepseek.com/v1",
                max_tokens=4096,
                timeout=120,
                default_headers={
                    "User-Agent": "federated-rag",
                    "Accept": "application/json",
                },
            )
        else:
            self._llm = None

    def cluster(
        self,
        papers: List[Dict[str, Any]],
        themes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Assign every paper to one or more themes.

        Args:
            papers: List of paper dicts, each with ``id``, ``title``, ``summary``.
            themes: List of theme dicts, each with ``theme`` and ``sub_query``.

        Returns:
            Dict with ``clusters`` (theme→paper_ids), ``paper_assignments``
            (paper_id→themes), and ``unassigned`` (paper_ids with no theme).
        """
        if not papers:
            return {"clusters": {}, "paper_assignments": {}, "unassigned": []}
        if not themes:
            return {
                "clusters": {"unthemed": [p.get("id", str(i)) for i, p in enumerate(papers)]},
                "paper_assignments": {p.get("id", str(i)): ["unthemed"] for i, p in enumerate(papers)},
                "unassigned": [],
            }

        if self.use_embeddings:
            raw = self._cluster_by_embeddings(papers, themes)
        else:
            raw = self._cluster_by_llm(papers, themes)

        clusters = raw.get("assignments", {})
        unassigned = raw.get("unassigned", [])

        paper_assignments: Dict[str, List[str]] = {}
        for p in papers:
            pid = p.get("id") or p.get("title") or "unknown"
            paper_assignments[pid] = []
        for theme_name, paper_ids in clusters.items():
            if isinstance(paper_ids, list):
                for pid in paper_ids:
                    pid_str = str(pid)
                    if pid_str not in paper_assignments:
                        paper_assignments[pid_str] = []
                    paper_assignments[pid_str].append(theme_name)

        return {
            "clusters": clusters,
            "paper_assignments": paper_assignments,
            "unassigned": unassigned,
        }

    # ------------------------------------------------------------------
    #  Embedding-based clustering (primary method)
    # ------------------------------------------------------------------
    def _cluster_by_embeddings(
        self,
        papers: List[Dict[str, Any]],
        themes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        from src.ingestion.pre_extractor import PreExtractor

        embedder = _get_embedder()
        threshold = _SIMILARITY_THRESHOLD

        paper_ids = [p.get("id") or p.get("title") or f"paper_{i}"
                      for i, p in enumerate(papers)]
        paper_texts = [p.get("summary", "")[:2000] for p in papers]

        # Try to load pre-computed embeddings from ingest cache.
        # Papers without cached embeddings are encoded on the fly.
        cached_embeddings = PreExtractor.load_all_embeddings()
        paper_embeddings: List[np.ndarray] = []
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        missing_ids: List[str] = []

        for i, pid in enumerate(paper_ids):
            cached = cached_embeddings.get(pid)
            if cached is not None:
                paper_embeddings.append(cached)
            else:
                paper_embeddings.append(np.zeros(0))  # placeholder
                missing_indices.append(i)
                missing_texts.append(paper_texts[i])
                missing_ids.append(pid)

        # Encode any papers missing from the cache
        if missing_texts:
            logger.debug("Embedding %d uncached papers...", len(missing_texts))
            new_embeddings = embedder.encode(missing_texts, show_progress_bar=False)
            for j, idx in enumerate(missing_indices):
                paper_embeddings[idx] = new_embeddings[j]
                # Cache for future queries
                try:
                    PreExtractor.save_embedding(missing_ids[j], new_embeddings[j])
                    logger.debug("  cached embedding for %s", missing_ids[j])
                except Exception:
                    pass

        paper_embeddings = np.array(paper_embeddings)

        # Embed theme descriptions (these change per query — can't cache)
        theme_names = [t["theme"] for t in themes]
        theme_texts = [f"{t['theme']}: {t.get('sub_query', '')}" for t in themes]
        theme_embeddings = embedder.encode(theme_texts, show_progress_bar=False)

        # Compute cosine similarity (paper × theme matrix)
        paper_norms = np.linalg.norm(paper_embeddings, axis=1, keepdims=True) + 1e-10
        theme_norms = np.linalg.norm(theme_embeddings, axis=1, keepdims=True) + 1e-10
        paper_emb_norm = paper_embeddings / paper_norms
        theme_emb_norm = theme_embeddings / theme_norms
        sim_matrix = np.dot(paper_emb_norm, theme_emb_norm.T)

        # Assign papers to themes above threshold
        assignments: Dict[str, list] = {name: [] for name in theme_names}
        assigned_papers: set = set()
        for i, pid in enumerate(paper_ids):
            for j, tname in enumerate(theme_names):
                if sim_matrix[i, j] >= threshold:
                    assignments[tname].append(pid)
                    assigned_papers.add(pid)

        unassigned = [pid for pid in paper_ids if pid not in assigned_papers]

        logger.info(
            "Embedding clustering: %d themes, %d papers, threshold=%.2f, %d unassigned",
            len(theme_names), len(papers), threshold, len(unassigned),
        )

        return {"assignments": assignments, "unassigned": unassigned}

    # ------------------------------------------------------------------
    #  LLM-based clustering (legacy fallback)
    # ------------------------------------------------------------------
    def _cluster_by_llm(
        self,
        papers: List[Dict[str, Any]],
        themes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        paper_lines = []
        for i, p in enumerate(papers):
            pid = p.get("id", p.get("title", f"paper_{i}"))
            title = p.get("title", pid)
            summary = p.get("summary", "")[:500]
            paper_lines.append(f"Paper ID: {pid}\n  Title: {title}\n  Summary: {summary}")
        papers_text = "\n\n".join(paper_lines)

        theme_lines = [f"- {t['theme']}: {t.get('sub_query', '')}" for t in themes]
        themes_text = "\n".join(theme_lines)

        system_prompt = (
            "You are a biomedical literature classifier. Given a set of papers and a set "
            "of thematic categories, assign each paper to ALL themes that it meaningfully "
            "addresses. A single paper may belong to multiple themes.\n\n"
            "Rules:\n"
            "- Read each paper's summary and decide which theme(s) it contributes to.\n"
            "- A paper can belong to 0, 1, or many themes.\n"
            "- Only assign a paper to a theme if the paper contains substantial content "
            "relevant to that theme.\n"
            "- CRITICAL: Use the EXACT Paper ID shown above (e.g., 'test.pdf'), not a number.\n"
            "- Output ONLY a JSON object with the structure below. No other text.\n\n"
            '{"assignments": {"theme_name_1": ["Paper_ID_1", "Paper_ID_2"]}, '
            '"unassigned": ["Paper_ID_3"]}\n\n'
            "Use the exact Paper IDs provided. Plain ASCII only."
        )

        user_prompt = (
            f"Themes:\n{themes_text}\n\n"
            f"Papers:\n{papers_text}\n\n"
            "Assign each paper to ALL themes that it addresses. Output ONLY the JSON object."
        )

        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt, model=self.model)
        if cached is not None:
            raw = scrub_unicode(cached)
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            if self._llm is None:
                self._llm = ChatOpenAI(
                    model=self.model,
                    temperature=0.0,
                    api_key=sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")),
                    base_url="https://api.deepseek.com/v1",
                    max_tokens=4096,
                    timeout=120,
                    default_headers={
                        "User-Agent": "federated-rag",
                        "Accept": "application/json",
                    },
                )
            response = self._llm.invoke(messages)
            raw = scrub_unicode((response.content or "").strip())
            cache.set(system_prompt, user_prompt, raw, model=self.model)

        return self._parse_json(raw)

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if "```" in text:
            for segment in text.split("```"):
                seg = segment.strip()
                if seg.lower().startswith("json"):
                    seg = seg[4:].lstrip()
                if seg.startswith("{") or seg.startswith("["):
                    text = seg
                    break
        l, r = text.find("{"), text.rfind("}")
        if l != -1 and r != -1 and r > l:
            text = text[l : r + 1]
        return json.loads(text)
