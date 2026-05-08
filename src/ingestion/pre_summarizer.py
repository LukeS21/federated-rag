"""Pre-summarization at ingest time — summarizes each chunk once during PDF
ingestion and stores the summary in chunk metadata.

At query time, pre‑written summaries are concatenated instead of running a
separate LLM call, eliminating the query‑time Summarize node.
"""

import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.unicode_map import sanitize_api_key, scrub_unicode

logger = logging.getLogger(__name__)


class PreSummarizer:
    """Summarizes individual chunks at ingest time and stores results in metadata.

    Uses DeepSeek Chat for cost efficiency — summarization fidelity matters less
    than generation since raw chunks remain available for evidence grounding.
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.0,
            api_key=sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")),
            base_url="https://api.deepseek.com/v1",
            max_tokens=200,
            timeout=120,
            default_headers={
                "User-Agent": "federated-rag",
                "Accept": "application/json",
            },
        )

    def summarize_chunk(self, chunk: Dict[str, Any]) -> str:
        """Produce a 1–2 sentence summary of a single chunk."""
        text = scrub_unicode(str(chunk.get("text", "") or ""))
        if len(text) < 100:
            return text  # Too short to summarize; use verbatim

        system_prompt = (
            "Summarize this biomedical text chunk in 1-2 sentences. "
            "Include specific entities, findings, and quantitative data. "
            "Output plain ASCII only."
        )
        user_prompt = f"Chunk text:\n{text}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = self._llm.invoke(messages)
            return scrub_unicode((response.content or "").strip())
        except Exception:
            logger.warning("Pre-summarization failed for a chunk; using raw text fallback.")
            return text[:200]  # Fallback: truncated raw text

    def summarize_all(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add 'chunk_summary' field to every chunk's metadata."""
        enriched = []
        for ch in chunks:
            summary = self.summarize_chunk(ch)
            meta = dict(ch.get("metadata", {}))
            meta["chunk_summary"] = summary
            enriched.append({**ch, "metadata": meta, "text": scrub_unicode(ch.get("text", "") or "")})
        logger.info("Pre-summarized %d chunks.", len(enriched))
        return enriched
