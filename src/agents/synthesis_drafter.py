"""Synthesis Drafter – first-pass literature review synthesis."""

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.unicode_map import scrub_unicode


class SynthesisDrafter:
    """Writes an initial synthesis paragraph with inline citations.

    Uses Qwen3.6 35B‑A3B for its strong agentic tool‑use score.
    """

    def __init__(self, model_name: str = "qwen3.6:35b-a3b") -> None:
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def draft(
        self,
        query: str,
        entities: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        citations: List[str],
        kg_context: Dict[str, Any],
    ) -> str:
        """Produce a draft synthesis paragraph (README §5.2)."""
        system_prompt = (
            "You are a biomedical literature synthesis drafter. Given extracted entities, "
            "evidence summaries, and citation keys, write a concise literature review paragraph. "
            "Every factual claim must be traceable to a provided evidence chunk. "
            "Use inline citation keys (@author2025). Output plain ASCII only."
        )

        entities_json = json.dumps(entities, indent=2, ensure_ascii=False)
        chunk_texts = "\n\n".join(f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(chunks))
        cite_keys = ", ".join(citations) if citations else "none provided"
        subgraph_json = json.dumps(kg_context or {}, indent=2, ensure_ascii=False)

        user_prompt = (
            f"Query: {query}\n"
            f"Extracted Entities: {entities_json}\n"
            f"Evidence Summaries: {chunk_texts}\n"
            f"Available Citations: {cite_keys}\n"
            f"Knowledge Graph Context: {subgraph_json}\n"
            "Write a draft paragraph synthesizing this information."
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return scrub_unicode((response.content or "").strip())

