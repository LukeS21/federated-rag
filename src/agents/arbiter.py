"""Arbiter – evidence-anchored revision of the draft."""

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.unicode_map import scrub_unicode


class Arbiter:
    """Revises the draft only where the Critic identified genuine gaps.

    Model: Qwen3.6 35B‑A3B (different prompt than Drafter).
    """

    def __init__(self, model_name: str = "qwen3.6:35b-a3b") -> None:
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def revise(
        self,
        draft: str,
        critique: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Produce a revised synthesis paragraph (README §5.2)."""
        system_prompt = (
            "You are a biomedical synthesis arbiter. You receive a draft, a Socratic critique, "
            "and the original evidence. Revise the draft to address the critique.\n"
            "- For each critique, either cite specific evidence that supports the claim or modify/remove the claim.\n"
            "- Do not alter claims that were not critiqued.\n"
            "- Output the complete revised paragraph. Plain ASCII only."
        )

        original_chunks = "\n\n".join(f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(chunks))

        user_prompt = (
            f"Draft: {draft}\n"
            f"Critique: {critique}\n"
            f"Evidence: {original_chunks}\n"
            "Revise the draft, addressing each critique."
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return scrub_unicode((response.content or "").strip())

