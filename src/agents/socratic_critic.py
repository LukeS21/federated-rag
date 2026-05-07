"""Socratic Critic – evidence-grounded question generation.

Uses Gemma 4 26B A4B to resist peer-pressure convergence.
"""

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.unicode_map import scrub_unicode


class SocraticCritic:
    """Identifies unsupported claims without proposing alternative text.

    Model: Gemma 4 26B A4B (``gemma4:26b-a4b``).
    """

    def __init__(self, model_name: str = "gemma4:26b-a4b") -> None:
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def critique(
        self,
        draft: str,
        chunks: List[Dict[str, Any]],
        entities: Dict[str, Any],
    ) -> str:
        """Return a list of critiques or ``NO_CRITIQUE`` (README §5.2)."""
        system_prompt = (
            "You are a Socratic critic. Your job is to identify claims in the draft that lack "
            "sufficient evidence or overstate what the evidence supports.\n"
            "- For each questionable claim, state what the evidence actually says.\n"
            "- Ask a specific question about an unsupported assertion.\n"
            "- NEVER propose alternative text or \"correct\" the draft.\n"
            "- If the draft is fully supported, state: \"NO_CRITIQUE: All claims are evidence-grounded.\"\n"
            "Output plain ASCII only."
        )

        original_chunks = "\n\n".join(f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(chunks))
        extracted_entities = json.dumps(entities, indent=2, ensure_ascii=False)

        user_prompt = (
            f"Draft: {draft}\n"
            f"Evidence: {original_chunks}\n"
            f"Entities: {extracted_entities}\n"
            "Identify unsupported claims and state the gap."
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return scrub_unicode((response.content or "").strip())

