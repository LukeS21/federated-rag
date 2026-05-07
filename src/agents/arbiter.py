"""Arbiter – evidence-anchored revision of the draft."""

import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.unicode_map import scrub_unicode


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Arbiter:
    """Revises the draft only where the Critic identified genuine gaps.

    Model: Qwen3.6 35B‑A3B (different prompt than Drafter).
    """

    def __init__(
        self,
        model_name: str = "qwen3.6:35b",
        num_ctx: int = 8192,
        client_kwargs: dict | None = None,
        callback=None,
    ) -> None:
        if client_kwargs is None:
            client_kwargs = {}
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            num_ctx=num_ctx,
            client_kwargs=client_kwargs,
        )
        self.callback = callback

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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]
        response = self.llm.invoke(messages, config=config)
        raw = _strip_thinking((response.content or "").strip())
        return scrub_unicode(raw)

